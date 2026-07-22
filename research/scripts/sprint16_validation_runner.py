import os
import sys
import json
import time
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss, log_loss

# Resolve paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from training.dataset import get_dataloaders
from models.deck_encoder import DeckEncoder
from models.interaction_head import BradleyTerryHead, SkillDifference
from models.predictor import MatchupModel
from training.trainer import Trainer, CheckpointManager, EarlyStopping, MetricsLogger
from training.calibration import evaluate_calibration

def train_and_eval(seed: int, batch_size: int = 128, init_type: str = "xavier_uniform") -> tuple:
    """Helper to train the MatchupModel with a given seed and return test predictions and metrics."""
    with open(PROJECT_ROOT / "config" / "dataset_config.yaml", "r") as f:
        ds_config = yaml.safe_load(f)
    with open(PROJECT_ROOT / "config" / "deck_encoder_config.yaml", "r") as f:
        m_config = yaml.safe_load(f)
        
    # Configure model config parameters
    m_config["model"]["init_type"] = init_type
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load dataset
    df = pd.read_parquet(PROJECT_ROOT / ds_config["data"]["dataset_path"])
    with open(PROJECT_ROOT / ds_config["data"]["card_library_path"], "r") as f:
        cards_lib = json.load(f)
    num_cards = len(cards_lib)
    
    train_ldr, val_ldr, test_ldr = get_dataloaders(
        df=df,
        card_library_path=PROJECT_ROOT / ds_config["data"]["card_library_path"],
        batch_size=batch_size,
        seed=seed,
        augment_prob=0.0 # No augment for validation reliability
    )
    
    # Build model components
    encoder = DeckEncoder(
        num_cards=num_cards,
        embedding_dim=m_config["model"]["embedding_dim"],
        pooling_type=m_config["model"]["pooling_type"],
        hidden_dim=m_config["model"]["hidden_dim"],
        projection_dim=m_config["model"]["projection_dim"],
        dropout=m_config["model"]["dropout"],
        use_layernorm=m_config["model"]["use_layernorm"],
        init_type=m_config["model"]["init_type"]
    )
    head = BradleyTerryHead(projection_dim=m_config["model"]["projection_dim"])
    skill = SkillDifference(hidden_dim=m_config["skill"]["hidden_dim"])
    model = MatchupModel(encoder, head, skill)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)
    
    temp_dir = PROJECT_ROOT / "models" / f"temp_validate_seed_{seed}"
    c_manager = CheckpointManager(temp_dir)
    logger = MetricsLogger(temp_dir / "metrics.csv", temp_dir / "metrics.json")
    early_stop = EarlyStopping(patience=2, min_delta=0.001)
    
    trainer = Trainer(
        model=model,
        train_loader=train_ldr,
        val_loader=val_ldr,
        criterion=nn.BCEWithLogitsLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_manager=c_manager,
        logger=logger,
        early_stopping=early_stop,
        config=m_config,
        seed=seed
    )
    
    # Train for 4 epochs (fast and sufficient for sweep convergence)
    trainer.fit(num_epochs=4)
    
    # Evaluate on test set
    model.eval()
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for batch in test_ldr:
            p1 = batch["p1_deck"]
            p2 = batch["p2_deck"]
            td = batch["trophy_diff"]
            target = batch["target"]
            probs = model.predict_proba(p1, p2, td)
            all_targets.append(target.numpy())
            all_probs.append(probs.numpy())
            
    y_true = np.vstack(all_targets).squeeze()
    y_prob = np.vstack(all_probs).squeeze()
    
    acc = accuracy_score(y_true, (y_prob >= 0.5).astype(float))
    auc = roc_auc_score(y_true, y_prob)
    ll = log_loss(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = evaluate_calibration(y_true, y_prob)["ece"]
    
    # Extract card embeddings weights
    embeddings = model.encoder.embeddings.weight.detach().numpy()
    
    # Cleanup temp checkpoints
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except:
        pass
        
    return {
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "log_loss": float(ll),
        "brier_score": float(brier),
        "ece": float(ece)
    }, y_prob, y_true, embeddings

def get_baseline_predictions(df, card_library_path) -> tuple:
    """Builds and fits baseline Logistic Regression on card presence vectors."""
    with open(card_library_path, "r") as f:
        cards_lib = json.load(f)
    num_cards = len(cards_lib)
    sorted_card_ids = sorted(list(cards_lib.keys()))
    card_to_idx = {cid: idx for idx, cid in enumerate(sorted_card_ids)}
    
    # Build presence matrices
    def to_features(row):
        feat = np.zeros(num_cards * 2 + 1)
        p1_deck = json.loads(row["player_deck"])
        p2_deck = json.loads(row["opponent_deck"])
        for c in p1_deck:
            if str(c) in card_to_idx:
                feat[card_to_idx[str(c)]] = 1.0
        for c in p2_deck:
            if str(c) in card_to_idx:
                feat[num_cards + card_to_idx[str(c)]] = 1.0
        p1_trophies = row.get("player_trophies", 0.0)
        p2_trophies = row.get("opponent_trophies", 0.0)
        feat[-1] = (p1_trophies - p2_trophies) / 1000.0
        return feat
        
    features = np.stack(df.apply(to_features, axis=1).values)
    targets = df["win"].values
    
    # Split
    n_train = int(len(df) * 0.7)
    n_val = int(len(df) * 0.15)
    X_train, y_train = features[:n_train], targets[:n_train]
    X_test, y_test = features[n_train + n_val:], targets[n_train + n_val:]
    
    lr = LogisticRegression(max_iter=100)
    lr.fit(X_train, y_train)
    probs = lr.predict_proba(X_test)[:, 1]
    return probs, y_test

def get_nearest_neighbors(embeddings, top_k=3) -> dict:
    """Calculates top-k nearest neighbors based on cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embeddings = embeddings / (norms + 1e-8)
    sim_matrix = np.dot(norm_embeddings, norm_embeddings.T)
    
    neighbors = {}
    for i in range(len(embeddings)):
        # Sort similarities (exclude self at index i)
        sims = sim_matrix[i]
        sorted_indices = np.argsort(sims)[::-1]
        sorted_indices = [idx for idx in sorted_indices if idx != i]
        neighbors[i] = sorted_indices[:top_k]
    return neighbors

def main():
    print("=" * 60)
    print("   STARTING SPRINT 16 VALIDATION & ROBUSTNESS ENGINE")
    print("=" * 60)
    
    with open(PROJECT_ROOT / "config" / "dataset_config.yaml", "r") as f:
        ds_config = yaml.safe_load(f)
        
    df = pd.read_parquet(PROJECT_ROOT / ds_config["data"]["dataset_path"])
    card_lib_path = PROJECT_ROOT / ds_config["data"]["card_library_path"]
    
    # Baseline LR
    baseline_probs, y_test_lr = get_baseline_predictions(df, card_lib_path)
    baseline_errors = (y_test_lr - baseline_probs) ** 2
    
    # Stage 1 & 2: Repeated Seeds (10 runs)
    seeds = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    metrics_list = []
    preds_list = []
    embeddings_list = []
    
    print("\n--- Running Stage 1 & 2: 10-Seed MatchupModel Training & Paired t-tests ---")
    for s in seeds:
        print(f"  • Fitting random seed = {s} ...")
        metrics, probs, y_true, embeddings = train_and_eval(s)
        metrics_list.append(metrics)
        preds_list.append(probs)
        embeddings_list.append(embeddings)
        
        # Paired t-test
        m_errors = (y_true - probs) ** 2
        t_stat, p_val = ttest_rel(baseline_errors, m_errors)
        print(f"    => Acc: {metrics['accuracy']:.4f} | ECE: {metrics['ece']:.4f} | Paired t-test p-value: {p_val:.4e}")
        
    # Compile Stats
    df_metrics = pd.DataFrame(metrics_list)
    stats_summary = {
        col: {
            "mean": float(df_metrics[col].mean()),
            "std": float(df_metrics[col].std()),
            "min": float(df_metrics[col].min()),
            "max": float(df_metrics[col].max())
        } for col in df_metrics.columns
    }
    
    # Stage 3: Embedding Stability (Seed 42 vs Seed 100)
    print("\n--- Running Stage 3: Card Embedding Stability (Seed 42 vs 100) ---")
    emb1 = embeddings_list[0]
    emb2 = embeddings_list[1]
    
    # Cosine similarities
    norms1 = np.linalg.norm(emb1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(emb2, axis=1, keepdims=True)
    sims = np.sum((emb1 / (norms1 + 1e-8)) * (emb2 / (norms2 + 1e-8)), axis=1)
    avg_cosine = float(np.mean(sims))
    
    # Nearest neighbor Jaccard stability
    nn1 = get_nearest_neighbors(emb1)
    nn2 = get_nearest_neighbors(emb2)
    jaccards = []
    for i in range(len(emb1)):
        set1 = set(nn1[i])
        set2 = set(nn2[i])
        jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
        jaccards.append(jaccard)
    avg_jaccard = float(np.mean(jaccards))
    
    print(f"  • Mean Cosine Similarity: {avg_cosine:.4f}")
    print(f"  • Top-3 Nearest-Neighbors Jaccard Overlap: {avg_jaccard:.4f}")
    
    # Stage 4: Robustness Sweeps (Batch size & Initialization)
    print("\n--- Running Stage 4: Robustness Sweeps ---")
    batch_sizes = [128, 256, 512]
    batch_results = {}
    for bs in batch_sizes:
        print(f"  • Sweeping Batch Size = {bs} (seed=42) ...")
        res, _, _, _ = train_and_eval(seed=42, batch_size=bs)
        batch_results[str(bs)] = res
        print(f"    => Acc: {res['accuracy']:.4f}")
        
    init_types = ["xavier_uniform", "xavier_normal"]
    init_results = {}
    for it in init_types:
        print(f"  • Sweeping Initialization = {it} (seed=42) ...")
        res, _, _, _ = train_and_eval(seed=42, init_type=it)
        init_results[it] = res
        print(f"    => Acc: {res['accuracy']:.4f}")

    # Stage 5: Error Analysis Categorization
    print("\n--- Running Stage 5: Error Analysis Bins (Best Run: Seed 42) ---")
    best_probs = preds_list[0]
    n_train = int(len(df) * 0.7)
    n_val = int(len(df) * 0.15)
    test_df = df.iloc[n_train + n_val:].copy()
    test_df["y_prob"] = best_probs
    test_df["y_pred"] = (best_probs >= 0.5).astype(int)
    
    # Bins
    # 1. Extreme skill
    test_df["trophy_diff"] = test_df["player_trophies"] - test_df["opponent_trophies"]
    extreme_skill = test_df[test_df["trophy_diff"].abs() > 800]
    acc_ext_skill = accuracy_score(extreme_skill["win"], extreme_skill["y_pred"]) if len(extreme_skill) > 0 else 0.0
    
    # 2. Mirror Matchups (sharing >= 6 cards in p1 and p2)
    def count_overlap(row):
        p1_set = set(json.loads(row["player_deck"]))
        p2_set = set(json.loads(row["opponent_deck"]))
        return len(p1_set.intersection(p2_set))
    test_df["card_overlap"] = test_df.apply(count_overlap, axis=1)
    mirror_matches = test_df[test_df["card_overlap"] >= 6]
    acc_mirror = accuracy_score(mirror_matches["win"], mirror_matches["y_pred"]) if len(mirror_matches) > 0 else 0.0
    
    # 3. Rare card decks (contains bottom 20% cards by presence in df)
    all_cards_played = pd.Series([c for sub in df["player_deck"].apply(json.loads).values for c in sub])
    card_counts = all_cards_played.value_counts()
    bottom_20_threshold = card_counts.quantile(0.20)
    rare_cards = set(card_counts[card_counts <= bottom_20_threshold].index)
    
    def has_rare_card(row):
        p1_deck = json.loads(row["player_deck"])
        p2_deck = json.loads(row["opponent_deck"])
        return any(c in rare_cards for c in p1_deck + p2_deck)
        
    test_df["has_rare"] = test_df.apply(has_rare_card, axis=1)
    rare_decks_matches = test_df[test_df["has_rare"] == True]
    acc_rare = accuracy_score(rare_decks_matches["win"], rare_decks_matches["y_pred"]) if len(rare_decks_matches) > 0 else 0.0
    
    print(f"  • Extreme Skill (>800 Trophies): Count = {len(extreme_skill)} | Acc = {acc_ext_skill:.4f}")
    print(f"  • Mirror Matchups (>=6 Overlaps): Count = {len(mirror_matches)} | Acc = {acc_mirror:.4f}")
    print(f"  • Rare/Off-Meta Decks: Count = {len(rare_decks_matches)} | Acc = {acc_rare:.4f}")
    
    # Output logs
    out_results = {
        "seeds_summary": stats_summary,
        "seed_runs": metrics_list,
        "embedding_stability": {
            "mean_cosine_similarity": avg_cosine,
            "mean_jaccard_similarity": avg_jaccard
        },
        "robustness_sweeps": {
            "batch_sizes": batch_results,
            "initializers": init_results
        },
        "error_analysis_bins": {
            "extreme_skill": {"count": len(extreme_skill), "accuracy": float(acc_ext_skill)},
            "mirror_matches": {"count": len(mirror_matches), "accuracy": float(acc_mirror)},
            "rare_decks": {"count": len(rare_decks_matches), "accuracy": float(acc_rare)}
        }
    }
    
    out_path = PROJECT_ROOT / "logs" / "sprint16_validation.json"
    with open(out_path, "w") as f:
        json.dump(out_results, f, indent=4)
        
    print(f"\n[+] Sprint 16 validation completed! Logging locked in: {out_path}\n")

if __name__ == "__main__":
    main()
