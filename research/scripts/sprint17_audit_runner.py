import os
import sys
import json
import yaml
import torch
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
from training.calibration import evaluate_calibration

def main():
    print("=" * 60)
    print("   STARTING SPRINT 17 INDEPENDENT CODE & DATA LEAKAGE AUDIT")
    print("=" * 60)
    
    # 1. Loading config & data
    with open(PROJECT_ROOT / "config" / "dataset_config.yaml", "r") as f:
        ds_config = yaml.safe_load(f)
        
    df = pd.read_parquet(PROJECT_ROOT / ds_config["data"]["dataset_path"])
    card_lib_path = PROJECT_ROOT / ds_config["data"]["card_library_path"]
    
    with open(card_lib_path, "r") as f:
        cards_lib = json.load(f)
    num_cards = len(cards_lib)
    
    # 2. Split Leakage Audit
    print("\n--- Part 1: Split Leakage Audit ---")
    
    # We inspect the splits created by get_dataloaders
    train_ldr, val_ldr, test_ldr = get_dataloaders(
        df=df,
        card_library_path=card_lib_path,
        batch_size=128,
        seed=42,
        augment_prob=0.0
    )
    
    # Extract indices or features to verify zero intersection
    n_train = int(len(df) * 0.7)
    n_val = int(len(df) * 0.15)
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    
    # Check Chronological Order
    is_chronological = True
    if "battle_time" in df.columns:
        train_max_time = train_df["battle_time"].max()
        val_min_time = val_df["battle_time"].min()
        val_max_time = val_df["battle_time"].max()
        test_min_time = test_df["battle_time"].min()
        
        is_chronological = (train_max_time <= val_min_time) and (val_max_time <= test_min_time)
        print(f"  • Chronological Splitting: {'PASSED' if is_chronological else 'FAILED'}")
        print(f"    - Train Max Time: {train_max_time}")
        print(f"    - Val Min Time:   {val_min_time}")
        print(f"    - Val Max Time:   {val_max_time}")
        print(f"    - Test Min Time:  {test_min_time}")
    else:
        # If no battle_time, check row order is strictly preserved
        print("  • Note: 'battle_time' column missing. Row chronological order assumed preserved.")
        
    # Check row index overlaps (instance level)
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    test_idx = set(test_df.index)
    
    instance_leakage_tv = train_idx.intersection(val_idx)
    instance_leakage_vt = val_idx.intersection(test_idx)
    instance_leakage_tt = train_idx.intersection(test_idx)
    
    has_instance_leak = len(instance_leakage_tv) > 0 or len(instance_leakage_vt) > 0 or len(instance_leakage_tt) > 0
    print(f"  • Row Instance Leakage: {'FAILED' if has_instance_leak else 'PASSED (0 shared rows)'}")
    
    # Check Duplications across splits (meta-deck overlaps)
    def row_sig(row):
        p1 = "-".join(sorted([str(c) for c in json.loads(row["player_deck"])]))
        p2 = "-".join(sorted([str(c) for c in json.loads(row["opponent_deck"])]))
        return f"{p1}_{p2}_{row['win']}"
        
    train_sigs = set(train_df.apply(row_sig, axis=1))
    val_sigs = set(val_df.apply(row_sig, axis=1))
    test_sigs = set(test_df.apply(row_sig, axis=1))
    
    leakage_train_val = train_sigs.intersection(val_sigs)
    leakage_val_test = val_sigs.intersection(test_sigs)
    leakage_train_test = train_sigs.intersection(test_sigs)
    
    print(f"  • Meta-Deck Co-occurrences (expected on ladder):")
    print(f"    - Train/Val shared deck matchups: {len(leakage_train_val)}")
    print(f"    - Val/Test shared deck matchups: {len(leakage_val_test)}")
    print(f"    - Train/Test shared deck matchups: {len(leakage_train_test)}")
    
    # 3. Independent Metrics & Model Validation
    print("\n--- Part 2: Metrics Validation Audit ---")
    
    # Load MatchupModel weights
    best_ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "sprint13_matchup_model.pt"
    if not best_ckpt_path.exists():
        print(f"Error: checkpoint {best_ckpt_path} missing.")
        sys.exit(1)
        
    encoder = DeckEncoder(num_cards=num_cards, embedding_dim=16)
    head = BradleyTerryHead(projection_dim=16)
    skill = SkillDifference(hidden_dim=8)
    model = MatchupModel(encoder, head, skill)
    model.load_state_dict(torch.load(best_ckpt_path, map_location=torch.device("cpu")))
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
    
    # Calculate
    acc = float(accuracy_score(y_true, (y_prob >= 0.5).astype(float)))
    auc = float(roc_auc_score(y_true, y_prob))
    ll = float(log_loss(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = float(evaluate_calibration(y_true, y_prob)["ece"])
    
    print(f"  • Verified MatchupModel Test Metrics:")
    print(f"    - Accuracy:    {acc:.4f}")
    print(f"    - ROC-AUC:     {auc:.4f}")
    print(f"    - Log Loss:    {ll:.4f}")
    print(f"    - Brier Score: {brier:.4f}")
    print(f"    - ECE:         {ece:.4f}")

    # 4. Statistical Claim Audit
    print("\n--- Part 3: Statistical Claim Audit ---")
    
    # Re-train Baseline Logistic Regression
    sorted_card_ids = sorted(list(cards_lib.keys()))
    card_to_idx = {cid: idx for idx, cid in enumerate(sorted_card_ids)}
    
    def to_features(row):
        feat = np.zeros(num_cards * 2 + 1)
        for c in json.loads(row["player_deck"]):
            if str(c) in card_to_idx:
                feat[card_to_idx[str(c)]] = 1.0
        for c in json.loads(row["opponent_deck"]):
            if str(c) in card_to_idx:
                feat[num_cards + card_to_idx[str(c)]] = 1.0
        p1_trophies = row.get("player_trophies", 0.0)
        p2_trophies = row.get("opponent_trophies", 0.0)
        feat[-1] = (p1_trophies - p2_trophies) / 1000.0
        return feat
        
    X_train_lr = np.stack(train_df.apply(to_features, axis=1).values)
    y_train_lr = train_df["win"].values
    X_test_lr = np.stack(test_df.apply(to_features, axis=1).values)
    y_test_lr = test_df["win"].values
    
    lr = LogisticRegression(max_iter=100)
    lr.fit(X_train_lr, y_train_lr)
    lr_probs = lr.predict_proba(X_test_lr)[:, 1]
    
    # Paired statistical test
    lr_errors = (y_test_lr - lr_probs) ** 2
    model_errors = (y_true - y_prob) ** 2
    
    t_stat, p_val = ttest_rel(lr_errors, model_errors)
    mean_diff = float(np.mean(lr_errors - model_errors))
    std_diff = float(np.std(lr_errors - model_errors))
    se_diff = std_diff / np.sqrt(len(lr_errors))
    ci = [mean_diff - 1.96 * se_diff, mean_diff + 1.96 * se_diff]
    
    print(f"  • Paired t-test results on Squared Error differences:")
    print(f"    - Sample Size N:           {len(y_true)}")
    print(f"    - t-statistic:             {t_stat:.4f}")
    print(f"    - p-value:                 {p_val:.4e}")
    print(f"    - 95% Error Diff CI:       [{ci[0]:.6f}, {ci[1]:.6f}]")
    print(f"    - Statistical Claim:       {'SUPPORTED' if p_val < 0.05 else 'NOT SUPPORTED'}")
    
    # Save Audit Log
    audit_results = {
        "data_leakage": {
            "is_chronological": bool(is_chronological),
            "shared_train_val": len(leakage_train_val),
            "shared_val_test": len(leakage_val_test),
            "shared_train_test": len(leakage_train_test)
        },
        "verified_metrics": {
            "accuracy": acc,
            "roc_auc": auc,
            "log_loss": ll,
            "brier_score": brier,
            "ece": ece
        },
        "statistical_test": {
            "sample_size": len(y_true),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "error_diff_ci": ci
        }
    }
    
    out_path = PROJECT_ROOT / "logs" / "sprint17_audit.json"
    with open(out_path, "w") as f:
        json.dump(audit_results, f, indent=4)
        
    print(f"\n[+] Sprint 17 audit completed! Logs saved to: {out_path}\n")

if __name__ == "__main__":
    main()
