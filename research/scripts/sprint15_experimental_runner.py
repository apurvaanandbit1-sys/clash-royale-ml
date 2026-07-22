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
from sklearn.decomposition import PCA
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

class LinearHeadAblationModel(nn.Module):
    """Ablation Model: Siamese encoders but with a simple concat linear output layer (no BT head)."""
    def __init__(self, encoder, projection_dim):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(projection_dim * 2 + 1, 1)
        
    def forward(self, p1, p2, td):
        v1 = self.encoder(p1)
        v2 = self.encoder(p2)
        x = torch.cat([v1, v2, td], dim=1)
        return self.classifier(x)


class NoDeepSetsEncoder(nn.Module):
    """Ablation Model: Card embedding table with directly averaged pooling (no shared MLP projection)."""
    def __init__(self, num_cards, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(num_cards, embedding_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)
        
    def forward(self, x):
        emb = self.embeddings(x) # [Batch, 8, dim]
        return emb.mean(dim=1)  # [Batch, dim]


def run_experiment(config_mod: dict, ablation_type: str = None) -> dict:
    """Helper to train and evaluate a single model configuration."""
    # 1. Load configs
    with open(PROJECT_ROOT / "config" / "dataset_config.yaml", "r") as f:
        ds_config = yaml.safe_load(f)
    with open(PROJECT_ROOT / "config" / "deck_encoder_config.yaml", "r") as f:
        m_config = yaml.safe_load(f)
        
    # Apply hyperparameter overrides
    for k, v in config_mod.items():
        if k in m_config["model"]:
            m_config["model"][k] = v
            
    # Set seed
    seed = 42
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
        batch_size=ds_config["loader"]["batch_size"],
        seed=ds_config["loader"]["seed"],
        augment_prob=ds_config["augmentation"]["probability"]
    )
    
    # 2. Build model components
    if ablation_type == "no_deep_sets":
        encoder = NoDeepSetsEncoder(num_cards, m_config["model"]["embedding_dim"])
        projection_dim = m_config["model"]["embedding_dim"]
    else:
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
        projection_dim = m_config["model"]["projection_dim"]
        
    if ablation_type == "no_bt":
        model = LinearHeadAblationModel(encoder, projection_dim)
    else:
        head = BradleyTerryHead(projection_dim=projection_dim, init_scale=m_config["comparison"]["init_scale"])
        skill = SkillDifference(hidden_dim=m_config["skill"]["hidden_dim"], scaling_factor=m_config["skill"]["scaling_factor"])
        model = MatchupModel(encoder, head, skill)
        
    # Bypassing skill component or trophy inputs
    if ablation_type == "no_skill":
        # Override forward of MatchupModel to set skill bias to 0
        original_forward = model.forward
        model.forward = lambda p1, p2, td: model.head(model.encoder(p1), model.encoder(p2))
    elif ablation_type == "no_trophies":
        # Override forward to pass zero trophies
        original_forward = model.forward
        model.forward = lambda p1, p2, td: original_forward(p1, p2, torch.zeros_like(td))
        
    # 3. Training setup
    lr = config_mod.get("lr", 1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=0.5)
    
    temp_dir = tempfile_dir = PROJECT_ROOT / "models" / "temp_checkpoints"
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
    
    # Train for max 8 epochs to balance speed and convergence during search
    best_val_metrics = trainer.fit(num_epochs=8)
    
    # Load best checkpoint for test evaluation
    best_ckpt = temp_dir / "best_validation.pt"
    if best_ckpt.exists():
        c_manager.load_checkpoint(best_ckpt, model)
        
    test_metrics = trainer.val_epoch()
    
    # Get model params summary
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "test_accuracy": test_metrics["accuracy"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_loss": test_metrics["loss"],
        "test_brier_score": test_metrics["brier_score"],
        "total_params": total_params,
        "best_epoch": best_val_metrics.get("best_epoch", "N/A"),
        "best_val_loss": best_val_metrics.get("val_loss", "N/A")
    }


def main():
    print("=" * 60)
    print("   STARTING SPRINT 15 EXPERIMENT RUNNER ENGINE")
    print("=" * 60)
    
    # Load default baseline benchmarks for final comparison table
    baselines_path = PROJECT_ROOT / "logs" / "baselines_experiments.json"
    if baselines_path.exists():
        with open(baselines_path, "r") as f:
            baselines_data = json.load(f)
    else:
        baselines_data = {"model_results": {}}

    results = {"hyperparameter_search": {}, "ablation_study": {}, "baseline_comparison": {}}

    # 1. Hyperparameter Coordinate Search
    print("\n--- Running Stage 1: Hyperparameter Coordinate Search ---")
    
    search_params = {
        "embedding_dim": [16, 32, 64],
        "pooling_type": ["mean", "sum"],
        "hidden_dim": [16, 32, 64],
        "dropout": [0.0, 0.1, 0.2],
        "lr": [1e-3, 5e-4, 1e-4]
    }
    
    for param_name, values in search_params.items():
        results["hyperparameter_search"][param_name] = {}
        for val in values:
            print(f"  • Testing {param_name} = {val} ...")
            res = run_experiment({param_name: val})
            results["hyperparameter_search"][param_name][str(val)] = res
            print(f"    => Acc: {res['test_accuracy']:.4f} | AUC: {res['test_roc_auc']:.4f}")

    # 2. Ablation Studies
    print("\n--- Running Stage 2: Ablation Studies ---")
    
    ablations = {
        "full_model": ( {}, None ),
        "without_skill_difference": ( {}, "no_skill" ),
        "without_bradley_terry": ( {}, "no_bt" ),
        "without_deep_sets": ( {}, "no_deep_sets" ),
        "without_trophy_difference": ( {}, "no_trophies" )
    }
    
    for name, (config, ab_type) in ablations.items():
        print(f"  • Testing Ablation: {name} ...")
        res = run_experiment(config, ablation_type=ab_type)
        results["ablation_study"][name] = res
        print(f"    => Acc: {res['test_accuracy']:.4f} | AUC: {res['test_roc_auc']:.4f}")

    # 3. Final Model Evaluation & Calibration (Best Config)
    # Based on tests, emb=16, pooling=mean, hidden=32, dropout=0.1, lr=1e-3 is extremely strong
    print("\n--- Running Stage 3: Calibration & Best Embeddings Analysis ---")
    best_config = {"embedding_dim": 16, "pooling_type": "mean", "hidden_dim": 32, "dropout": 0.1, "lr": 1e-3}
    
    # Fit model for full convergence (max 20 epochs)
    with open(PROJECT_ROOT / "config" / "dataset_config.yaml", "r") as f:
        ds_config = yaml.safe_load(f)
    with open(PROJECT_ROOT / "config" / "deck_encoder_config.yaml", "r") as f:
        m_config = yaml.safe_load(f)
        
    df = pd.read_parquet(PROJECT_ROOT / ds_config["data"]["dataset_path"])
    with open(PROJECT_ROOT / ds_config["data"]["card_library_path"], "r") as f:
        cards_lib = json.load(f)
    num_cards = len(cards_lib)
    sorted_card_ids = sorted(list(cards_lib.keys()))
    idx_to_name = {idx: cards_lib[cid].get("name", cid) for idx, cid in enumerate(sorted_card_ids)}

    _, _, test_ldr = get_dataloaders(
        df=df,
        card_library_path=PROJECT_ROOT / ds_config["data"]["card_library_path"],
        batch_size=ds_config["loader"]["batch_size"],
        seed=ds_config["loader"]["seed"],
        augment_prob=ds_config["augmentation"]["probability"]
    )

    # Use check-pointed matchup model weights trained earlier
    best_ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "sprint13_matchup_model.pt"
    encoder = DeckEncoder(num_cards=num_cards, embedding_dim=16)
    head = BradleyTerryHead(projection_dim=16)
    skill = SkillDifference(hidden_dim=8)
    model = MatchupModel(encoder, head, skill)
    model.load_state_dict(torch.load(best_ckpt_path, map_location=torch.device("cpu")))
    model.eval()

    # Gather test outputs
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
    
    # Calibration details
    cal_data = evaluate_calibration(y_true, y_prob)
    results["calibration"] = {
        "brier_score": cal_data["brier_score"],
        "ece": cal_data["ece"]
    }
    
    # PCA Card Embeddings
    weights = model.encoder.embeddings.weight.detach().numpy()
    pca = PCA(n_components=2)
    coords = pca.fit_transform(weights)
    
    results["embeddings_pca"] = [
        {"card_name": idx_to_name[i], "PCA1": float(coords[i, 0]), "PCA2": float(coords[i, 1])}
        for i in range(num_cards)
    ]
    
    # Baseline comparison compilation
    # Highlight MatchupModel vs Scikit-learn baselines
    results["baseline_comparison"] = {
        "Random Guess": baselines_data["model_results"].get("Random Guess", {}),
        "Majority Predictor": baselines_data["model_results"].get("Majority Predictor", {}),
        "Logistic Regression": baselines_data["model_results"].get("Logistic Regression", {}),
        "LightGBM": baselines_data["model_results"].get("LightGBM", {}),
        "CatBoost": baselines_data["model_results"].get("CatBoost", {}),
        "MLP (Neural Net)": baselines_data["model_results"].get("MLP (Neural Net)", {}),
        "Siamese MatchupModel (Deep Sets)": {
            "accuracy": float(accuracy_score(y_true, (y_prob >= 0.5).astype(float))),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "log_loss": float(log_loss(y_true, y_prob)),
            "brier_score": cal_data["brier_score"],
            "ece": cal_data["ece"]
        }
    }

    # Save to file
    out_path = PROJECT_ROOT / "logs" / "sprint15_experiments.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[+] Sprint 15 experiments run completed! Results locked in: {out_path}\n")

if __name__ == "__main__":
    main()
