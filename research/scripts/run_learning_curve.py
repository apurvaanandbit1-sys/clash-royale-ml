import json
import sys
import yaml
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from training.dataset import get_dataloaders
from models.deck_encoder import DeckEncoder
from models.interaction_head import BradleyTerryHead, SkillDifference
from models.predictor import MatchupModel
from training.trainer import Trainer, EarlyStopping, CheckpointManager, MetricsLogger
from training.calibration import evaluate_calibration

def train_and_eval_subsample(sz, seed, full_train_df, val_df, test_df, card_lib_path, num_cards, epochs=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if sz <= len(full_train_df):
        sub_train_df = full_train_df.iloc[:sz].reset_index(drop=True)
    else:
        sub_train_df = full_train_df.reset_index(drop=True)
        
    # Build DataLoaders using canonical get_dataloaders function with seeded PyTorch generator
    train_ldr, val_ldr, test_ldr = get_dataloaders(
        df=pd.concat([sub_train_df, val_df, test_df], ignore_index=True),
        card_library_path=card_lib_path,
        batch_size=128,
        seed=seed,
        augment_prob=0.5
    )
    
    encoder = DeckEncoder(num_cards=num_cards, embedding_dim=16)
    head = BradleyTerryHead(projection_dim=16)
    skill = SkillDifference(hidden_dim=8)
    model = MatchupModel(encoder, head, skill)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    
    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints" / f"temp_lc_sz{sz}_s{seed}"
    log_file = PROJECT_ROOT / "logs" / f"temp_lc_sz{sz}_s{seed}.csv"
    json_file = PROJECT_ROOT / "logs" / f"temp_lc_sz{sz}_s{seed}.json"
    
    ckpt_mgr = CheckpointManager(ckpt_dir)
    logger = MetricsLogger(log_file, json_file)
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    
    trainer = Trainer(
        model=model,
        train_loader=train_ldr,
        val_loader=val_ldr,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_manager=ckpt_mgr,
        logger=logger,
        early_stopping=early_stopping,
        config={"model": f"learning_curve_sz{sz}_s{seed}"},
        seed=seed
    )
    
    trainer.fit(num_epochs=epochs)
    
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
    probs = np.vstack(all_probs).squeeze()
    
    acc = float(accuracy_score(y_true, (probs >= 0.5).astype(float)))
    auc = float(roc_auc_score(y_true, probs))
    return acc, auc

def main():
    print("=== D4 & E1: Synchronized Multi-Seed Learning Curve Study (Seeds 42, 43, 44) ===")
    
    with open(PROJECT_ROOT / "config" / "dataset_config.yaml", "r") as f:
        ds_config = yaml.safe_load(f)
        
    df = pd.read_parquet(PROJECT_ROOT / ds_config["data"]["dataset_path"])
    card_lib_path = PROJECT_ROOT / ds_config["data"]["card_library_path"]
    with open(card_lib_path, "r") as f:
        cards_lib = json.load(f)
    num_cards = len(cards_lib)
    
    n_train = int(len(df) * 0.7)
    n_val = int(len(df) * 0.15)
    
    full_train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val:].reset_index(drop=True)
    
    sizes = [10000, 25000, 50000, 70379]
    seeds = [42, 43, 44]
    
    table_rows = []
    
    for sz in sizes:
        print(f"\n--- Evaluating Subsample Size: {sz:,} Battles across Seeds 42, 43, 44 ---")
        accs_sz = []
        aucs_sz = []
        seed_details = {}
        
        for s in seeds:
            acc_s, auc_s = train_and_eval_subsample(sz, s, full_train_df, val_df, test_df, card_lib_path, num_cards, epochs=10)
            accs_sz.append(acc_s)
            aucs_sz.append(auc_s)
            seed_details[f"seed_{s}_acc"] = float(acc_s)
            seed_details[f"seed_{s}_auc"] = float(auc_s)
            print(f"  • Seed {s} -> Acc: {acc_s*100:.2f}%, AUC: {auc_s:.4f}")
            
        mean_acc = np.mean(accs_sz)
        std_acc = np.std(accs_sz)
        mean_auc = np.mean(aucs_sz)
        std_auc = np.std(aucs_sz)
        
        print(f"  --> Size {sz:,} Summary: Acc = {mean_acc*100:.2f}% ± {std_acc*100:.2f}% | AUC = {mean_auc:.4f} ± {std_auc:.4f}")
        
        row = {
            "size": sz,
            "seed_42_acc": seed_details["seed_42_acc"],
            "seed_43_acc": seed_details["seed_43_acc"],
            "seed_44_acc": seed_details["seed_44_acc"],
            "mean_acc": float(mean_acc),
            "std_acc": float(std_acc),
            "seed_42_auc": seed_details["seed_42_auc"],
            "seed_43_auc": seed_details["seed_43_auc"],
            "seed_44_auc": seed_details["seed_44_auc"],
            "mean_auc": float(mean_auc),
            "std_auc": float(std_auc)
        }
        table_rows.append(row)
        
    # Save CSV
    df_res = pd.DataFrame(table_rows)
    csv_path = PROJECT_ROOT / "research" / "results" / "learning_curve.csv"
    df_res.to_csv(csv_path, index=False)
    print(f"\n[+] Saved synchronized multi-seed learning curve metrics to {csv_path}")
    
    # Plot Multi-Seed Learning Curve with Error Bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    xs = [r["size"] for r in table_rows]
    mean_accs_pct = [r["mean_acc"] * 100 for r in table_rows]
    std_accs_pct = [r["std_acc"] * 100 for r in table_rows]
    
    mean_aucs = [r["mean_auc"] for r in table_rows]
    std_aucs = [r["std_auc"] for r in table_rows]
    
    ax1.errorbar(xs, mean_accs_pct, yerr=std_accs_pct, fmt='o-', color='#1f77b4', ecolor='#aec7e8', elinewidth=2, capsize=5, linewidth=2, markersize=8, label='3-Seed Mean ± Std')
    ax1.set_title("Test Accuracy vs Training Size (Within Existing Corpus)")
    ax1.set_xlabel("Training Set Subsample Size (Battles)")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    
    ax2.errorbar(xs, mean_aucs, yerr=std_aucs, fmt='s-', color='#2ca02c', ecolor='#a1d99b', elinewidth=2, capsize=5, linewidth=2, markersize=8, label='3-Seed Mean ± Std')
    ax2.set_title("Test ROC-AUC vs Training Size (Within Existing Corpus)")
    ax2.set_xlabel("Training Set Subsample Size (Battles)")
    ax2.set_ylabel("Test ROC-AUC")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    img_path = PROJECT_ROOT / "research" / "results" / "learning_curve.png"
    plt.savefig(img_path, dpi=300)
    plt.close()
    print(f"[+] Saved synchronized learning curve chart to {img_path}")

if __name__ == "__main__":
    main()
