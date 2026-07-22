import os
import sys
import json
import time
import argparse
import yaml
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess

# Resolve paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

from training.dataset import get_dataloaders
from models.deck_encoder import DeckEncoder
from models.interaction_head import BradleyTerryHead, SkillDifference
from models.predictor import MatchupModel
from training.trainer import Trainer, CheckpointManager, EarlyStopping, MetricsLogger

def get_git_commit() -> str:
    """Retrieves the current git commit hash."""
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        return git_hash
    except Exception:
        return "N/A"

def main():
    parser = argparse.ArgumentParser(description="Clash Royale Matchup Model Production Training Entry Point")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs count")
    parser.add_argument("--config", type=str, default="config/training_config.yaml", help="Path to training config")
    args = parser.parse_args()

    print("=" * 60)
    print("      CLASH ROYALE ML PRODUCTION TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load Configurations
    config_path = Path(args.config)
    with open(config_path, "r") as f:
        t_config = yaml.safe_load(f)
    
    with open(PROJECT_ROOT / "config" / "dataset_config.yaml", "r") as f:
        ds_config = yaml.safe_load(f)
        
    with open(PROJECT_ROOT / "config" / "deck_encoder_config.yaml", "r") as f:
        m_config = yaml.safe_load(f)

    # 2. Set Seed for Reproducibility
    seed = t_config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 3. Load Data & Dataloaders
    print("Loading parquet dataset...")
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

    # 4. Construct MatchupModel
    print("Constructing MatchupModel...")
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
    
    head = BradleyTerryHead(
        projection_dim=m_config["model"]["projection_dim"],
        init_scale=m_config["comparison"]["init_scale"]
    )
    
    skill = SkillDifference(
        hidden_dim=m_config["skill"]["hidden_dim"],
        scaling_factor=m_config["skill"]["scaling_factor"]
    )
    
    model = MatchupModel(encoder, head, skill)

    # 5. Initialize Optimizer
    opt_type = t_config["optimizer"]["type"]
    if opt_type == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=t_config["optimizer"]["lr"],
            weight_decay=t_config["optimizer"]["weight_decay"],
            betas=tuple(t_config["optimizer"]["betas"])
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_type}")

    # 6. Initialize LR Scheduler
    sched_type = t_config["scheduler"]["type"]
    if sched_type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=t_config["scheduler"]["patience"],
            factor=t_config["scheduler"]["factor"],
            min_lr=t_config["scheduler"]["min_lr"]
        )
    elif sched_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_config["scheduler"]["T_max"]
        )
    else:
        scheduler = None

    # 7. Initialize Trainer helpers
    c_manager = CheckpointManager(PROJECT_ROOT / t_config["checkpoint"]["dir"])
    logger = MetricsLogger(
        PROJECT_ROOT / t_config["logging"]["metrics_csv"],
        PROJECT_ROOT / t_config["logging"]["metrics_json"]
    )
    early_stop = EarlyStopping(
        patience=t_config["early_stopping"]["patience"],
        min_delta=t_config["early_stopping"]["min_delta"]
    )

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
        config=t_config,
        seed=seed
    )

    # 8. Resume Checkpoint (if provided)
    start_epoch = 1
    total_epochs = args.epochs if args.epochs is not None else t_config["training"]["epochs"]

    if args.resume:
        print(f"Resuming training from checkpoint: {args.resume} ...")
        res_info = c_manager.load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = res_info["epoch"] + 1
        print(f"Successfully loaded checkpoint state! Resuming from epoch {start_epoch}.")

    # 9. Run Fit
    t_start = time.time()
    best_epoch_metrics = trainer.fit(num_epochs=total_epochs, start_epoch=start_epoch)
    duration = time.time() - t_start

    # 10. Load best validation model for final test evaluation
    best_ckpt_path = PROJECT_ROOT / t_config["checkpoint"]["dir"] / "best_validation.pt"
    if best_ckpt_path.exists():
        print("\nLoading best model checkpoint for final test evaluation...")
        c_manager.load_checkpoint(best_ckpt_path, model)
    
    # Run test evaluation
    test_metrics = trainer.val_epoch()
    print("=" * 60)
    print("                FINAL TEST EVALUATION")
    print("=" * 60)
    print(f"  • Test Loss:                 {test_metrics['loss']:.4f}")
    print(f"  • Test Accuracy:             {test_metrics['accuracy']:.4f}")
    print(f"  • Test ROC-AUC:              {test_metrics['roc_auc']:.4f}")
    print(f"  • Test Brier Score:          {test_metrics['brier_score']:.4f}")
    print("=" * 60)

    # 11. Write Experiment Manifest
    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": get_git_commit(),
        "dataset_version": ds_config["data"]["dataset_path"],
        "config_path": args.config,
        "random_seed": seed,
        "checkpoint_dir": t_config["checkpoint"]["dir"],
        "training_duration_seconds": duration,
        "best_epoch": best_epoch_metrics.get("best_epoch", "N/A"),
        "best_val_loss": best_epoch_metrics.get("val_loss", "N/A"),
        "best_val_accuracy": best_epoch_metrics.get("val_accuracy", "N/A"),
        "best_val_roc_auc": best_epoch_metrics.get("val_roc_auc", "N/A"),
        "final_test_metrics": test_metrics
    }
    
    manifest_path = PROJECT_ROOT / t_config["logging"]["manifest_json"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"\n[+] Training run completed! Experiment manifest locked in: {manifest_path}\n")

if __name__ == "__main__":
    main()
