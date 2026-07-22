import json
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from training.dataset import get_dataloaders
from models.deck_encoder import DeckEncoder
from models.interaction_head import BradleyTerryHead, SkillDifference
from models.predictor import MatchupModel
from training.trainer import Trainer, EarlyStopping, CheckpointManager, MetricsLogger

def train_canonical_model(seed, df, card_lib_path, num_cards, epochs=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_ldr, val_ldr, test_ldr = get_dataloaders(
        df=df,
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
    
    ckpt_dir = PROJECT_ROOT / "models" / "checkpoints" / f"temp_sync_s{seed}"
    log_file = PROJECT_ROOT / "logs" / f"temp_sync_s{seed}.csv"
    json_file = PROJECT_ROOT / "logs" / f"temp_sync_s{seed}.json"
    
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
        config={"model": f"canonical_s{seed}"},
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
    print("=== Testing Canonical Model Training Across Seeds ===")
    with open(PROJECT_ROOT / "config" / "dataset_config.yaml", "r") as f:
        ds_config = yaml.safe_load(f)
        
    df = pd.read_parquet(PROJECT_ROOT / ds_config["data"]["dataset_path"])
    card_lib_path = PROJECT_ROOT / ds_config["data"]["card_library_path"]
    with open(card_lib_path, "r") as f:
        cards_lib = json.load(f)
    num_cards = len(cards_lib)
    
    for s in [42, 43, 44]:
        acc, auc = train_canonical_model(s, df, card_lib_path, num_cards, epochs=10)
        print(f"Seed {s} -> Acc: {acc*100:.4f}% ({acc:.6f}), AUC: {auc:.6f}")

if __name__ == "__main__":
    main()
