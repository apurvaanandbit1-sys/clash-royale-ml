import os
import sys
import json
import yaml
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path

# Resolve paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from training.dataset import get_dataloaders
from models.deck_encoder import DeckEncoder
from models.interaction_head import BradleyTerryHead, SkillDifference
from models.predictor import MatchupModel

def run_matchup_training_verification():
    print("=" * 60)
    print("   RUNNING SPRINT 13 TRAINING VERIFICATION LOOP")
    print("=" * 60)

    # 1. Load configs
    with open(PROJECT_ROOT / "config" / "dataset_config.yaml", "r") as f:
        ds_config = yaml.safe_load(f)
    with open(PROJECT_ROOT / "config" / "deck_encoder_config.yaml", "r") as f:
        m_config = yaml.safe_load(f)

    # 2. Get DataLoaders
    print("Loading datasets...")
    df = pd.read_parquet(PROJECT_ROOT / ds_config["data"]["dataset_path"])
    
    with open(PROJECT_ROOT / ds_config["data"]["card_library_path"], "r") as f:
        cards_lib = json.load(f)
    num_cards = len(cards_lib)

    train_ldr, val_ldr, _ = get_dataloaders(
        df=df,
        card_library_path=PROJECT_ROOT / ds_config["data"]["card_library_path"],
        batch_size=ds_config["loader"]["batch_size"],
        seed=ds_config["loader"]["seed"],
        augment_prob=ds_config["augmentation"]["probability"]
    )

    # 3. Instantiate model components
    print("Instantiating Siamese MatchupModel components...")
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
    
    # 4. Train loop setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    print("\nStarting Training (2 epochs)...")
    for epoch in range(1, 3):
        model.train()
        epoch_loss = 0.0
        batches = 0
        
        for batch in train_ldr:
            optimizer.zero_grad()
            
            p1 = batch["p1_deck"]
            p2 = batch["p2_deck"]
            t_diff = batch["trophy_diff"]
            target = batch["target"]
            
            # Forward pass
            out = model(p1, p2, t_diff)
            loss = criterion(out, target)
            
            # Backward pass
            loss.backward()
            
            # Check gradient flow to embeddings on batch 1
            if epoch == 1 and batches == 0:
                emb_grad = model.encoder.embeddings.weight.grad
                head_grad = model.head.M.grad
                skill_grad = model.skill.mlp[0].weight.grad
                
                print(f"  [Gradient Audit] Card Embedding grad norm: {emb_grad.norm().item():.6f}")
                print(f"  [Gradient Audit] BT Head Matrix grad norm:   {head_grad.norm().item():.6f}")
                print(f"  [Gradient Audit] Skill MLP grad norm:       {skill_grad.norm().item():.6f}")
                
                if any(g is None or g.norm().item() == 0.0 for g in [emb_grad, head_grad, skill_grad]):
                    print("  CRITICAL ERROR: Zero gradients detected in backpropagation!")
                else:
                    print("  SUCCESS: Gradients are non-zero across all modules.")

            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        avg_loss = epoch_loss / batches
        print(f"  Epoch {epoch:02d} | Avg BCE Loss: {avg_loss:.4f}")

    # 5. Serialization Check
    os.makedirs(PROJECT_ROOT / "models" / "checkpoints", exist_ok=True)
    checkpoint_path = PROJECT_ROOT / "models" / "checkpoints" / "sprint13_matchup_model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n[+] Checkpoint successfully saved in: {checkpoint_path}\n")

if __name__ == "__main__":
    run_matchup_training_verification()
