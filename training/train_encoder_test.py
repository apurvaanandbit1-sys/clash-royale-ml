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

class SimpleClassifier(nn.Module):
    """
    Temporary wrapper to verify training of the DeckEncoder backbone.
    Concatenates player 1 deck vector, player 2 deck vector,
    and relative trophy difference.
    """
    def __init__(self, encoder: DeckEncoder, projection_dim: int):
        super().__init__()
        self.encoder = encoder
        # Input features: p1_deck (projection_dim) + p2_deck (projection_dim) + trophy_diff (1)
        self.classifier = nn.Linear(projection_dim * 2 + 1, 1)
        
    def forward(self, p1: torch.Tensor, p2: torch.Tensor, trophy_diff: torch.Tensor) -> torch.Tensor:
        v1 = self.encoder(p1)
        v2 = self.encoder(p2)
        x = torch.cat([v1, v2, trophy_diff], dim=1)
        return self.classifier(x)

def run_training_verification():
    print("=" * 60)
    print("   RUNNING SPRINT 12 TRAINING VERIFICATION LOOP")
    print("=" * 60)

    # 1. Load configs
    with open(PROJECT_ROOT / "config" / "dataset_config.yaml", "r") as f:
        ds_config = yaml.safe_load(f)
    with open(PROJECT_ROOT / "config" / "deck_encoder_config.yaml", "r") as f:
        enc_config = yaml.safe_load(f)

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

    # 3. Instantiate model
    print("Instantiating DeckEncoder & Simple Classifier...")
    encoder = DeckEncoder(
        num_cards=num_cards,
        embedding_dim=enc_config["model"]["embedding_dim"],
        pooling_type=enc_config["model"]["pooling_type"],
        hidden_dim=enc_config["model"]["hidden_dim"],
        projection_dim=enc_config["model"]["projection_dim"],
        dropout=enc_config["model"]["dropout"],
        use_layernorm=enc_config["model"]["use_layernorm"],
        init_type=enc_config["model"]["init_type"]
    )
    
    model = SimpleClassifier(encoder, enc_config["model"]["projection_dim"])
    
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
                grad_norm = model.encoder.embeddings.weight.grad.norm().item()
                print(f"  [Gradient Audit] Card Embedding grad norm: {grad_norm:.6f}")
                if grad_norm == 0.0:
                    print("  CRITICAL ERROR: Gradients are zero. Backpropagation is broken!")
                else:
                    print("  SUCCESS: Gradients are non-zero. Backpropagation verified.")

            optimizer.step()
            
            epoch_loss += loss.item()
            batches += 1
            
        avg_loss = epoch_loss / batches
        print(f"  Epoch {epoch:02d} | Avg BCE Loss: {avg_loss:.4f}")

    # 5. Serialization Check
    os.makedirs(PROJECT_ROOT / "models" / "checkpoints", exist_ok=True)
    checkpoint_path = PROJECT_ROOT / "models" / "checkpoints" / "sprint12_test_encoder.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n[+] Checkpoint successfully saved in: {checkpoint_path}\n")

if __name__ == "__main__":
    run_training_verification()
