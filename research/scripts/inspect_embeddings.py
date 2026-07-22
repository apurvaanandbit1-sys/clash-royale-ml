import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

# Resolve paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from models.deck_encoder import DeckEncoder
from training.train_encoder_test import SimpleClassifier

def analyze_embeddings():
    print("=" * 60)
    print("         CLASH ROYALE CARD EMBEDDINGS ANALYTICS")
    print("=" * 60)

    # 1. Paths and Configs
    checkpoint_path = PROJECT_ROOT / "models" / "checkpoints" / "sprint12_test_encoder.pt"
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at '{checkpoint_path}'! Run train_encoder_test.py first.")
        sys.exit(1)

    card_lib_path = PROJECT_ROOT / "features" / "card_library.json"
    with open(card_lib_path, "r") as f:
        card_lib = json.load(f)
    
    sorted_card_ids = sorted(list(card_lib.keys()))
    card_to_idx = {str(cid): idx for idx, cid in enumerate(sorted_card_ids)}
    idx_to_name = {idx: card_lib[str(cid)].get("name", str(cid)) for cid, idx in card_to_idx.items()}
    num_cards = len(card_lib)

    # 2. Load model state
    encoder = DeckEncoder(num_cards=num_cards, embedding_dim=16)
    model = SimpleClassifier(encoder, projection_dim=16)
    
    # Load state dict
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    
    # Extract weights
    weights = model.encoder.embeddings.weight.detach().numpy() # Shape [122, 16]
    
    # 3. Compute vector norms
    norms = np.linalg.norm(weights, axis=1)
    sorted_norm_indices = np.argsort(norms)
    
    print("\n[Analysis 1] Embedding Vector Norms:")
    print("  • Bottom 5 smallest norm cards:")
    for idx in sorted_norm_indices[:5]:
        print(f"    - {idx_to_name[idx]:20s}: Norm {norms[idx]:.4f}")
        
    print("  • Top 5 largest norm cards:")
    for idx in sorted_norm_indices[-5:]:
        print(f"    - {idx_to_name[idx]:20s}: Norm {norms[idx]:.4f}")

    # 4. Nearest Neighbors (Cosine Similarity)
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

    print("\n[Analysis 2] Cosine Similarity Nearest Neighbors (Target Cards):")
    # Let's inspect some landmark cards (e.g. Witch, Mega Knight, Giant) if they are in the library
    landmark_cards = ["Witch", "Mega Knight", "Giant", "The Log"]
    landmark_indices = []
    
    for name in landmark_cards:
        # Find index matching name
        found = False
        for idx, c_name in idx_to_name.items():
            if c_name.lower() == name.lower():
                landmark_indices.append(idx)
                found = True
                break
        if not found and len(idx_to_name) > 0:
            landmark_indices.append(0) # Fallback to index 0

    # Remove duplicates
    landmark_indices = list(set(landmark_indices))

    for idx in landmark_indices:
        target_vector = weights[idx]
        target_name = idx_to_name[idx]
        
        # Compute similarities with all other cards
        similarities = []
        for other_idx in range(num_cards):
            if other_idx == idx:
                continue
            sim = cosine_similarity(target_vector, weights[other_idx])
            similarities.append((other_idx, sim))
            
        # Sort and take top-3
        similarities.sort(key=lambda x: x[1], reverse=True)
        print(f"  • Nearest neighbors for '{target_name}':")
        for i in range(min(3, len(similarities))):
            other_name = idx_to_name[similarities[i][0]]
            sim_val = similarities[i][1]
            print(f"    - {other_name:20s}: Similarity = {sim_val:.4f}")

    # 5. Dimensionality Reduction (PCA)
    print("\n[Analysis 3] Principal Component Analysis (PCA) Coordinates:")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(weights)
    
    # Save PCA results table
    pca_df = pd.DataFrame({
        "card_name": [idx_to_name[i] for i in range(num_cards)],
        "PCA_1": coords[:, 0],
        "PCA_2": coords[:, 1]
    })
    
    # Save coordinates CSV
    coords_path = PROJECT_ROOT / "research" / "results" / "sprint12_embeddings_pca.csv"
    pca_df.to_csv(coords_path, index=False)
    print(f"  • PCA Coordinates successfully saved to: {coords_path}")
    print("  • Sample PCA coordinates:")
    for i in range(min(5, num_cards)):
        print(f"    - {pca_df.iloc[i]['card_name']:20s}: PCA1={pca_df.iloc[i]['PCA_1']:.4f}, PCA2={pca_df.iloc[i]['PCA_2']:.4f}")
        
    print("=" * 60)

if __name__ == "__main__":
    analyze_embeddings()
