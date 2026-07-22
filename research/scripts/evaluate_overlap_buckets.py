import json
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from training.dataset import get_dataloaders
from models.deck_encoder import DeckEncoder
from models.interaction_head import BradleyTerryHead, SkillDifference
from models.predictor import MatchupModel
from training.calibration import evaluate_calibration

def main():
    print("=== Phase 0 Task 3: Overlap-Bucket Analysis (0 to 8 Shared Cards) ===")
    
    with open(PROJECT_ROOT / "config" / "dataset_config.yaml", "r") as f:
        ds_config = yaml.safe_load(f)
        
    df = pd.read_parquet(PROJECT_ROOT / ds_config["data"]["dataset_path"])
    card_lib_path = PROJECT_ROOT / ds_config["data"]["card_library_path"]
    
    with open(card_lib_path, "r") as f:
        cards_lib = json.load(f)
    num_cards = len(cards_lib)
    
    train_ldr, val_ldr, test_ldr = get_dataloaders(
        df=df,
        card_library_path=card_lib_path,
        batch_size=128,
        seed=42,
        augment_prob=0.0
    )
    
    # Load MatchupModel
    best_ckpt_path = PROJECT_ROOT / "models" / "checkpoints" / "sprint13_matchup_model.pt"
    encoder = DeckEncoder(num_cards=num_cards, embedding_dim=16)
    head = BradleyTerryHead(projection_dim=16)
    skill = SkillDifference(hidden_dim=8)
    model = MatchupModel(encoder, head, skill)
    model.load_state_dict(torch.load(best_ckpt_path, map_location=torch.device("cpu")))
    model.eval()
    
    all_p1 = []
    all_p2 = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_ldr:
            p1 = batch["p1_deck"]
            p2 = batch["p2_deck"]
            td = batch["trophy_diff"]
            target = batch["target"]
            probs = model.predict_proba(p1, p2, td)
            
            all_p1.append(p1.numpy())
            all_p2.append(p2.numpy())
            all_targets.append(target.numpy())
            all_probs.append(probs.numpy())
            
    p1_arr = np.vstack(all_p1)
    p2_arr = np.vstack(all_p2)
    y_true = np.vstack(all_targets).squeeze()
    y_prob = np.vstack(all_probs).squeeze()
    
    # Calculate exact overlap for each battle
    overlaps = []
    for i in range(len(y_true)):
        set1 = set(p1_arr[i])
        set2 = set(p2_arr[i])
        overlaps.append(len(set1.intersection(set2)))
    overlaps = np.array(overlaps)
    
    print(f"\nTotal Test Matches: {len(overlaps)}")
    print("-" * 75)
    print(f"{'Overlap':<8}{'n':<8}{'Pct':<8}{'Accuracy':<12}{'ECE':<10}{'Mean Target':<14}{'Mean Prob':<12}")
    print("-" * 75)
    
    bucket_data = {}
    
    for k in range(9):
        mask = (overlaps == k)
        n_k = int(np.sum(mask))
        if n_k > 0:
            y_k = y_true[mask]
            p_k = y_prob[mask]
            acc_k = float(accuracy_score(y_k, (p_k >= 0.5).astype(float)))
            ece_k = float(evaluate_calibration(y_k, p_k)["ece"])
            mean_y_k = float(np.mean(y_k))
            mean_p_k = float(np.mean(p_k))
            pct_k = n_k / len(overlaps) * 100
        else:
            acc_k = ece_k = mean_y_k = mean_p_k = pct_k = 0.0
            
        bucket_data[k] = {
            "n": n_k,
            "pct": float(pct_k),
            "accuracy": acc_k,
            "ece": ece_k,
            "mean_target": mean_y_k,
            "mean_prob": mean_p_k
        }
        print(f"{k:<8}{n_k:<8}{pct_k:<8.2f}%{acc_k:<12.4f}{ece_k:<10.4f}{mean_y_k:<14.4f}{mean_p_k:<12.4f}")
        
    # Grouped 6+ bucket investigation
    mask_6plus = (overlaps >= 6)
    n_6plus = int(np.sum(mask_6plus))
    if n_6plus > 0:
        y_6p = y_true[mask_6plus]
        p_6p = y_prob[mask_6plus]
        acc_6p = float(accuracy_score(y_6p, (p_6p >= 0.5).astype(float)))
        ece_6p = float(evaluate_calibration(y_6p, p_6p)["ece"])
        mean_y_6p = float(np.mean(y_6p))
        mean_p_6p = float(np.mean(p_6p))
    else:
        acc_6p = ece_6p = mean_y_6p = mean_p_6p = 0.0
        
    print("-" * 75)
    print(f"\nInvestigation of 6+ Card Overlap Bucket:")
    print(f"  • Sample size n (6+ overlap): {n_6plus} ({n_6plus/len(overlaps)*100:.2f}% of test set)")
    print(f"  • Accuracy (6+ overlap):      {acc_6p:.4f}")
    print(f"  • Mean Actual Target y:       {mean_y_6p:.4f}")
    print(f"  • Mean Predicted Prob p:      {mean_p_6p:.4f}")
    print(f"  • ECE (6+ overlap):           {ece_6p:.4f}")
    
    # Save log
    out_payload = {
        "buckets": bucket_data,
        "bucket_6plus": {
            "n": n_6plus,
            "accuracy": acc_6p,
            "mean_target": mean_y_6p,
            "mean_prob": mean_p_6p,
            "ece": ece_6p
        }
    }
    
    out_path = PROJECT_ROOT / "logs" / "overlap_bucket_results.json"
    with open(out_path, "w") as f:
        json.dump(out_payload, f, indent=4)
        
    print(f"\nSaved overlap bucket investigation to {out_path}")

if __name__ == "__main__":
    main()
