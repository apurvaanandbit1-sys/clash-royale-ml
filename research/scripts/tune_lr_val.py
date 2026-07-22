import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def main():
    print("=== D2 Verification: Hyperparameter Search for Logistic Regression on Validation Set ===")
    
    with open(PROJECT_ROOT / "config" / "dataset_config.yaml", "r") as f:
        ds_config = yaml.safe_load(f)
        
    df = pd.read_parquet(PROJECT_ROOT / ds_config["data"]["dataset_path"])
    card_lib_path = PROJECT_ROOT / ds_config["data"]["card_library_path"]
    
    with open(card_lib_path, "r") as f:
        cards_lib = json.load(f)
    num_cards = len(cards_lib)
    
    n_train = int(len(df) * 0.7)
    n_val = int(len(df) * 0.15)
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    
    sorted_card_ids = sorted(list(cards_lib.keys()))
    card_to_idx = {cid: idx for idx, cid in enumerate(sorted_card_ids)}
    
    def to_features(row):
        feat = np.zeros(num_cards * 2 + 1, dtype=np.float32)
        for c in json.loads(row["player_deck"]):
            if str(c) in card_to_idx:
                feat[card_to_idx[str(c)]] = 1.0
        for c in json.loads(row["opponent_deck"]):
            if str(c) in card_to_idx:
                feat[num_cards + card_to_idx[str(c)]] = 1.0
        p1_t = row.get("player_trophies", 0.0)
        p2_t = row.get("opponent_trophies", 0.0)
        feat[-1] = float(p1_t - p2_t)
        return feat

    # Apply 50% symmetry augmentation on training set
    X_train_orig = np.stack(train_df.apply(to_features, axis=1).values)
    y_train_orig = train_df["win"].values
    
    # Augmented copy (swapped decks, negated trophy diff, inverted target)
    X_train_swap = X_train_orig.copy()
    X_train_swap[:, :num_cards] = X_train_orig[:, num_cards:2*num_cards]
    X_train_swap[:, num_cards:2*num_cards] = X_train_orig[:, :num_cards]
    X_train_swap[:, -1] = -X_train_orig[:, -1]
    y_train_swap = 1.0 - y_train_orig
    
    X_train = np.vstack([X_train_orig, X_train_swap])
    y_train = np.concatenate([y_train_orig, y_train_swap])
    
    X_val = np.stack(val_df.apply(to_features, axis=1).values)
    y_val = val_df["win"].values
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    
    X_train_scaled[:, -1] = scaler.fit_transform(X_train[:, -1].reshape(-1, 1)).squeeze()
    X_val_scaled[:, -1] = scaler.transform(X_val[:, -1].reshape(-1, 1)).squeeze()
    
    c_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0]
    print(f"\nTuning C parameter on Validation Set (N_val = {len(y_val):,}):")
    print(f"{'C Value':<10} | {'Val Accuracy':<15} | {'Val ROC-AUC':<15}")
    print("-" * 45)
    
    best_c = None
    best_val_auc = 0.0
    
    for c in c_values:
        lr = LogisticRegression(C=c, max_iter=1000, random_state=42)
        lr.fit(X_train_scaled, y_train)
        probs_val = lr.predict_proba(X_val_scaled)[:, 1]
        preds_val = (probs_val >= 0.5).astype(float)
        
        val_acc = accuracy_score(y_val, preds_val)
        val_auc = roc_auc_score(y_val, probs_val)
        print(f"{c:<10} | {val_acc*100:.2f}% ({val_acc:.6f}) | {val_auc:.6f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_c = c
            
    print("-" * 45)
    print(f"Optimal C parameter on Validation Set: C = {best_c} (Val AUC = {best_val_auc:.6f})")

if __name__ == "__main__":
    main()
