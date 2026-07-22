import os
import sys
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from training.calibration import evaluate_calibration
from training.splits import split_dataset

# Resolve paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

def load_card_ids() -> list:
    """Loads all unique card IDs from card_library.json as strings."""
    lib_path = PROJECT_ROOT / "features" / "card_library.json"
    if not lib_path.exists():
        raise FileNotFoundError(f"Card library not found at: {lib_path}")
    with open(lib_path, "r") as f:
        cards_data = json.load(f)
    return sorted(list(cards_data.keys()))

def build_one_hot_features(df: pd.DataFrame, card_ids: list) -> pd.DataFrame:
    """
    Constructs binary card presence matrices (one-hot representation)
    for both player 1 (team) and player 2 (opponent).
    """
    N = len(card_ids)
    card_to_idx = {str(cid): idx for idx, cid in enumerate(card_ids)}
    
    one_hot_p1 = np.zeros((len(df), N), dtype=np.float32)
    one_hot_p2 = np.zeros((len(df), N), dtype=np.float32)
    
    for i, row in enumerate(df.itertuples()):
        p1_deck = json.loads(row.player_deck)
        p2_deck = json.loads(row.opponent_deck)
        
        for cid in p1_deck:
            cid_str = str(cid)
            if cid_str in card_to_idx:
                one_hot_p1[i, card_to_idx[cid_str]] = 1.0
                
        for cid in p2_deck:
            cid_str = str(cid)
            if cid_str in card_to_idx:
                one_hot_p2[i, card_to_idx[cid_str]] = 1.0
                
    p1_cols = [f"p1_has_{cid}" for cid in card_ids]
    p2_cols = [f"p2_has_{cid}" for cid in card_ids]
    
    one_hot_df = pd.DataFrame(
        np.hstack([one_hot_p1, one_hot_p2]), 
        columns=p1_cols + p2_cols
    )
    return one_hot_df

def run_representation_benchmark():
    print("=" * 60)
    print("   RUNNING REPRESENTATION BENCHMARK COMPILATION (SPRINT 6)")
    print("=" * 60)

    # 1. Load dataset & card library
    parquet_path = PROJECT_ROOT / "matches_with_features.parquet"
    if not parquet_path.exists():
        print(f"Error: Parquet file not found. Run preprocessing first.")
        sys.exit(1)
        
    df = pd.read_parquet(parquet_path)
    card_ids = load_card_ids()
    print(f"Loaded {len(df):,} match rows.")
    print(f"Loaded {len(card_ids)} unique cards from library.")

    # 2. Build representations
    print("\n[Step 1] Constructing Feature Matrices...")
    # Representation A: Aggregates (24 features)
    non_feature_cols = ["win", "player_deck", "opponent_deck", "battle_time"]
    p1_cols = [c for c in df.columns if c.startswith("p1_") and c not in non_feature_cols]
    p2_cols = [c for c in df.columns if c.startswith("p2_") and c not in non_feature_cols]
    rep_a = df[p1_cols + p2_cols].copy()

    # Representation B: One-hot Card Presence (~244 features)
    rep_b = build_one_hot_features(df, card_ids)

    # Representation C: Combined features
    rep_c = pd.concat([rep_a.reset_index(drop=True), rep_b.reset_index(drop=True)], axis=1)

    print(f"  • Representation A (Aggregates): {rep_a.shape}")
    print(f"  • Representation B (One-Hot):   {rep_b.shape}")
    print(f"  • Representation C (Combined):  {rep_c.shape}")

    # Build targets
    y = df["win"].copy()
    
    # Pack representations
    representations = {
        "Rep A (Aggregates)": rep_a,
        "Rep B (One-Hot)": rep_b,
        "Rep C (Combined)": rep_c
    }

    results = {}
    
    # 3. Controlled Benchmark Run
    for rep_name, X in representations.items():
        print(f"\nEvaluating: {rep_name}...")
        
        # Sort and split chronologically
        # Note: splits.py splits based on indexes, we reuse chronological indices
        total_rows = len(X)
        train_end = int(total_rows * 0.70)
        val_end = train_end + int(total_rows * 0.15)
        
        X_train = X.iloc[:train_end].copy()
        X_val = X.iloc[train_end:val_end].copy()
        X_test = X.iloc[val_end:].copy()
        
        y_train = y.iloc[:train_end].copy()
        y_val = y.iloc[train_end:val_end].copy()
        y_test = y.iloc[val_end:].copy()

        # Scale data for LR & MLP
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_train)
        X_va_scaled = scaler.transform(X_val)
        X_te_scaled = scaler.transform(X_test)
        
        X_train_num = X_train.astype(float)
        X_val_num = X_val.astype(float)
        X_test_num = X_test.astype(float)

        models_config = {
            "Logistic Regression": {
                "model": LogisticRegression(C=0.1, max_iter=1000, random_state=42),
                "scaled": True
            },
            "Random Forest": {
                "model": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1),
                "scaled": False
            },
            "XGBoost": {
                "model": XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42, n_jobs=-1),
                "scaled": False
            },
            "MLP (Neural Net)": {
                "model": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=300, random_state=42),
                "scaled": True
            }
        }

        rep_results = {}
        
        # Train and test evaluation
        for m_name, config in models_config.items():
            model = config["model"]
            train_x = X_tr_scaled if config["scaled"] else X_train_num
            test_x = X_te_scaled if config["scaled"] else X_test_num
            val_x = X_va_scaled if config["scaled"] else X_val_num
            
            t0 = time.time()
            model.fit(train_x, y_train)
            train_time = time.time() - t0
            
            t1 = time.time()
            preds = model.predict(test_x)
            inference_time_total = time.time() - t1
            latency = inference_time_total / len(test_x)

            probs = model.predict_proba(test_x)[:, 1] if hasattr(model, "predict_proba") else preds
            val_probs = model.predict_proba(val_x)[:, 1] if hasattr(model, "predict_proba") else model.predict(val_x)

            cal = evaluate_calibration(y_test.values, probs)
            
            metrics = {
                "test_accuracy": float(accuracy_score(y_test, preds)),
                "test_logloss": float(log_loss(y_test, probs, labels=[0,1])),
                "test_roc_auc": float(roc_auc_score(y_test, probs)),
                "test_f1_score": float(f1_score(y_test, preds, average="binary")),
                "brier_score": cal["brier_score"],
                "ece": cal["ece"],
                "train_time": train_time,
                "latency_ms": latency * 1000
            }
            
            rep_results[m_name] = metrics
            print(f"    - {m_name:20s}: Acc: {metrics['test_accuracy']:.4f} | AUC: {metrics['test_roc_auc']:.4f} | ECE: {metrics['ece']:.4f}")

        # 4. Statistical Confidence Bounds (10-fold CV on XGBoost as best baseline)
        print("  Running 10-fold Cross-Validation...")
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=42, n_jobs=-1),
            X.astype(float), y, cv=cv, scoring="accuracy", n_jobs=-1
        )
        
        mean_cv = float(np.mean(cv_scores))
        std_cv = float(np.std(cv_scores))
        ci_lower = mean_cv - (1.96 * std_cv)
        ci_upper = mean_cv + (1.96 * std_cv)
        
        rep_results["XGBoost_10fold_CV"] = {
            "mean_accuracy": mean_cv,
            "std_dev": std_cv,
            "confidence_interval": [ci_lower, ci_upper]
        }
        print(f"    - XGBoost 10-fold CV : {mean_cv*100:.2f}% +/- {std_cv*100:.2f}% (95% CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%])")

        results[rep_name] = rep_results

    # 5. Feature & Asymmetry Coefficients Analysis (on Rep B - One-Hot using Logistic Regression)
    print("\n[Step 2] Auditing Predictive Feature Coefficients (Rep B)...")
    lr_rep_b = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    scaler_b = StandardScaler()
    X_b_scaled = scaler_b.fit_transform(rep_b)
    lr_rep_b.fit(X_b_scaled, y)
    
    # Coefficients map to p1_has_* and p2_has_*
    coef_series = pd.Series(lr_rep_b.coef_[0], index=rep_b.columns)
    
    # Load card names from metadata mapping
    lib_path = PROJECT_ROOT / "features" / "card_library.json"
    with open(lib_path, "r") as f:
        cards_data = json.load(f)
        
    def get_card_name(cid_col):
        cid = cid_col.split("_has_")[-1]
        return cards_data.get(cid, {}).get("name", cid)

    # p1 positive indicators (cards that predict win when played by player 1)
    p1_pos = coef_series[coef_series.index.str.startswith("p1_")].sort_values(ascending=False).head(5)
    print("  • Top 5 Positive Player 1 Cards:")
    for col, val in p1_pos.items():
        print(f"    - {get_card_name(col)}: {val:.4f}")

    # p1 negative indicators (cards that predict loss when played by player 1)
    p1_neg = coef_series[coef_series.index.str.startswith("p1_")].sort_values(ascending=True).head(5)
    print("  • Top 5 Negative Player 1 Cards:")
    for col, val in p1_neg.items():
        print(f"    - {get_card_name(col)}: {val:.4f}")

    # Save compile benchmark report
    summary = {
        "benchmark_results": results,
        "feature_coefficients": {
            "top_positive_p1": {get_card_name(c): float(val) for c, val in p1_pos.items()},
            "top_negative_p1": {get_card_name(c): float(val) for c, val in p1_neg.items()}
        }
    }

    report_path = PROJECT_ROOT / "logs" / "representation_benchmark_results.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\n[+] Sprints 6 representation audit complete! Results stored in {report_path}\n")

if __name__ == "__main__":
    run_representation_benchmark()
