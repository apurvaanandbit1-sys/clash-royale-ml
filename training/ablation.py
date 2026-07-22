import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

FEATURE_GROUPS = {
    "Elixir": ["average_elixir"],
    "Durability": ["durability_index", "building_count"],
    "Damage/Offense": ["damage_index", "win_condition_count", "splash_count"],
    "Spell": ["spell_count", "has_big_spell", "has_small_spell"],
    "Structure": ["has_evolution", "has_champion", "air_hitting_count"]
}

def run_ablation_study(
    X_train: pd.DataFrame, 
    X_val: pd.DataFrame, 
    X_test: pd.DataFrame, 
    y_train: pd.Series, 
    y_val: pd.Series, 
    y_test: pd.Series,
    random_seed: int = 42
) -> dict:
    """
    Runs systematic ablation experiments.
    For each feature group:
        1. Drops p1_ and p2_ columns belonging to the group.
        2. Trains Logistic Regression and XGBoost.
        3. Measures accuracy degradation on the test partition.
    """
    results = {}

    # Standard scale helper for LR
    from sklearn.preprocessing import StandardScaler
    
    # 0. Baseline (Full Features)
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train)
    X_te_scaled = scaler.transform(X_test)
    
    lr_full = LogisticRegression(C=0.1, max_iter=1000, random_state=random_seed)
    lr_full.fit(X_tr_scaled, y_train)
    lr_full_acc = accuracy_score(y_test, lr_full.predict(X_te_scaled))

    xgb_full = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=random_seed, n_jobs=-1)
    xgb_full.fit(X_train.astype(float), y_train)
    xgb_full_acc = accuracy_score(y_test, xgb_full.predict(X_test.astype(float)))

    results["Full Features"] = {
        "lr_accuracy": float(lr_full_acc),
        "xgb_accuracy": float(xgb_full_acc),
        "lr_degradation": 0.0,
        "xgb_degradation": 0.0
    }

    # 1. Ablate groups one by one
    for group_name, cols in FEATURE_GROUPS.items():
        # Identify both player and opponent columns
        cols_to_drop = []
        for col in cols:
            for prefix in ["p1_", "p2_"]:
                full_col = prefix + col
                if full_col in X_train.columns:
                    cols_to_drop.append(full_col)
                    
        # Create ablated sets
        X_train_abl = X_train.drop(columns=cols_to_drop)
        X_test_abl = X_test.drop(columns=cols_to_drop)
        
        # Scale for Logistic Regression
        scaler_abl = StandardScaler()
        X_tr_scaled_abl = scaler_abl.fit_transform(X_train_abl)
        X_te_scaled_abl = scaler_abl.transform(X_test_abl)

        # Train & Evaluate LR
        lr = LogisticRegression(C=0.1, max_iter=1000, random_state=random_seed)
        lr.fit(X_tr_scaled_abl, y_train)
        lr_acc = accuracy_score(y_test, lr.predict(X_te_scaled_abl))

        # Train & Evaluate XGBoost
        xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric="logloss", random_state=random_seed, n_jobs=-1)
        xgb.fit(X_train_abl.astype(float), y_train)
        xgb_acc = accuracy_score(y_test, xgb.predict(X_test_abl.astype(float)))

        results[f"Ablated {group_name}"] = {
            "lr_accuracy": float(lr_acc),
            "xgb_accuracy": float(xgb_acc),
            "lr_degradation": float(lr_full_acc - lr_acc),
            "xgb_degradation": float(xgb_full_acc - xgb_acc)
        }

    return results
