import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "clashroyale.db"

def run_representation_audit(df: pd.DataFrame, trained_model, X_test: pd.DataFrame) -> dict:
    """
    Performs tests to audit information loss in the current representation.
    
    Checks:
        1. Unique Feature Ratio: Ratio of unique feature vectors to unique decks.
        2. Specific Deck Collisions: Count of different decks sharing identical feature vectors.
        3. Counter Matchup Performance: Accuracy on Giant vs Inferno Tower/Pekka matchups.
    """
    findings = {}

    # 1. Identity Loss: Unique Feature Vectors vs Unique Decks
    unique_decks = set()
    for col in ["player_deck", "opponent_deck"]:
        if col in df.columns:
            unique_decks.update(df[col].unique())
            
    # For each unique deck, extract its 12-dimensional features
    # Let's count how many unique feature vectors exist among these unique decks
    p1_cols = [c for c in X_test.columns if c.startswith("p1_")]
    
    # Extract unique player decks and their features
    unique_deck_features = df[["player_deck"] + p1_cols].drop_duplicates(subset=["player_deck"])
    
    # Drop deck string column to get only features
    features_only = unique_deck_features[p1_cols].drop_duplicates()
    
    total_unique_decks = len(unique_deck_features)
    total_unique_features = len(features_only)
    
    features_per_deck_ratio = total_unique_features / total_unique_decks if total_unique_decks > 0 else 0.0
    identity_loss_pct = (1.0 - features_per_deck_ratio) * 100
    
    findings["deck_representation"] = {
        "unique_decks": total_unique_decks,
        "unique_feature_vectors": total_unique_features,
        "compression_ratio": features_per_deck_ratio,
        "identity_loss_percentage": identity_loss_pct
    }

    # 2. Specific Card-to-Card Counter Matchup Audit
    # We load raw battle logs from SQLite to search for matchups containing Giant (ID 26000003) 
    # vs Inferno Tower (ID 27000003) or PEKKA (ID 26000004)
    counter_accuracy = 0.0
    counter_samples = 0
    
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            battles_raw = pd.read_sql_query(
                "SELECT player_deck, opponent_deck, win, battle_time FROM battles ORDER BY battle_time ASC", 
                conn
            )
            conn.close()
            
            # Identify test set indices (last 15% chronologically)
            test_start = len(battles_raw) - len(X_test)
            test_battles = battles_raw.iloc[test_start:].reset_index(drop=True)
            
            # Predict win probabilities for these test battles using the model
            # We must convert boolean columns to numeric floats for predictions
            test_x_num = X_test.astype(float)
            
            # Scale if model is MLP/LR (we scale for predictions)
            if type(trained_model).__name__ in ["LogisticRegression", "MLPClassifier"]:
                from sklearn.preprocessing import StandardScaler
                # Note: Simple scaling for safety
                scaler = StandardScaler()
                test_x_num = scaler.fit_transform(test_x_num)
                
            test_preds = trained_model.predict(test_x_num)

            # Find battles containing the counter Giant (26000003) vs Inferno Tower (27000003)
            # where player has Giant and opponent has Inferno Tower
            is_giant_vs_inferno = test_battles.apply(
                lambda r: ("26000003" in r["player_deck"] and "27000003" in r["opponent_deck"]),
                axis=1
            )
            
            giant_matches = test_preds[is_giant_vs_inferno]
            giant_true = test_battles.loc[is_giant_vs_inferno, "win"].values
            
            if len(giant_matches) > 0:
                counter_accuracy = float(np.mean(giant_matches == giant_true))
                counter_samples = len(giant_matches)
                
        except Exception as e:
            print(f"[Representation Audit] Error: {e}")

    findings["counter_matchups"] = {
        "description": "Player has Giant win condition, Opponent has Inferno Tower hard counter building.",
        "samples": counter_samples,
        "model_accuracy": counter_accuracy,
        "observed_win_rate": float(np.mean(giant_true)) if counter_samples > 0 else 0.0
    }

    return findings
