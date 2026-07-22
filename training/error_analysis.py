import sqlite3
import pandas as pd
import numpy as np
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "clashroyale.db"

def analyze_prediction_errors(
    test_df: pd.DataFrame, 
    y_test: pd.Series, 
    predictions: np.ndarray,
    probabilities: np.ndarray
) -> dict:
    """
    Performs error analysis by grouping prediction failures into different cohorts:
        - Trophy cohorts (<8000, 8000-9000, >9000)
        - Game Arenas
        - Presence of variable elixir cards (Mirror)
        - Presence of Champions
    """
    # 1. Fetch metadata columns from SQLite
    meta_df = pd.DataFrame()
    if DB_PATH.exists():
        try:
            conn = sqlite3.connect(DB_PATH)
            meta_df = pd.read_sql_query(
                "SELECT player_deck, opponent_deck, player_trophies, opponent_trophies, arena_name, battle_time FROM battles ORDER BY battle_time ASC", 
                conn
            )
            conn.close()
        except Exception as e:
            print(f"[Error Analysis] DB Load failed: {e}")
            
    if meta_df.empty:
        print("[Error Analysis] WARNING: Metadata is empty. Error analysis slices will be skipped.")
        return {}

    # Since test_df is the last 15% chronologically, slice meta_df to align with test_df indices
    test_start_idx = len(meta_df) - len(test_df)
    test_meta = meta_df.iloc[test_start_idx:].reset_index(drop=True)
    
    # 2. Build analysis DataFrame
    analysis_df = pd.DataFrame({
        "win_true": y_test.values,
        "win_pred": predictions,
        "prob_pred": probabilities,
        "correct": (y_test.values == predictions).astype(int),
        "player_trophies": test_meta["player_trophies"].values,
        "opponent_trophies": test_meta["opponent_trophies"].values,
        "arena_name": test_meta["arena_name"].values,
        "player_deck": test_meta["player_deck"].values,
        "opponent_deck": test_meta["opponent_deck"].values
    })

    # Add features for special cards (Mirror ID=28000006)
    analysis_df["has_mirror"] = analysis_df.apply(
        lambda r: ("28000006" in r["player_deck"] or "28000006" in r["opponent_deck"]),
        axis=1
    )

    # 3. Slices
    # Trophy Buckets
    def get_trophy_bucket(trophies):
        if trophies < 8000:
            return "< 8,000"
        elif trophies <= 9000:
            return "8,000 - 9,000"
        else:
            return "> 9,000"
            
    analysis_df["trophy_bucket"] = analysis_df["player_trophies"].apply(get_trophy_bucket)
    
    trophy_analysis = analysis_df.groupby("trophy_bucket")["correct"].agg(["count", "mean"]).to_dict("index")
    # Error rate = 1 - mean (accuracy)
    trophy_errors = {
        bucket: {
            "samples": int(stats["count"]),
            "error_rate": float(1.0 - stats["mean"])
        } for bucket, stats in trophy_analysis.items()
    }

    # Arena Slices (Top 5)
    arena_analysis = analysis_df.groupby("arena_name")["correct"].agg(["count", "mean"])
    top_arenas = arena_analysis.sort_values(by="count", ascending=False).head(5).to_dict("index")
    arena_errors = {
        arena: {
            "samples": int(stats["count"]),
            "error_rate": float(1.0 - stats["mean"])
        } for arena, stats in top_arenas.items()
    }

    # Special Card Slices (Mirror)
    mirror_analysis = analysis_df.groupby("has_mirror")["correct"].agg(["count", "mean"]).to_dict("index")
    mirror_errors = {
        str(has_mirror): {
            "samples": int(stats["count"]),
            "error_rate": float(1.0 - stats["mean"])
        } for has_mirror, stats in mirror_analysis.items()
    }

    # 4. Calibration analysis (average error for confident vs unconfident predictions)
    # Confident: prob > 0.7 or prob < 0.3
    analysis_df["confident"] = ((analysis_df["prob_pred"] > 0.7) | (analysis_df["prob_pred"] < 0.3)).astype(int)
    conf_analysis = analysis_df.groupby("confident")["correct"].agg(["count", "mean"]).to_dict("index")
    confidence_errors = {
        str(conf): {
            "samples": int(stats["count"]),
            "error_rate": float(1.0 - stats["mean"])
        } for conf, stats in conf_analysis.items()
    }

    return {
        "trophy_errors": trophy_errors,
        "arena_errors": arena_errors,
        "mirror_errors": mirror_errors,
        "confidence_errors": confidence_errors
    }
