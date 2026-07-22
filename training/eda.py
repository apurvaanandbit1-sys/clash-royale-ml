import pandas as pd
import numpy as np

def perform_eda(df: pd.DataFrame) -> dict:
    """
    Performs exploratory data analysis on the engineered features.
    Computes:
        - Feature descriptive statistics (mean, std, variance)
        - Collinearity / Feature redundancy (correlation threshold > 0.8)
        - Class balance target distribution
    """
    # Exclude non-feature columns
    non_feature_cols = ["win", "player_deck", "opponent_deck", "battle_time"]
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    features_df = df[feature_cols]

    # 1. Feature descriptive statistics
    stats = {}
    for col in feature_cols:
        # Convert boolean columns to integer before computing corr/stats
        if features_df[col].dtype == bool:
            features_df[col] = features_df[col].astype(int)
            
        stats[col] = {
            "mean": float(features_df[col].mean()),
            "std": float(features_df[col].std()),
            "var": float(features_df[col].var()),
            "min": float(features_df[col].min()),
            "max": float(features_df[col].max())
        }

    # Ensure all columns are numeric for correlation matrix
    numeric_df = features_df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method="pearson")

    # 2. Identify redundant pairs (r > 0.8)
    redundant_pairs = []
    columns_list = list(corr_matrix.columns)
    for i in range(len(columns_list)):
        for j in range(i + 1, len(columns_list)):
            col1 = columns_list[i]
            col2 = columns_list[j]
            r = corr_matrix.iloc[i, j]
            if abs(r) > 0.8:
                redundant_pairs.append({
                    "feature_1": col1,
                    "feature_2": col2,
                    "correlation": float(r)
                })

    # 3. Class balance
    total_samples = len(df)
    win_count = int((df["win"] == 1).sum())
    loss_count = int((df["win"] == 0).sum())

    return {
        "feature_stats": stats,
        "redundant_pairs": redundant_pairs,
        "class_balance": {
            "total": total_samples,
            "wins": win_count,
            "losses": loss_count,
            "win_ratio": float(win_count / total_samples) if total_samples > 0 else 0.0
        }
    }
