import pandas as pd
import numpy as np

def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Performs data validation on the loaded dataframe.
    Checks:
        - Row and feature counts
        - Missing values
        - Duplicate rows
        - Duplicate battles (matchups)
        - Class balance
        - Value boundary violations
    """
    findings = {}

    # 1. Dimensions
    findings["row_count"] = len(df)
    findings["feature_count"] = len(df.columns)

    # 2. Missing values
    missing_counts = df.isnull().sum().to_dict()
    findings["missing_values"] = {k: int(v) for k, v in missing_counts.items() if v > 0}

    # 3. Duplicate rows (identical in all features and target)
    findings["duplicate_rows_count"] = int(df.duplicated().sum())

    # 4. Duplicate battles (identical matchup, regardless of order)
    if "player_deck" in df.columns and "opponent_deck" in df.columns:
        matchups = df.apply(
            lambda r: "_".join(sorted([r["player_deck"], r["opponent_deck"]])), 
            axis=1
        )
        findings["duplicate_matchups_count"] = int(matchups.duplicated().sum())
    else:
        findings["duplicate_matchups_count"] = 0

    # 5. Class balance
    win_counts = df["win"].value_counts().to_dict()
    findings["class_balance"] = {int(k): int(v) for k, v in win_counts.items()}

    # 6. Value boundaries check (elixir, counts, indices)
    invalid_boundaries = {}
    for col in df.columns:
        if col.endswith("average_elixir"):
            violators = df[(df[col] < 1.0) | (df[col] > 9.0)]
            if len(violators) > 0:
                invalid_boundaries[col] = len(violators)
        elif col.endswith("count") or col.endswith("index"):
            violators = df[df[col] < 0]
            if len(violators) > 0:
                invalid_boundaries[col] = len(violators)
    findings["invalid_values"] = invalid_boundaries

    return findings
