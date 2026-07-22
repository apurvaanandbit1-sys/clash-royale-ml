import pandas as pd
import numpy as np

def split_dataset(df: pd.DataFrame, train_ratio=0.70, val_ratio=0.15) -> tuple:
    """
    Splits the dataset chronologically based on 'battle_time' column.
    
    Parameters:
        df: pd.DataFrame containing the dataset.
        train_ratio: Float, proportion of training samples (default: 0.70).
        val_ratio: Float, proportion of validation samples (default: 0.15).
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # 1. Sort chronologically by battle_time
    if "battle_time" in df.columns:
        df_sorted = df.sort_values(by="battle_time", ascending=True).reset_index(drop=True)
    else:
        print("WARNING: 'battle_time' not found. Splitting in natural order.")
        df_sorted = df.copy()

    total_rows = len(df_sorted)
    train_end = int(total_rows * train_ratio)
    val_end = train_end + int(total_rows * val_ratio)

    # 2. Separate features (X) and target (y)
    y = df_sorted["win"]
    
    # Drop non-feature columns
    columns_to_drop = ["win", "player_deck", "opponent_deck", "battle_time"]
    X = df_sorted.drop(columns=[col for col in columns_to_drop if col in df_sorted.columns])

    # 3. Create splits
    X_train = X.iloc[:train_end].copy()
    y_train = y.iloc[:train_end].copy()

    X_val = X.iloc[train_end:val_end].copy()
    y_val = y.iloc[train_end:val_end].copy()

    X_test = X.iloc[val_end:].copy()
    y_test = y.iloc[val_end:].copy()

    # 4. Leakage Verification Checks
    # Assert index overlap is zero
    assert len(set(X_train.index).intersection(set(X_val.index))) == 0, "Train and Val indexes overlap!"
    assert len(set(X_val.index).intersection(set(X_test.index))) == 0, "Val and Test indexes overlap!"
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0, "Train and Test indexes overlap!"

    print(f"[Splits] Chronological partition complete:")
    print(f"  • Train:      {len(X_train):,} samples ({len(X_train)/total_rows*100:.1f}%)")
    print(f"  • Validation: {len(X_val):,} samples ({len(X_val)/total_rows*100:.1f}%)")
    print(f"  • Test:       {len(X_test):,} samples ({len(X_test)/total_rows*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test
