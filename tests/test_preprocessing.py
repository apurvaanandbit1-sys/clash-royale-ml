from pathlib import Path

import pandas as pd


def test_preprocessing():

    parquet_path = Path("matches_with_features.parquet")

    assert parquet_path.exists(), (
        "matches_with_features.parquet does not exist."
    )

    df = pd.read_parquet(parquet_path)

    # -------------------------------
    # Dataset exists
    # -------------------------------
    assert len(df) > 0, "Dataset is empty."

    # -------------------------------
    # Required columns
    # -------------------------------
    required_columns = [
        "player_deck",
        "opponent_deck",
        "win",
        "p1_average_elixir",
        "p2_average_elixir",
    ]

    for col in required_columns:
        assert col in df.columns, (
            f"Missing required column: {col}"
        )

    # -------------------------------
    # Missing values
    # -------------------------------
    assert df.isnull().sum().sum() == 0, (
        "Dataset contains missing values."
    )

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("✅ Preprocessing validation passed.")


if __name__ == "__main__":
    test_preprocessing()