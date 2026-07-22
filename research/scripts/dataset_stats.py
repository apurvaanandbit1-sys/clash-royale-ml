import os
import sys
import json
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

def compute_dataset_stats(parquet_path: str, card_library_path: str):
    """
    Computes statistical properties of the Clash Royale dataset for validation and debugging.
    """
    parquet_file = Path(parquet_path)
    if not parquet_file.exists():
        print(f"Error: Parquet file not found at '{parquet_path}'!")
        sys.exit(1)
        
    df = pd.read_parquet(parquet_file)
    
    # 1. Basic Stats
    total_battles = len(df)
    
    # 2. Decks and unique counts
    all_decks = []
    card_counts = Counter()
    
    for row in df.itertuples():
        p1_deck = json.loads(row.player_deck)
        p2_deck = json.loads(row.opponent_deck)
        
        all_decks.append(row.player_deck)
        all_decks.append(row.opponent_deck)
        
        card_counts.update(p1_deck)
        card_counts.update(p2_deck)
        
    unique_decks_count = len(set(all_decks))
    
    # Load card names
    card_names = {}
    if os.path.exists(card_library_path):
        with open(card_library_path, "r") as f:
            cards_data = json.load(f)
            card_names = {str(cid): info.get("name", str(cid)) for cid, info in cards_data.items()}

    # 3. Winner balance
    wins = int((df["win"] == 1).sum())
    losses = int((df["win"] == 0).sum())
    win_rate = wins / total_battles if total_battles > 0 else 0.0
    
    # 4. Trophies
    avg_p1_trophies = df["player_trophies"].mean() if "player_trophies" in df.columns else 0.0
    avg_p2_trophies = df["opponent_trophies"].mean() if "opponent_trophies" in df.columns else 0.0
    avg_trophy_diff = (df["player_trophies"] - df["opponent_trophies"]).mean() if ("player_trophies" in df.columns and "opponent_trophies" in df.columns) else 0.0
    
    print("=" * 60)
    print("         CLASH ROYALE DATASET STATISTICS AUDIT")
    print("=" * 60)
    print(f"  • Total Battles:             {total_battles:,}")
    print(f"  • Unique Decks:              {unique_decks_count:,}")
    print(f"  • Unique Decks Ratio:        {unique_decks_count / (2 * total_battles) * 100:.2f}%")
    print(f"  • Target Win Balance:        Win: {wins:,} ({win_rate*100:.2f}%) | Loss: {losses:,} ({(1-win_rate)*100:.2f}%)")
    print(f"  • Avg Player Trophies:       {avg_p1_trophies:.1f}")
    print(f"  • Avg Opponent Trophies:     {avg_p2_trophies:.1f}")
    print(f"  • Avg Trophy Difference:     {avg_trophy_diff:.1f}")
    
    # Missing values
    missing = df.isnull().sum().to_dict()
    print(f"  • Missing Values Audit:      {missing}")
    
    print("\n  • Top 10 Most Played Cards:")
    top_cards = card_counts.most_common(10)
    for cid, count in top_cards:
        name = card_names.get(str(cid), str(cid))
        appearance_rate = count / (2 * total_battles) * 100
        print(f"    - {name:20s}: {count:,} appearances ({appearance_rate:.2f}%)")
        
    print("=" * 60)

if __name__ == "__main__":
    import yaml
    config_path = PROJECT_ROOT / "config" / "dataset_config.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    compute_dataset_stats(
        parquet_path=config["data"]["dataset_path"],
        card_library_path=config["data"]["card_library_path"]
    )
