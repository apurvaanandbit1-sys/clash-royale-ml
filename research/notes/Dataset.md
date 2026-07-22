# Dataset Documentation

This document describes the schema, preprocessing, splitting, and augmentation strategies used for the Clash Royale matchup database.

## 1. Schema & Features

The raw data is parsed from competitive ladder play. The compiled parquet file `matches_with_features.parquet` contains the following fields:

*   `player_deck`: JSON string containing a list of 8 unique card IDs for Player A (e.g. `"[26000000, 26000001, ...]"`).
*   `opponent_deck`: JSON string containing a list of 8 unique card IDs for Player B.
*   `player_trophies`: Integer trophy count of Player A.
*   `opponent_trophies`: Integer trophy count of Player B.
*   `win`: Binary target label (1.0 if Player A won, 0.0 if Player B won).
*   `battle_time`: Chronological timestamp of the battle.

---

## 2. Preprocessing & Cleaning

During dataset parsing (`training/dataset.py`):
1.  **Card ID Verification**: Any match with a deck containing other than exactly 8 unique card IDs is discarded.
2.  **Vocabulary Mapping**: Card IDs are mapped to continuous stable indices `[0, C-1]` based on keys in `data/card_library.json`.
3.  **Trophy Check**: Negative trophy counts or missing targets are discarded.

---

## 3. Chronological Splitting

Matches are sorted by `battle_time` before splitting:
*   **Training Set**: First 70% of matches.
*   **Validation Set**: Next 15% of matches.
*   **Test Set**: Final 15% of matches.

Sorting prevents temporal data leakage and ensures evaluation simulates real-world future deck matchups.

---

## 4. Symmetry Augmentation

To enforce mathematical anti-symmetry, training samples are augmented with a probability of 50%:
*   Deck A and Deck B are swapped.
*   The target win label is inverted ($y \leftrightarrow 1-y$).
*   Trophy difference is negated ($\Delta T \leftrightarrow -\Delta T$).
This guarantees the loss function evaluates symmetric counterparts equally.
