import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class ClashRoyaleDataset(Dataset):
    """
    Production-quality PyTorch Dataset for Clash Royale match prediction.
    Performs data cleaning, validation, card ID mapping, and symmetry data augmentation.
    """
    def __init__(
        self, 
        df: pd.DataFrame, 
        card_library_path: str | Path,
        augment: bool = False,
        augment_prob: float = 0.5
    ):
        self.augment = augment
        self.augment_prob = augment_prob
        
        # Load and parse card library mapping
        self.card_library_path = Path(card_library_path)
        if not self.card_library_path.exists():
            raise FileNotFoundError(f"Card library not found at: {self.card_library_path}")
            
        with open(self.card_library_path, "r") as f:
            cards_data = json.load(f)
        
        # Sort keys to ensure stable mapping [0, C-1]
        sorted_card_ids = sorted(list(cards_data.keys()))
        self.card_to_idx = {str(cid): idx for idx, cid in enumerate(sorted_card_ids)}
        
        # Validate and clean data
        self.df = self._validate_and_clean(df)
        
        # Pre-parse inputs into memory for fast indexing
        self.targets = torch.tensor(self.df["win"].values, dtype=torch.float32)
        
        # Map decks
        p1_idx_list = []
        p2_idx_list = []
        for row in self.df.itertuples():
            p1_ids = [self.card_to_idx[str(c)] for c in json.loads(row.player_deck)]
            p2_ids = [self.card_to_idx[str(c)] for c in json.loads(row.opponent_deck)]
            p1_idx_list.append(p1_ids)
            p2_idx_list.append(p2_ids)
            
        self.p1_decks = torch.tensor(p1_idx_list, dtype=torch.long)
        self.p2_decks = torch.tensor(p2_idx_list, dtype=torch.long)
        
        # Trophies
        self.p1_trophies = self.df["player_trophies"].values if "player_trophies" in self.df.columns else np.zeros(len(self.df))
        self.p2_trophies = self.df["opponent_trophies"].values if "opponent_trophies" in self.df.columns else np.zeros(len(self.df))
        
    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates the incoming dataframe and returns the cleaned records.
        Raises ValueError for invalid shapes or critical data corruptions.
        """
        if df.empty:
            raise ValueError("Input dataframe is empty!")

        required_cols = ["player_deck", "opponent_deck", "win"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' is missing from the dataset!")

        cleaned_rows = []
        for idx, row in enumerate(df.itertuples()):
            # 1. Missing labels
            if pd.isnull(row.win):
                raise ValueError(f"Row {idx}: Missing winner label (win is NaN)!")
            if int(row.win) not in [0, 1]:
                raise ValueError(f"Row {idx}: Invalid winner label '{row.win}' (must be 0 or 1)!")

            # 2. Check trophies (if present)
            if "player_trophies" in df.columns:
                if pd.isnull(row.player_trophies) or row.player_trophies < 0:
                    raise ValueError(f"Row {idx}: Invalid player trophy value '{row.player_trophies}'!")
            if "opponent_trophies" in df.columns:
                if pd.isnull(row.opponent_trophies) or row.opponent_trophies < 0:
                    raise ValueError(f"Row {idx}: Invalid opponent trophy value '{row.opponent_trophies}'!")

            # 3. Parse Decks and validate deck structures
            try:
                p1_deck = json.loads(row.player_deck)
                p2_deck = json.loads(row.opponent_deck)
            except Exception as e:
                raise ValueError(f"Row {idx}: Failed to parse JSON deck strings. Error: {e}")

            # Validate lengths
            if len(p1_deck) != 8:
                raise ValueError(f"Row {idx}: Player deck must contain exactly 8 cards, got {len(p1_deck)}: {p1_deck}")
            if len(p2_deck) != 8:
                raise ValueError(f"Row {idx}: Opponent deck must contain exactly 8 cards, got {len(p2_deck)}: {p2_deck}")

            # Validate duplicates in a single deck
            if len(set(p1_deck)) != 8:
                raise ValueError(f"Row {idx}: Duplicate cards detected in Player deck: {p1_deck}")
            if len(set(p2_deck)) != 8:
                raise ValueError(f"Row {idx}: Duplicate cards detected in Opponent deck: {p2_deck}")

            # 4. Check for missing card IDs in library
            for card in p1_deck + p2_deck:
                if str(card) not in self.card_to_idx:
                    raise ValueError(f"Row {idx}: Card ID '{card}' is missing from the card library mapping registry!")

            cleaned_rows.append(row._asdict())

        return pd.DataFrame(cleaned_rows)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict:
        p1 = self.p1_decks[idx]
        p2 = self.p2_decks[idx]
        
        t_diff = float(self.p1_trophies[idx] - self.p2_trophies[idx])
        target = float(self.targets[idx])

        # Optional symmetry augmentation
        if self.augment and (np.random.rand() < self.augment_prob):
            p1, p2 = p2, p1
            t_diff = -t_diff
            target = 1.0 - target

        return {
            "p1_deck": p1,                                      # Shape [8] (Long)
            "p2_deck": p2,                                      # Shape [8] (Long)
            "trophy_diff": torch.tensor([t_diff], dtype=torch.float32), # Shape [1]
            "target": torch.tensor([target], dtype=torch.float32)       # Shape [1]
        }


def get_dataloaders(
    df: pd.DataFrame,
    card_library_path: str | Path,
    batch_size: int = 256,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    augment_prob: float = 0.5,
    num_workers: int = 0,
    pin_memory: bool = True
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    DataLoader factory. Performs chronological splitting (70% train, 15% val, 15% test),
    and builds PyTorch DataLoader instances. Augmentation is active ONLY on the training loader.
    """
    # 1. Sort chronologically
    if "battle_time" in df.columns:
        df_sorted = df.sort_values(by="battle_time", ascending=True).reset_index(drop=True)
    else:
        df_sorted = df.copy()

    total_rows = len(df_sorted)
    train_end = int(total_rows * (1.0 - val_ratio - test_ratio))
    val_end = train_end + int(total_rows * val_ratio)

    df_train = df_sorted.iloc[:train_end].reset_index(drop=True)
    df_val = df_sorted.iloc[train_end:val_end].reset_index(drop=True)
    df_test = df_sorted.iloc[val_end:].reset_index(drop=True)

    # 2. Build datasets
    train_ds = ClashRoyaleDataset(df_train, card_library_path, augment=True, augment_prob=augment_prob)
    val_ds = ClashRoyaleDataset(df_val, card_library_path, augment=False)
    test_ds = ClashRoyaleDataset(df_test, card_library_path, augment=False)

    # Seed generator for reproducibility
    g = torch.Generator()
    g.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)

    # 3. Create loaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
