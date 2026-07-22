import unittest
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from training.dataset import ClashRoyaleDataset, get_dataloaders

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class TestClashRoyaleDataset(unittest.TestCase):
    def setUp(self):
        # Create a tiny mock card library config
        self.mock_card_lib = PROJECT_ROOT / "config" / "mock_card_library.json"
        self.mock_card_lib.parent.mkdir(parents=True, exist_ok=True)
        
        # Populate library with 10 mock card keys
        cards = {str(26000000 + i): {"id": 26000000 + i, "name": f"Card_{i}"} for i in range(10)}
        with open(self.mock_card_lib, "w") as f:
            json.dump(cards, f)

        # Build valid mock matchups DataFrame
        self.valid_df = pd.DataFrame({
            "player_deck": [json.dumps([26000000 + i for i in range(8)])],
            "opponent_deck": [json.dumps([26000000 + i for i in range(1, 9)])],
            "win": [1.0],
            "player_trophies": [8000],
            "opponent_trophies": [7500],
            "battle_time": ["2026-07-17 12:00:00"]
        })

    def tearDown(self):
        if self.mock_card_lib.exists():
            self.mock_card_lib.unlink()

    def test_valid_dataset_init_and_shapes(self):
        dataset = ClashRoyaleDataset(self.valid_df, self.mock_card_lib, augment=False)
        self.assertEqual(len(dataset), 1)
        
        sample = dataset[0]
        # Verify keys
        self.assertIn("p1_deck", sample)
        self.assertIn("p2_deck", sample)
        self.assertIn("trophy_diff", sample)
        self.assertIn("target", sample)
        
        # Verify shapes & types
        self.assertEqual(sample["p1_deck"].shape, (8,))
        self.assertEqual(sample["p1_deck"].dtype, torch.long)
        self.assertEqual(sample["p2_deck"].shape, (8,))
        self.assertEqual(sample["p2_deck"].dtype, torch.long)
        self.assertEqual(sample["trophy_diff"].shape, (1,))
        self.assertEqual(sample["trophy_diff"].dtype, torch.float32)
        self.assertEqual(sample["target"].shape, (1,))
        self.assertEqual(sample["target"].dtype, torch.float32)
        
        # Verify values
        self.assertEqual(float(sample["trophy_diff"][0]), 500.0)
        self.assertEqual(float(sample["target"][0]), 1.0)

    def test_symmetry_augmentation(self):
        # Force augmentation to always trigger (augment_prob=1.0)
        dataset = ClashRoyaleDataset(self.valid_df, self.mock_card_lib, augment=True, augment_prob=1.0)
        
        # Seed to verify swap determinism (though augment_prob=1.0 is deterministic swap)
        np.random.seed(42)
        sample = dataset[0]
        
        # Verify swapped decks, negated trophies, and inverted target
        # Original: p1=[0..7], p2=[1..8], diff=500, win=1
        # Swapped: p1=[1..8], p2=[0..7], diff=-500, win=0
        self.assertEqual(list(sample["p1_deck"].numpy()), [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(list(sample["p2_deck"].numpy()), [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(float(sample["trophy_diff"][0]), -500.0)
        self.assertEqual(float(sample["target"][0]), 0.0)

    def test_invalid_deck_length_check(self):
        invalid_df = self.valid_df.copy()
        invalid_df["player_deck"] = [json.dumps([26000000, 26000001])] # Only 2 cards
        
        with self.assertRaises(ValueError):
            ClashRoyaleDataset(invalid_df, self.mock_card_lib)

    def test_duplicate_cards_check(self):
        invalid_df = self.valid_df.copy()
        # Deck contains duplicate card ID 26000000
        invalid_df["player_deck"] = [json.dumps([26000000, 26000000, 26000002, 26000003, 26000004, 26000005, 26000006, 26000007])]
        
        with self.assertRaises(ValueError):
            ClashRoyaleDataset(invalid_df, self.mock_card_lib)

    def test_invalid_trophies_check(self):
        invalid_df = self.valid_df.copy()
        invalid_df["player_trophies"] = [-100] # Negative trophies
        
        with self.assertRaises(ValueError):
            ClashRoyaleDataset(invalid_df, self.mock_card_lib)

    def test_missing_labels_check(self):
        invalid_df = self.valid_df.copy()
        invalid_df["win"] = [np.nan] # Missing label
        
        with self.assertRaises(ValueError):
            ClashRoyaleDataset(invalid_df, self.mock_card_lib)

    def test_dataloader_builder(self):
        # Create a mock dataframe with 10 matches
        df_large = pd.DataFrame({
            "player_deck": [json.dumps([26000000 + i for i in range(8)])] * 10,
            "opponent_deck": [json.dumps([26000000 + i for i in range(1, 9)])] * 10,
            "win": [1.0] * 10,
            "player_trophies": [8000] * 10,
            "opponent_trophies": [7500] * 10,
            "battle_time": [f"2026-07-17 12:00:0{i}" for i in range(10)]
        })
        
        train_ldr, val_ldr, test_ldr = get_dataloaders(
            df_large, self.mock_card_lib, batch_size=4, val_ratio=0.2, test_ratio=0.2
        )
        
        # Splits: 10 matches -> 6 train, 2 val, 2 test
        # Verify sizes
        self.assertEqual(len(train_ldr.dataset), 6)
        self.assertEqual(len(val_ldr.dataset), 2)
        self.assertEqual(len(test_ldr.dataset), 2)
        
        # Verify iteration
        for batch in train_ldr:
            self.assertEqual(batch["p1_deck"].shape[0], 4) # batch size 4
            break
            
if __name__ == "__main__":
    unittest.main()
