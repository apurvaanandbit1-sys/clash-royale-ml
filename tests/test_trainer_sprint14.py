import unittest
import torch
import numpy as np
import tempfile
import csv
import json
from pathlib import Path
from training.trainer import Trainer, CheckpointManager, EarlyStopping, MetricsLogger

class MockDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = [
            {
                "p1_deck": torch.zeros(8, dtype=torch.long),
                "p2_deck": torch.zeros(8, dtype=torch.long),
                "trophy_diff": torch.zeros(1, dtype=torch.float32),
                "target": torch.ones(1, dtype=torch.float32)
            }
        ] * 10
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]


class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.tensor([1.0]))
        # Need some mock linear layer to check grad norm
        self.linear = torch.nn.Linear(17, 1) # p1 (8) + p2 (8) + td (1) = 17
    def forward(self, p1, p2, td):
        # Concatenate mocks to pass linear
        x = torch.cat([p1.float(), p2.float(), td], dim=1)
        return self.linear(x)


class TestTrainerSprint14(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        
        self.model = MockModel()
        self.dataset = MockDataset()
        self.loader = torch.utils.data.DataLoader(self.dataset, batch_size=2)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.1)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=1, factor=0.5)
        
        self.c_manager = CheckpointManager(self.temp_path / "checkpoints")
        self.logger = MetricsLogger(self.temp_path / "metrics.csv", self.temp_path / "metrics.json")
        self.early_stop = EarlyStopping(patience=2, min_delta=0.01)
        
        self.config = {
            "optimizer": {"grad_clipping": 1.0},
            "checkpoint": {"save_every_epoch": True}
        }
        
        self.trainer = Trainer(
            model=self.model,
            train_loader=self.loader,
            val_loader=self.loader,
            criterion=torch.nn.BCEWithLogitsLoss(),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            checkpoint_manager=self.c_manager,
            logger=self.logger,
            early_stopping=self.early_stop,
            config=self.config,
            seed=42
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_trainer_initialization(self):
        self.assertEqual(self.trainer.seed, 42)
        self.assertEqual(self.trainer.best_val_loss, float("inf"))

    def test_checkpoint_save_and_load(self):
        metrics = {"loss": 0.5, "accuracy": 0.8}
        self.c_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=3,
            metrics=metrics,
            config=self.config,
            seed=42,
            name="test_ckpt.pt"
        )
        
        ckpt_file = self.temp_path / "checkpoints" / "test_ckpt.pt"
        self.assertTrue(ckpt_file.exists())
        
        # Modify model param to verify load updates it
        with torch.no_grad():
            self.model.param.copy_(torch.tensor([5.0]))
            
        # Load back
        new_opt = torch.optim.AdamW(self.model.parameters(), lr=0.1)
        new_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(new_opt, patience=1)
        
        info = self.c_manager.load_checkpoint(ckpt_file, self.model, new_opt, new_sched)
        
        self.assertEqual(info["epoch"], 3)
        self.assertEqual(info["seed"], 42)
        self.assertEqual(float(self.model.param.item()), 1.0) # Restored back to 1.0

    def test_early_stopping_trigger(self):
        early_stopping = EarlyStopping(patience=2, min_delta=0.01)
        self.assertFalse(early_stopping.early_stop)
        
        # Epoch 1: best loss = 0.5
        early_stopping(0.5)
        self.assertFalse(early_stopping.early_stop)
        
        # Epoch 2: no improvement
        early_stopping(0.51)
        self.assertFalse(early_stopping.early_stop)
        
        # Epoch 3: no improvement (triggers early stop)
        early_stopping(0.52)
        self.assertTrue(early_stopping.early_stop)

    def test_scheduler_step_plateau(self):
        orig_lr = self.optimizer.param_groups[0]["lr"]
        
        # Trigger plateau scheduler step twice with bad losses to trigger decay
        self.scheduler.step(0.9)
        self.scheduler.step(0.9)
        self.scheduler.step(0.9) # Patience is 1, so after 2 steps with no improvement, it decays
        
        decayed_lr = self.optimizer.param_groups[0]["lr"]
        self.assertLess(decayed_lr, orig_lr)

    def test_metrics_logging_files(self):
        metrics = {"loss": 0.45, "accuracy": 0.75, "roc_auc": 0.81}
        self.logger.log_epoch(1, metrics)
        
        self.assertTrue(self.logger.csv_path.exists())
        self.assertTrue(self.logger.json_path.exists())
        
        # Read JSON
        with open(self.logger.json_path, "r") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["epoch"], 1)
        self.assertEqual(data[0]["accuracy"], 0.75)

if __name__ == "__main__":
    unittest.main()
