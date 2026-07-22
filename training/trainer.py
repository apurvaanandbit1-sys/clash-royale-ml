import os
import csv
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

class EarlyStopping:
    """
    Early stopping helper to terminate training when validation loss stops improving.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class CheckpointManager:
    """
    Manages saving and loading of model training checkpoints.
    Saves model, optimizer, scheduler state, epochs, and hyperparameters.
    """
    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler._LRScheduler | None, 
        epoch: int, 
        metrics: dict, 
        config: dict, 
        seed: int, 
        name: str = "latest.pt"
    ):
        state = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "metrics": metrics,
            "config": config,
            "seed": seed
        }
        filepath = self.checkpoint_dir / name
        torch.save(state, filepath)

    def load_checkpoint(
        self, 
        filepath: str | Path, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer = None, 
        scheduler: torch.optim.lr_scheduler._LRScheduler = None
    ) -> dict:
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {filepath}")
            
        state = torch.load(filepath, map_location=torch.device("cpu"))
        
        # Load states
        model.load_state_dict(state["model_state"])
        if optimizer and "optimizer_state" in state:
            optimizer.load_state_dict(state["optimizer_state"])
        if scheduler and state["scheduler_state"] is not None:
            scheduler.load_state_dict(state["scheduler_state"])
            
        return {
            "epoch": state["epoch"],
            "metrics": state["metrics"],
            "config": state["config"],
            "seed": state["seed"]
        }


class MetricsLogger:
    """
    Logs epoch metrics to CSV and JSON formats.
    """
    def __init__(self, csv_path: str | Path, json_path: str | Path):
        self.csv_path = Path(csv_path)
        self.json_path = Path(json_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.records = []

    def log_epoch(self, epoch: int, metrics: dict):
        record = {"epoch": epoch, **metrics}
        self.records.append(record)
        
        # Save to CSV
        df = pd.DataFrame(self.records)
        df.to_csv(self.csv_path, index=False)
        
        # Save to JSON
        with open(self.json_path, "w") as f:
            json.dump(self.records, f, indent=4)


class Trainer:
    """
    General, architecture-independent Trainer class.
    Manages epochs, backpropagation, validation metrics, early stopping, and callbacks.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        checkpoint_manager: CheckpointManager,
        logger: MetricsLogger,
        early_stopping: EarlyStopping,
        config: dict,
        seed: int
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger
        self.early_stopping = early_stopping
        self.config = config
        self.seed = seed
        
        self.best_val_loss = float("inf")

    def train_epoch(self) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        batches = 0
        
        grad_clip = self.config.get("optimizer", {}).get("grad_clipping", None)

        for batch in self.train_loader:
            self.optimizer.zero_grad()
            
            p1 = batch["p1_deck"]
            p2 = batch["p2_deck"]
            td = batch["trophy_diff"]
            target = batch["target"]
            
            # Forward pass
            out = self.model(p1, p2, td)
            loss = self.criterion(out, target)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                
            # Compute gradient norm
            grad_norm = sum(p.grad.detach().data.norm().item()**2 for p in self.model.parameters() if p.grad is not None)**0.5
            total_grad_norm += grad_norm
            
            self.optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        return total_loss / batches, total_grad_norm / batches

    def val_epoch(self) -> dict:
        self.model.eval()
        total_loss = 0.0
        batches = 0
        
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                p1 = batch["p1_deck"]
                p2 = batch["p2_deck"]
                td = batch["trophy_diff"]
                target = batch["target"]
                
                out = self.model(p1, p2, td)
                loss = self.criterion(out, target)
                
                probs = torch.sigmoid(out)
                
                total_loss += loss.item()
                all_targets.append(target.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                batches += 1
                
        y_true = np.vstack(all_targets).squeeze()
        y_prob = np.vstack(all_probs).squeeze()
        
        # Metrics
        loss_avg = total_loss / batches
        accuracy = float(accuracy_score(y_true, (y_prob >= 0.5).astype(float)))
        roc_auc = float(roc_auc_score(y_true, y_prob))
        brier = float(brier_score_loss(y_true, y_prob))
        
        return {
            "loss": loss_avg,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "brier_score": brier
        }

    def fit(self, num_epochs: int, start_epoch: int = 1) -> dict:
        print(f"Beginning training loop from epoch {start_epoch} to {num_epochs}...")
        
        best_metrics = {}
        
        for epoch in range(start_epoch, num_epochs + 1):
            t0 = time.time()
            
            # Train and validate
            train_loss, grad_norm = self.train_epoch()
            val_metrics = self.val_epoch()
            
            epoch_time = time.time() - t0
            
            # Fetch current learning rate
            curr_lr = self.optimizer.param_groups[0]["lr"]
            
            # Log metrics
            epoch_metrics = {
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_roc_auc": val_metrics["roc_auc"],
                "val_brier_score": val_metrics["brier_score"],
                "learning_rate": curr_lr,
                "grad_norm": grad_norm,
                "epoch_time_seconds": epoch_time
            }
            self.logger.log_epoch(epoch, epoch_metrics)
            
            # Scheduler Step
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
                    
            print(f"  Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | LR: {curr_lr:.6f} | Time: {epoch_time:.1f}s")
            
            # Checkpoint: Save best validation
            val_loss = val_metrics["loss"]
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_metrics = epoch_metrics
                best_metrics["best_epoch"] = epoch
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    config=self.config,
                    seed=self.seed,
                    name="best_validation.pt"
                )
                print("    --> Saved new best checkpoint (best_validation.pt).")
                
            # Checkpoint: Save latest
            if self.config.get("checkpoint", {}).get("save_every_epoch", True):
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=val_metrics,
                    config=self.config,
                    seed=self.seed,
                    name="latest.pt"
                )
                
            # Early stopping check
            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch} due to no validation loss improvement.")
                break
                
        return best_metrics
