"""Atomic checkpoint persistence, metrics logging, and interruption recovery."""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import torch
import torch.nn as nn
from university.config import TrainingConfig


class CheckpointManager:
    """Manages atomic saving, loading, and recovery of training states."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.config.setup_directories()

    def has_checkpoint(self) -> bool:
        """Returns True if a prior checkpoint exists."""
        return self.config.latest_checkpoint_path.exists() or self.config.best_checkpoint_path.exists()

    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Optional[torch.cuda.amp.GradScaler],
        best_val_auc: float,
        history: List[Dict[str, Any]],
        is_best: bool = False
    ):
        """Atomically saves the current training session state."""
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_auc': best_val_auc,
            'history': history
        }
        if scaler is not None and hasattr(scaler, 'state_dict'):
            state['scaler_state_dict'] = scaler.state_dict()

        # Save latest checkpoint
        torch.save(state, self.config.latest_checkpoint_path)

        # Save best model
        if is_best:
            torch.save(model.state_dict(), self.config.best_checkpoint_path)
            print(f"  >>> [★] NEW HIGHEST VALIDATION ROC-AUC ({best_val_auc:.4f})! Saved weights to {self.config.best_checkpoint_path} <<<")

        # Save metrics history to CSV
        if history:
            pd.DataFrame(history).to_csv(self.config.metrics_log_path, index=False)

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: Optional[torch.cuda.amp.GradScaler],
        device: torch.device
    ) -> Tuple[int, float, List[Dict[str, Any]]]:
        """Loads prior checkpoint for seamless automatic resumption.
        
        Returns:
            (start_epoch, best_val_auc, history)
        """
        start_epoch = 1
        best_val_auc = 0.0
        history: List[Dict[str, Any]] = []

        if self.config.resume and self.config.latest_checkpoint_path.exists():
            print(f"[*] DETECTED PRIOR CHECKPOINT at {self.config.latest_checkpoint_path}! Resuming session...")
            try:
                chk = torch.load(self.config.latest_checkpoint_path, map_location=device)
                model.load_state_dict(chk['model_state_dict'])
                optimizer.load_state_dict(chk['optimizer_state_dict'])
                scheduler.load_state_dict(chk['scheduler_state_dict'])
                if scaler is not None and 'scaler_state_dict' in chk:
                    scaler.load_state_dict(chk['scaler_state_dict'])
                start_epoch = chk.get('epoch', 0) + 1
                best_val_auc = chk.get('best_val_auc', 0.0)
                history = chk.get('history', [])
                print(f"[✓] AUTOMATIC RESUME SUCCESSFUL: Picking up from Epoch {start_epoch:02d}/{self.config.epochs:02d}!")
                print(f"    • Baseline Best Validation ROC-AUC so far: {best_val_auc:.4f}")
            except Exception as e:
                print(f"[-] Note on checkpoint loading: {e}. Starting fresh.")

        elif self.config.best_checkpoint_path.exists():
            print(f"[*] Discovered pre-trained weights at {self.config.best_checkpoint_path}. Loading parameters...")
            try:
                chk = torch.load(self.config.best_checkpoint_path, map_location=device)
                if isinstance(chk, dict) and 'model_state_dict' in chk:
                    model.load_state_dict(chk['model_state_dict'])
                    best_val_auc = chk.get('val_auc', 0.0)
                else:
                    model.load_state_dict(chk)
                print("[✓] Loaded pre-trained model weights cleanly!")
            except Exception as e:
                print(f"[-] Note: Fresh initialization ({e})")

        return start_epoch, best_val_auc, history
