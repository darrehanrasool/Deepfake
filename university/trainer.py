"""Production training and validation engine with real-time ASCII telemetry."""

import sys
import time
import gc
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
import psutil

from university.config import TrainingConfig
from university.checkpoints import CheckpointManager


class ModelTrainer:
    """Orchestrates regularized training, validation, and real-time terminal reporting."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor,
        config: TrainingConfig,
        device: torch.device,
        use_amp: bool = True
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_weights = class_weights.to(device)
        self.config = config
        self.device = device
        self.use_amp = use_amp

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=config.label_smoothing
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-6
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.checkpoint_manager = CheckpointManager(config)

    def train_epoch(self, epoch: int) -> float:
        """Executes a single training epoch with live console feedback."""
        self.model.train()
        running_loss = 0.0
        n_batches = len(self.train_loader)
        current_lr = self.optimizer.param_groups[0]['lr']

        print(f"\n=====================================================================================")
        print(f"[*] >>> EPOCH {epoch:02d}/{self.config.epochs:02d} IN PROGRESS [Learning Rate: {current_lr:.2e}] <<<")
        print(f"=====================================================================================")
        sys.stdout.flush()

        t_step = time.time()
        for b_idx, batch in enumerate(self.train_loader, 1):
            spatial = batch['spatial'].to(self.device, non_blocking=True)
            temporal = batch['temporal'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(spatial, temporal)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(spatial, temporal)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
                self.optimizer.step()

            running_loss += loss.item()
            step_time = time.time() - t_step
            t_step = time.time()

            # Real-time console progress
            pct = (b_idx / n_batches) * 100
            bar_len = 20
            filled = int(bar_len * b_idx / n_batches)
            bar_str = '█' * filled + '░' * (bar_len - filled)
            avg_loss = running_loss / b_idx
            remaining_steps = n_batches - b_idx
            eta_sec = int(remaining_steps * step_time)
            eta_str = f"{eta_sec // 60:02d}m {eta_sec % 60:02d}s"

            mem_telemetry = ''
            if torch.cuda.is_available():
                v_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
                mem_telemetry = f" | VRAM: {v_alloc:.0f}MB"
            try:
                mem_telemetry += f" | RAM: {psutil.virtual_memory().percent:.0f}%"
            except Exception:
                pass

            print(f"  [TRAIN] Epoch {epoch:02d}/{self.config.epochs:02d} | Batch {b_idx:02d}/{n_batches:02d} [{bar_str}] {pct:5.1f}% | Loss: {loss.item():.4f} (Avg: {avg_loss:.4f}){mem_telemetry} | Step: {step_time:.1f}s | ETA: {eta_str}")
            sys.stdout.flush()

            del spatial, temporal, labels, logits, loss
            if b_idx % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.scheduler.step()
        return running_loss / max(n_batches, 1)

    def validate(self) -> Tuple[float, float, float, float, float, float, np.ndarray, np.ndarray]:
        """Evaluates model discrimination on unseen validation videos."""
        self.model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        n_batches = len(self.val_loader)

        print(f"\n  [VALIDATION] Commencing validation on {len(self.val_loader.dataset)} unseen test videos...")
        sys.stdout.flush()

        with torch.no_grad():
            for vb_idx, batch in enumerate(self.val_loader, 1):
                spatial = batch['spatial'].to(self.device, non_blocking=True)
                temporal = batch['temporal'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model(spatial, temporal)
                        v_loss = self.criterion(logits, labels)
                else:
                    logits = self.model(spatial, temporal)
                    v_loss = self.criterion(logits, labels)

                val_loss += v_loss.item()
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                val_preds.extend(probs.tolist())
                val_targets.extend(labels.cpu().numpy().tolist())

                pct_v = (vb_idx / n_batches) * 100
                bar_len = 20
                filled_v = int(bar_len * vb_idx / n_batches)
                bar_v = '█' * filled_v + '░' * (bar_len - filled_v)
                print(f"  [VAL]   Batch {vb_idx:02d}/{n_batches:02d} [{bar_v}] {pct_v:5.1f}% | Val Loss: {v_loss.item():.4f}")
                sys.stdout.flush()

                del spatial, temporal, labels, logits, v_loss

        val_loss /= max(n_batches, 1)
        val_preds_arr = np.array(val_preds)
        val_targets_arr = np.array(val_targets)

        binary_preds = (val_preds_arr >= 0.5).astype(int)
        acc = accuracy_score(val_targets_arr, binary_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(val_targets_arr, binary_preds, average='binary', zero_division=0)
        try:
            val_auc = roc_auc_score(val_targets_arr, val_preds_arr)
        except Exception:
            val_auc = 0.5

        return val_loss, val_auc, acc, precision, recall, f1, val_targets_arr, val_preds_arr

    def run(self) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        """Runs the complete training campaign with automatic resumption and interrupt handling."""
        start_epoch, best_val_auc, history = self.checkpoint_manager.load_checkpoint(
            self.model, self.optimizer, self.scheduler, self.scaler, self.device
        )

        last_targets = np.array([])
        last_preds = np.array([])

        print('=' * 85)
        print('      LAUNCHING MASTER MULTIMODAL ST2-SYNCNET SOTA TRAINING PIPELINE')
        print('=' * 85)

        try:
            for epoch in range(start_epoch, self.config.epochs + 1):
                t_start = time.time()
                train_loss = self.train_epoch(epoch)
                val_loss, val_auc, acc, prec, rec, f1, last_targets, last_preds = self.validate()
                epoch_time = time.time() - t_start

                rec_dict = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_auc': val_auc,
                    'val_acc': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'lr': self.optimizer.param_groups[0]['lr'],
                    'epoch_sec': epoch_time
                }
                history.append(rec_dict)

                print(f"\n" + "-" * 85)
                print(f"[✓] EPOCH {epoch:02d}/{self.config.epochs:02d} SUMMARY ({epoch_time:.1f}s)")
                print(f"  • Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"  • Val ROC-AUC: {val_auc:.4f} | Accuracy: {acc * 100:.2f}% | F1-Score: {f1:.4f}")
                print(f"  • Precision : {prec:.4f} | Recall: {rec:.4f}")
                print("-" * 85)

                is_best = val_auc > best_val_auc
                if is_best:
                    best_val_auc = val_auc

                self.checkpoint_manager.save_checkpoint(
                    epoch, self.model, self.optimizer, self.scheduler,
                    self.scaler, best_val_auc, history, is_best=is_best
                )

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            print('\n' + '=' * 85)
            print(f"     TRAINING CAMPAIGN CONCLUDED CLEANLY! PEAK VAL ROC-AUC: {best_val_auc:.4f}")
            print('=' * 85)

        except KeyboardInterrupt:
            print("\n[!] Session interrupted by user (KeyboardInterrupt)!")
            self.checkpoint_manager.save_checkpoint(
                epoch, self.model, self.optimizer, self.scheduler,
                self.scaler, best_val_auc, history, is_best=False
            )
            print(f"[✓] Clean recovery checkpoint saved to {self.config.latest_checkpoint_path}.")
            print("    You can resume training anytime simply by restarting the script!")

        return history, last_targets, last_preds
