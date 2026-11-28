"""Publication-grade performance curves and ROC visualization."""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_training_curves(
    history: List[Dict[str, Any]],
    val_targets: np.ndarray,
    val_preds: np.ndarray,
    save_path: Path
):
    """Generates and saves publication-quality 3-panel performance visualizations."""
    if not history:
        print("[-] No history available to generate training curves.")
        return

    hist_df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Cross-Entropy Loss Progression
    axes[0].plot(hist_df['epoch'], hist_df['train_loss'], label='Train Loss', color='#1f77b4', lw=2, marker='o')
    axes[0].plot(hist_df['epoch'], hist_df['val_loss'], label='Val Loss', color='#d62728', lw=2, marker='s')
    axes[0].set_title('Cross-Entropy Loss Progression', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # 2. Discrimination Trajectory (ROC-AUC & Accuracy)
    axes[1].plot(hist_df['epoch'], hist_df['val_auc'], label='Val ROC-AUC', color='#2ca02c', lw=2, marker='^')
    axes[1].plot(hist_df['epoch'], hist_df['val_acc'], label='Val Accuracy', color='#9467bd', lw=2, marker='d')
    axes[1].set_title('Validation Discrimination Trajectory', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    # 3. Final Validation ROC Curve
    if len(val_targets) > 0 and len(np.unique(val_targets)) > 1:
        fpr, tpr, _ = roc_curve(val_targets, val_preds)
        roc_score = auc(fpr, tpr)
        axes[2].plot(fpr, tpr, color='#ff7f0e', lw=2.5, label=f'ST2-SyncNet (AUC = {roc_score:.4f})')
        axes[2].plot([0, 1], [0, 1], color='navy', linestyle='--', alpha=0.7)
        axes[2].set_title('Validation ROC Curve', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('False Positive Rate')
        axes[2].set_ylabel('True Positive Rate')
        axes[2].grid(True, linestyle='--', alpha=0.6)
        axes[2].legend()

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] Publication-quality training curves saved to: {save_path}")
