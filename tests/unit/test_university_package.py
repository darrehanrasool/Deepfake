"""Unit tests for the university modular package."""

import torch
import pytest
from pathlib import Path

from university.config import TrainingConfig
from university.hardware import detect_hardware
from university.model import ST2SyncNet
from university.checkpoints import CheckpointManager


def test_university_config(tmp_path):
    output_dir = tmp_path / "test_weights"
    viz_dir = tmp_path / "test_viz"
    config = TrainingConfig(
        epochs=5,
        batch_size=4,
        learning_rate=1e-4,
        output_dir=output_dir,
        viz_dir=viz_dir
    )
    assert config.epochs == 5
    assert config.batch_size == 4
    assert config.learning_rate == 1e-4
    config.setup_directories()
    assert config.output_dir.exists()
    assert config.viz_dir.exists()


def test_university_hardware():
    device, use_amp, telemetry = detect_hardware()
    assert isinstance(device, torch.device)
    assert isinstance(use_amp, bool)
    assert 'ram_total_gb' in telemetry
    assert 'ram_available_gb' in telemetry
    assert telemetry['cpu_cores'] > 0


def test_university_st2_syncnet():
    model = ST2SyncNet()
    batch_size = 2
    # Spatial: (B, 3, 224, 224)
    spatial = torch.randn(batch_size, 3, 224, 224)
    # Temporal: (B, 8, 3, 224, 224)
    temporal = torch.randn(batch_size, 8, 3, 224, 224)

    logits = model(spatial, temporal)
    assert logits.shape == (batch_size, 2)
    assert not torch.isnan(logits).any()


def test_university_checkpoints(tmp_path):
    config = TrainingConfig(
        output_dir=tmp_path / "checkpoints",
        viz_dir=tmp_path / "viz"
    )
    manager = CheckpointManager(config)
    assert not manager.has_checkpoint()

    model = ST2SyncNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    manager.save_checkpoint(
        epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        best_val_auc=0.85,
        history=[{'epoch': 1, 'val_auc': 0.85}],
        is_best=True
    )

    assert manager.has_checkpoint()
    assert config.best_checkpoint_path.exists()
    assert config.latest_checkpoint_path.exists()

    new_model = ST2SyncNet()
    new_opt = torch.optim.AdamW(new_model.parameters(), lr=1e-4)
    new_sched = torch.optim.lr_scheduler.CosineAnnealingLR(new_opt, T_max=5)

    start_epoch, best_auc, history = manager.load_checkpoint(
        model=new_model,
        optimizer=new_opt,
        scheduler=new_sched,
        scaler=None,
        device=torch.device('cpu')
    )
    assert start_epoch == 2
    assert best_auc == 0.85
    assert len(history) == 1
