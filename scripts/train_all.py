#!/usr/bin/env python3
"""Train all spatial, frequency, and spatio-temporal deepfake models."""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from deepfake_ai.models.model_factory import ModelFactory
from deepfake_ai.training.trainer import Trainer
from deepfake_ai.data.dataset_adapter import DeepfakeDatasetAdapter


def main():
    print('=== Initializing DeepFake-AI Model Training Suite ===')
    base = Path(__file__).resolve().parent.parent / 'data'
    train_csv = base / 'splits/train/train_samples.csv'
    val_csv = base / 'splits/validation/val_samples.csv'

    if not train_csv.exists() or not val_csv.exists():
        print('Splits not found. Please run scripts/prepare_data.py first.')
        return

    train_ds = DeepfakeDatasetAdapter(train_csv)
    val_ds = DeepfakeDatasetAdapter(val_csv)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    models_to_train = ['mesoinception4', 'xception', 'dual_stream_frequency']

    for name in models_to_train:
        print(f"\n--> Training Model: {name}")
        model = ModelFactory.create(name)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
        criterion = nn.CrossEntropyLoss()

        trainer = Trainer(model, optimizer, criterion, model_name=name)
        for epoch in range(1, 3): # Fast demo epoch run
            loss = trainer.train_epoch(train_loader)
            metrics = trainer.evaluate(val_loader)
            metrics['train_loss'] = loss
            trainer.callbacks.on_epoch_end(epoch, metrics)
        trainer.ckpt_manager.save_checkpoint(model, optimizer, epoch=2, metrics=metrics, is_best=True, model_name=name)

    print("\nAll models trained and saved to checkpoints/!")


if __name__ == '__main__':
    main()
