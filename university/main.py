"""Master CLI entrypoint for the university training pipeline."""

import argparse
import random
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path so running directly works from any directory
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import numpy as np
import torch
from torch.utils.data import DataLoader

from university.config import TrainingConfig
from university.hardware import detect_hardware, print_hardware_banner
from university.dataset import MultimodalDeepfakeDataset, discover_datasets
from university.model import ST2SyncNet
from university.trainer import ModelTrainer
from university.visualization import plot_training_curves
from university.verify import verify_video_authenticity


def seed_everything(seed: int = 42):
    """Seed all pseudo-random number generators."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def parse_arguments() -> TrainingConfig:
    """Parses command-line arguments into a TrainingConfig instance."""
    parser = argparse.ArgumentParser(
        description='DeepFake-AI: University Multimodal Forensic Training Pipeline'
    )
    parser.add_argument('--epochs', type=int, default=10, help='Total training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='AdamW weight decay (default: 1e-4)')
    parser.add_argument('--label-smoothing', type=float, default=0.05, help='Label smoothing factor (default: 0.05)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Gradient clipping threshold (default: 1.0)')
    parser.add_argument('--num-frames', type=int, default=8, help='Frames per video tube (default: 8)')
    parser.add_argument('--output-dir', type=str, default='models/weights', help='Checkpoint directory (default: models/weights)')
    parser.add_argument('--no-resume', action='store_true', help='Disable automatic resume from checkpoint')
    parser.add_argument('--device', type=str, default='auto', help='Target device: auto, cuda, mps, or cpu')

    args = parser.parse_args()

    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        max_grad_norm=args.max_grad_norm,
        num_frames=args.num_frames,
        output_dir=args.output_dir,
        resume=not args.no_resume,
        device=args.device
    )
    return config


def main():
    """Main execution orchestrator."""
    config = parse_arguments()
    seed_everything(config.random_seed)

    # 1. Hardware Detection & Telemetry
    device, use_amp, telemetry = detect_hardware(config.device)
    config.use_amp = use_amp
    print_hardware_banner(telemetry)

    # 2. Dual Dataset Discovery & Partitioning
    train_df, val_df, class_weights = discover_datasets(config)

    # 3. Model Architecture
    model = ST2SyncNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[✓] ST2-SyncNet Multimodal Architecture Instantiated:")
    print(f"  • Total Parameters     : {total_params:,}")
    print(f"  • Trainable Parameters : {trainable_params:,}")

    # 4. DataLoaders
    batch_size = config.batch_size if device.type != 'cpu' else max(2, config.batch_size // 2)
    train_dataset = MultimodalDeepfakeDataset(train_df, num_frames=config.num_frames, is_train=True)
    val_dataset = MultimodalDeepfakeDataset(val_df, num_frames=config.num_frames, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # 5. Trainer Engine
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        config=config,
        device=device,
        use_amp=use_amp
    )

    # 6. Run Training Campaign
    history, val_targets, val_preds = trainer.run()

    # 7. Generate Publication Figures
    plot_training_curves(history, val_targets, val_preds, config.plot_save_path)

    # 8. Forensic Verification Demonstration
    if len(val_df) > 0 and config.best_checkpoint_path.exists():
        demo_video = val_df.iloc[0]['video_path']
        verify_video_authenticity(
            video_path=demo_video,
            model_weights_path=config.best_checkpoint_path,
            device=device,
            num_frames=config.num_frames
        )


if __name__ == '__main__':
    main()
