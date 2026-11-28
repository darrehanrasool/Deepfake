"""Configuration parameters and paths for university training."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class TrainingConfig:
    """Hyperparameter and runtime configuration for DeepFake-AI training."""
    
    # Training Hyperparameters
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    max_grad_norm: float = 1.0
    
    # Dataset Parameters
    num_frames: int = 8
    img_size: int = 224
    val_split_ratio: float = 0.20
    random_seed: int = 42
    
    # Device & Precision
    device: str = 'auto'  # 'auto', 'cuda', 'mps', or 'cpu'
    use_amp: bool = True
    
    # Checkpointing & Outputs
    output_dir: Path = Path('models/weights')
    checkpoint_name: str = 'latest_checkpoint.pt'
    best_model_name: str = 'best_model.pt'
    metrics_csv_name: str = 'training_metrics.csv'
    resume: bool = True
    
    # Visualization
    viz_dir: Path = Path('visualizations')
    plot_filename: str = 'real_multimodal_training_curves.png'

    def __post_init__(self):
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.viz_dir, str):
            self.viz_dir = Path(self.viz_dir)

    # Candidate Dataset Search Directories
    dfd_search_roots: List[Path] = field(default_factory=lambda: [
        Path('data/raw/dfd_benchmark'),
        Path('../data/raw/dfd_benchmark'),
        Path('/kaggle/input/google-dfd-deepfake-detection-dataset'),
        Path('/content/drive/MyDrive/dfd_benchmark'),
    ])
    
    dfdc_search_roots: List[Path] = field(default_factory=lambda: [
        Path('data/raw/dfdc'),
        Path('../data/raw/dfdc'),
        Path('/kaggle/input/deepfake-detection-challenge/train_sample_videos'),
        Path('/kaggle/input/dfdc-sample-video'),
        Path('/content/drive/MyDrive/dfdc'),
    ])

    def setup_directories(self):
        """Ensure all required output and visualization directories exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)

    @property
    def latest_checkpoint_path(self) -> Path:
        return self.output_dir / self.checkpoint_name

    @property
    def best_checkpoint_path(self) -> Path:
        return self.output_dir / self.best_model_name

    @property
    def metrics_log_path(self) -> Path:
        return self.output_dir / self.metrics_csv_name

    @property
    def plot_save_path(self) -> Path:
        return self.viz_dir / self.plot_filename
