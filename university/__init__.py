"""DeepFake-AI: University Multimodal Forensic Training Package.

Modular, production-grade engineering package for university workstations and clusters.
Decouples hardware detection, dual-benchmark ingestion, ST2-SyncNet modeling,
SOTA regularized training, and publication visualization.

Institutional Authors:
- Dar Rehan Rasool (M.Tech Student, Centre for Artificial Intelligence, IUST)
- Dr. Riyaz Ahmed (M.Tech Student, Centre for Artificial Intelligence, IUST)
- Dr. M. Abdul Jawad (M.Tech Student, Centre for Artificial Intelligence, IUST)

Copyright (c) 2026 Dar Rehan Rasool (@darrehanrasool). All rights reserved.
"""

from university.config import TrainingConfig
from university.hardware import detect_hardware, print_hardware_banner
from university.dataset import MultimodalDeepfakeDataset, discover_datasets
from university.model import ST2SyncNet
from university.trainer import ModelTrainer
from university.checkpoints import CheckpointManager

__all__ = [
    'TrainingConfig',
    'detect_hardware',
    'print_hardware_banner',
    'MultimodalDeepfakeDataset',
    'discover_datasets',
    'ST2SyncNet',
    'ModelTrainer',
    'CheckpointManager'
]
