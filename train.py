#!/usr/bin/env python3
"""DeepFake-AI: Master Multimodal Deepfake Training Pipeline (Standalone CLI).

Dual-Benchmark SOTA Forensic Training across Google/Jigsaw DFD & Meta DFDC.
Zero external Jupyter dependencies - runs directly via standard Python CLI.

Institutional Authors:
- Dar Rehan Rasool (M.Tech Student, Centre for Artificial Intelligence, IUST)
- Dr. Riyaz Ahmed (M.Tech Student, Centre for Artificial Intelligence, IUST)
- Dr. M. Abdul Jawad (M.Tech Student, Centre for Artificial Intelligence, IUST)

Affiliation: Centre for Artificial Intelligence, Islamic University of Science and Technology (IUST)
Copyright (c) 2026 Dar Rehan Rasool (@darrehanrasool). All rights reserved.
"""

import os
import sys
import time
import gc
import json
import math
import random
import shutil
import argparse
import subprocess
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Self-healing dependency verification for fresh university workstations
REQUIRED_MODULES = [
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('torch', 'torch'),
    ('cv2', 'opencv-python'),
    ('sklearn', 'scikit-learn'),
    ('matplotlib', 'matplotlib'),
    ('psutil', 'psutil')
]

if '-h' not in sys.argv and '--help' not in sys.argv:
    for mod_name, pip_name in REQUIRED_MODULES:
        try:
            __import__(mod_name)
        except ImportError:
            print(f'[*] Installing missing dependency: {pip_name}...')
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pip_name])
            except Exception as e:
                print(f'[-] Note on {pip_name} installation: {e}')

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, roc_curve, auc
import matplotlib.pyplot as plt
import psutil


def seed_everything(seed=42):
    """Seed all pseudo-random number generators for strict reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# =============================================================================
# 1. ARCHITECTURE DEFINITION: ST2-SYNCNET
# =============================================================================

class WaveletDecomposition2D(nn.Module):
    """2D Haar Discrete Wavelet Transform (DWT) producing 12 sub-bands (4 per RGB channel)."""
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('kernel', kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_reshaped = x.view(b * c, 1, h, w)
        subbands = F.conv2d(x_reshaped, self.kernel, stride=2)
        _, _, sh, sw = subbands.shape
        return subbands.view(b, c * 4, sh, sw)


class SpectralPhaseCoherenceModule(nn.Module):
    """Exposes generator high-frequency upsampling grid residuals."""
    def __init__(self, in_channels: int = 12, out_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Linear(128 * 4 * 4, out_dim)

    def forward(self, subbands: torch.Tensor) -> torch.Tensor:
        feat = self.conv(subbands)
        return self.fc(torch.flatten(feat, 1))


class SpatialBoundaryAttentionModule(nn.Module):
    """Multi-scale convolutional backbone isolating face-swap boundary blending seams."""
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Linear(128 * 4 * 4, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = F.relu(x + self.res_block(x))
        x = self.head(x)
        return self.fc(torch.flatten(x, 1))


class TemporalBiologicalConsistencyModule(nn.Module):
    """Bi-directional GRU tracking optical flow and biological blinking continuity over frames."""
    def __init__(self, feature_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, feature_dim)

    def forward(self, seq_feats: torch.Tensor) -> torch.Tensor:
        if hasattr(self.gru, 'flatten_parameters'):
            self.gru.flatten_parameters()
        gru_out, _ = self.gru(seq_feats)
        return self.fc(gru_out[:, -1, :])


class ST2SyncNet(nn.Module):
    """Full ST2-SyncNet Multimodal Deepfake Architecture with Cross-Attention Fusion."""
    def __init__(self, num_classes: int = 2, feature_dim: int = 128):
        super().__init__()
        self.spatial_encoder = SpatialBoundaryAttentionModule(out_dim=feature_dim)
        self.wavelet_transform = WaveletDecomposition2D()
        self.spectral_encoder = SpectralPhaseCoherenceModule(in_channels=12, out_dim=feature_dim)
        self.temporal_encoder = TemporalBiologicalConsistencyModule(feature_dim=feature_dim)

        self.cross_attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=4)
        self.norm = nn.LayerNorm(feature_dim)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_spatial: torch.Tensor, x_temporal: torch.Tensor):
        f_spatial = self.spatial_encoder(x_spatial)
        subbands = self.wavelet_transform(x_spatial)
        f_spectral = self.spectral_encoder(subbands)

        q = f_spatial.unsqueeze(0)
        kv = torch.stack([f_spatial, f_spectral], dim=0)
        attn_out, _ = self.cross_attention(q, kv, kv)
        f_fused = self.norm(f_spatial + attn_out.squeeze(0))

        b, s, c, h, w = x_temporal.shape
        x_temp_flat = x_temporal.view(b * s, c, h, w)
        temp_spatial = self.spatial_encoder(x_temp_flat).view(b, s, -1)
        f_temporal = self.temporal_encoder(temp_spatial)

        f_combined = torch.cat([f_fused, f_temporal], dim=1)
        logits = self.classifier(f_combined)
        return logits


# =============================================================================
# 2. DATASET & DATALOADER
# =============================================================================

class MultimodalDeepfakeDataset(Dataset):
    """Production Multimodal Video Dataset with on-the-fly OpenCV frame tube decoding."""
    def __init__(self, df: pd.DataFrame, num_frames: int = 8, img_size: int = 224, is_train: bool = True):
        self.df = df
        self.num_frames = num_frames
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def _extract_frames(self, video_path: str) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        if total_frames <= 0 or not cap.isOpened():
            cap.release()
            return np.zeros((self.num_frames, self.img_size, self.img_size, 3), dtype=np.uint8)

        if total_frames < self.num_frames:
            indices = list(range(total_frames)) + [total_frames - 1] * (self.num_frames - total_frames)
        else:
            if self.is_train:
                step = total_frames / self.num_frames
                indices = [int(i * step + random.uniform(0, max(0.1, step - 1))) for i in range(self.num_frames)]
                indices = [min(idx, total_frames - 1) for idx in indices]
            else:
                indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int).tolist()

        current_frame = 0
        idx_set = set(indices)
        collected = {}

        while cap.isOpened() and len(collected) < len(idx_set):
            ret, frame = cap.read()
            if not ret:
                break
            if current_frame in idx_set:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, _ = frame_rgb.shape
                min_dim = min(h, w)
                sy = (h - min_dim) // 2
                sx = (w - min_dim) // 2
                crop = frame_rgb[sy:sy + min_dim, sx:sx + min_dim]
                resized = cv2.resize(crop, (self.img_size, self.img_size))
                collected[current_frame] = resized
            current_frame += 1

        cap.release()

        fallback = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        for idx in indices:
            frames.append(collected.get(idx, fallback))
        return np.array(frames, dtype=np.uint8)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        raw_frames = self._extract_frames(row['video_path'])
        norm_frames = (raw_frames.astype(np.float32) / 127.5) - 1.0

        if self.is_train:
            if random.random() > 0.5:
                norm_frames = np.flip(norm_frames, axis=2).copy()
            if random.random() > 0.5:
                brightness = random.uniform(0.90, 1.10)
                norm_frames = np.clip(norm_frames * brightness, -1.0, 1.0)

        temporal_tube = torch.from_numpy(norm_frames).permute(0, 3, 1, 2)
        spatial_frame = temporal_tube[self.num_frames // 2]

        return {
            'spatial': spatial_frame,
            'temporal': temporal_tube,
            'label': torch.tensor(row['label'], dtype=torch.long),
            'dataset': row['dataset'],
            'path': row['video_path']
        }


# =============================================================================
# 3. DUAL DATASET DISCOVERY & LEAK-FREE SPLITTING
# =============================================================================

def discover_datasets():
    """Auto-discovers Google/Jigsaw DFD and Meta DFDC datasets across directories."""
    print('\n[*] Scanning environment for Google/Jigsaw DFD and Meta DFDC video corpora...')

    dfd_search_roots = [
        Path('data/raw/dfd_benchmark'),
        Path('../data/raw/dfd_benchmark'),
        Path('/kaggle/input/google-dfd-deepfake-detection-dataset'),
        Path('/content/drive/MyDrive/dfd_benchmark'),
    ]

    dfdc_search_roots = [
        Path('data/raw/dfdc'),
        Path('../data/raw/dfdc'),
        Path('/kaggle/input/deepfake-detection-challenge/train_sample_videos'),
        Path('/kaggle/input/dfdc-sample-video'),
        Path('/content/drive/MyDrive/dfdc'),
    ]

    records = []

    # 1. Ingest Google & Jigsaw DFD Benchmark
    dfd_found = None
    for cand in dfd_search_roots:
        if cand.exists() and (cand / 'DFD_original sequences').exists():
            dfd_found = cand
            break

    if dfd_found:
        print(f'  [+] Discovered Google/Jigsaw DFD at: {dfd_found}')
        dfd_real = list((dfd_found / 'DFD_original sequences').glob('*.mp4'))
        for p in dfd_real:
            actor_id = p.stem.split('__')[0] if '__' in p.stem else 'dfd_orig'
            records.append({
                'video_path': str(p.resolve()),
                'label': 0,
                'label_name': 'REAL',
                'dataset': 'Google_DFD',
                'actor_group': f'dfd_actor_{actor_id}',
                'filesize_mb': p.stat().st_size / (1024 * 1024)
            })
        dfd_fake = list((dfd_found / 'DFD_manipulated_sequences').glob('*.mp4'))
        for p in dfd_fake:
            actor_id = p.stem.split('__')[0] if '__' in p.stem else 'dfd_synth'
            records.append({
                'video_path': str(p.resolve()),
                'label': 1,
                'label_name': 'FAKE',
                'dataset': 'Google_DFD',
                'actor_group': f'dfd_actor_{actor_id}',
                'filesize_mb': p.stat().st_size / (1024 * 1024)
            })
        print(f'      • Google DFD Ingested: {len(dfd_real)} REAL | {len(dfd_fake)} FAKE (Total: {len(dfd_real) + len(dfd_fake)})')
    else:
        print('  [-] Google/Jigsaw DFD not found in standard directories. Skipping DFD stream.')

    # 2. Ingest Meta DFDC (DeepFake Detection Challenge)
    dfdc_found = None
    for cand in dfdc_search_roots:
        if cand.exists() and (list(cand.glob('*.mp4')) or (cand / 'metadata.json').exists()):
            dfdc_found = cand
            break

    if dfdc_found:
        print(f'  [+] Discovered Meta DFDC at: {dfdc_found}')
        meta_file = dfdc_found / 'metadata.json'
        if not meta_file.exists():
            meta_file = dfdc_found.parent / 'metadata.json'
        
        meta_dict = {}
        if meta_file.exists():
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta_dict = json.load(f)
        
        dfdc_videos = list(dfdc_found.glob('*.mp4'))
        dfdc_real_cnt, dfdc_fake_cnt = 0, 0
        for p in dfdc_videos:
            fname = p.name
            info = meta_dict.get(fname, {})
            lbl_str = info.get('label', 'FAKE' if 'fake' in str(p).lower() else 'REAL').upper()
            lbl = 0 if lbl_str == 'REAL' else 1
            orig = info.get('original', None)
            actor_group = f'dfdc_group_{orig if orig else fname}'
            
            if lbl == 0:
                dfdc_real_cnt += 1
            else:
                dfdc_fake_cnt += 1
                
            records.append({
                'video_path': str(p.resolve()),
                'label': lbl,
                'label_name': 'REAL' if lbl == 0 else 'FAKE',
                'dataset': 'Meta_DFDC',
                'actor_group': actor_group,
                'filesize_mb': p.stat().st_size / (1024 * 1024)
            })
        print(f'      • Meta DFDC Ingested: {dfdc_real_cnt} REAL | {dfdc_fake_cnt} FAKE (Total: {len(dfdc_videos)})')
    else:
        print('  [-] Meta DFDC not found in standard directories. Skipping DFDC stream.')

    corpus_df = pd.DataFrame(records)
    if len(corpus_df) == 0:
        raise RuntimeError('No video files found! Ensure either data/raw/dfd_benchmark or data/raw/dfdc is populated.')

    print('\n' + '=' * 85)
    print(f'      TOTAL UNIFIED VIDEO CORPUS: {len(corpus_df)} SAMPLES ACROSS BENCHMARKS')
    print('=' * 85)
    print(corpus_df.groupby(['dataset', 'label_name']).size().unstack(fill_value=0))

    # Strict Actor-Isolated Split to prevent data leakage
    groups = corpus_df['actor_group'].unique()
    np.random.seed(42)
    np.random.shuffle(groups)

    val_group_count = max(1, int(len(groups) * 0.20))
    val_groups = set(groups[:val_group_count])
    train_groups = set(groups[val_group_count:])

    train_df = corpus_df[corpus_df['actor_group'].isin(train_groups)].reset_index(drop=True)
    val_df = corpus_df[corpus_df['actor_group'].isin(val_groups)].reset_index(drop=True)

    print(f'\n[✓] Actor-Isolated Partitioning Completed:')
    print(f'  • Train Split : {len(train_df):5d} videos ({len(train_groups)} distinct actor clusters)')
    print(f'  • Val Split   : {len(val_df):5d} videos ({len(val_groups)} distinct actor clusters)')

    return train_df, val_df


# =============================================================================
# 4. MAIN TRAINING & INFERENCE PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='DeepFake-AI Master Multimodal Training Pipeline')
    parser.add_argument('--epochs', type=int, default=10, help='Total training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='AdamW weight decay (default: 1e-4)')
    parser.add_argument('--label-smoothing', type=float, default=0.05, help='Label smoothing factor (default: 0.05)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='Gradient clipping threshold (default: 1.0)')
    parser.add_argument('--num-frames', type=int, default=8, help='Frames per video tube (default: 8)')
    parser.add_argument('--output-dir', type=str, default='models/weights', help='Checkpoint output directory')
    parser.add_argument('--no-resume', action='store_true', help='Disable automatic resume from latest checkpoint')
    parser.add_argument('--device', type=str, default='auto', help='Target device: auto, cuda, mps, or cpu')
    args = parser.parse_args()

    seed_everything(42)

    # Telemetry & Hardware Selection
    print('=' * 85)
    print('             DEEPFAKE-AI: SYSTEM HARDWARE & COMPUTE TELEMETRY')
    print('=' * 85)
    print(f' • Operating System      : {sys.platform.upper()} ({os.name})')
    print(f' • Python Runtime        : {sys.version.split()[0]}')
    print(f' • PyTorch Version       : {torch.__version__}')
    print(f' • CPU Core Count        : {os.cpu_count()} logical cores')

    ram_info = psutil.virtual_memory()
    print(f' • Host System RAM       : {ram_info.total / (1024**3):.2f} GB Total | {ram_info.available / (1024**3):.2f} GB Available ({ram_info.percent}% used)')

    use_amp = False
    if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()):
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        cuda_ver = torch.version.cuda
        use_amp = True
        print(f' • Primary Accelerator   : NVIDIA CUDA GPU ({gpu_name})')
        print(f' • Total VRAM Allocated  : {vram_gb:.2f} GB')
        print(f' • CUDA Toolkit Version  : {cuda_ver}')
        print(' • Mixed Precision (AMP) : ENABLED (FP16 Accelerated)')
        try:
            smi_out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=temperature.gpu,fan.speed,utilization.gpu,power.draw', '--format=csv,noheader,nounits'],
                text=True, stderr=subprocess.DEVNULL
            ).strip().split(',')
            if len(smi_out) >= 4:
                temp, fan, util, power = [x.strip() for x in smi_out[:4]]
                print(f' • GPU Sensor Telemetry  : Temp: {temp}°C | Fan: {fan}% | Core Load: {util}% | Power: {power}W')
        except Exception:
            pass
    elif args.device == 'mps' or (args.device == 'auto' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        device = torch.device('mps')
        print(' • Primary Accelerator   : Apple Silicon Metal Performance Shaders (MPS)')
        print(' • Mixed Precision (AMP) : Standard FP32 on MPS')
    else:
        device = torch.device('cpu')
        print(' • Primary Accelerator   : High-Performance Multi-Threaded CPU')
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))
    print(f' • Active Compute Target : {device}')
    print('=' * 85)

    # Ingest Datasets
    train_df, val_df = discover_datasets()

    # Inverse Class Weight Calculation
    n_train = len(train_df)
    n_real = max(1, (train_df['label'] == 0).sum())
    n_fake = max(1, (train_df['label'] == 1).sum())
    weight_real = n_train / (2.0 * n_real)
    weight_fake = n_train / (2.0 * n_fake)
    class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float32).to(device)

    print(f'  • Class Balance: {n_real} REAL vs {n_fake} FAKE')
    print(f'  • Dynamic Loss Weights: REAL = {weight_real:.3f} | FAKE = {weight_fake:.3f}')

    # Model Instantiation
    model = ST2SyncNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\n[✓] ST2-SyncNet Multimodal Architecture Instantiated:')
    print(f'  • Total Parameters     : {total_params:,}')
    print(f'  • Trainable Parameters : {trainable_params:,}')

    # DataLoaders
    batch_size = args.batch_size if (device.type != 'cpu') else max(2, args.batch_size // 2)
    train_dataset = MultimodalDeepfakeDataset(train_df, num_frames=args.num_frames, is_train=True)
    val_dataset = MultimodalDeepfakeDataset(val_df, num_frames=args.num_frames, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / 'best_model.pt'
    latest_checkpoint_path = output_dir / 'latest_checkpoint.pt'
    metrics_log_path = output_dir / 'training_metrics.csv'

    start_epoch = 1
    best_val_auc = 0.0
    history = []

    # Automatic Resume from Interrupted Session
    if not args.no_resume and latest_checkpoint_path.exists():
        print(f'[*] DETECTED PRIOR CHECKPOINT at {latest_checkpoint_path}! Resuming session...')
        try:
            chk = torch.load(latest_checkpoint_path, map_location=device)
            model.load_state_dict(chk['model_state_dict'])
            optimizer.load_state_dict(chk['optimizer_state_dict'])
            scheduler.load_state_dict(chk['scheduler_state_dict'])
            if use_amp and 'scaler_state_dict' in chk:
                scaler.load_state_dict(chk['scaler_state_dict'])
            start_epoch = chk.get('epoch', 0) + 1
            best_val_auc = chk.get('best_val_auc', 0.0)
            history = chk.get('history', [])
            print(f'[✓] AUTOMATIC RESUME SUCCESSFUL: Picking up from Epoch {start_epoch:02d}/{args.epochs:02d}!')
            print(f'    • Baseline Best Validation ROC-AUC so far: {best_val_auc:.4f}')
        except Exception as e:
            print(f'[-] Note on checkpoint loading: {e}. Starting fresh.')
    elif best_checkpoint_path.exists():
        print(f'[*] Discovered pre-trained weights at {best_checkpoint_path}. Loading parameters...')
        try:
            chk = torch.load(best_checkpoint_path, map_location=device)
            if isinstance(chk, dict) and 'model_state_dict' in chk:
                model.load_state_dict(chk['model_state_dict'])
                best_val_auc = chk.get('val_auc', 0.0)
            else:
                model.load_state_dict(chk)
            print(f'[✓] Loaded pre-trained model weights cleanly!')
        except Exception as e:
            print(f'[-] Note: Fresh initialization ({e})')

    print('=' * 85)
    print('      LAUNCHING MASTER MULTIMODAL ST2-SYNCNET SOTA TRAINING PIPELINE')
    print('=' * 85)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            t_epoch_start = time.time()
            model.train()
            running_loss = 0.0
            n_train_batches = len(train_loader)
            current_lr = optimizer.param_groups[0]['lr']

            print(f'\n=====================================================================================')
            print(f'[*] >>> EPOCH {epoch:02d}/{args.epochs:02d} IN PROGRESS [Learning Rate: {current_lr:.2e}] <<<')
            print(f'=====================================================================================')
            sys.stdout.flush()

            t_step_start = time.time()
            for b_idx, batch in enumerate(train_loader, 1):
                spatial = batch['spatial'].to(device, non_blocking=True)
                temporal = batch['temporal'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)

                optimizer.zero_grad()
                if use_amp:
                    with torch.cuda.amp.autocast():
                        logits = model(spatial, temporal)
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(spatial, temporal)
                    loss = criterion(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                    optimizer.step()

                running_loss += loss.item()
                step_time = time.time() - t_step_start
                t_step_start = time.time()

                pct = (b_idx / n_train_batches) * 100
                bar_len = 20
                filled = int(bar_len * b_idx / n_train_batches)
                bar_str = '█' * filled + '░' * (bar_len - filled)
                avg_loss = running_loss / b_idx
                remaining_steps = n_train_batches - b_idx
                eta_sec = int(remaining_steps * step_time)
                eta_str = f'{eta_sec // 60:02d}m {eta_sec % 60:02d}s'

                mem_telemetry = ''
                if torch.cuda.is_available():
                    v_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
                    mem_telemetry = f' | VRAM: {v_alloc:.0f}MB'
                try:
                    mem_telemetry += f' | RAM: {psutil.virtual_memory().percent:.0f}%'
                except Exception:
                    pass

                print(f'  [TRAIN] Epoch {epoch:02d}/{args.epochs:02d} | Batch {b_idx:02d}/{n_train_batches:02d} [{bar_str}] {pct:5.1f}% | Loss: {loss.item():.4f} (Avg: {avg_loss:.4f}){mem_telemetry} | Step: {step_time:.1f}s | ETA: {eta_str}')
                sys.stdout.flush()

                del spatial, temporal, labels, logits, loss
                if b_idx % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            train_loss = running_loss / max(n_train_batches, 1)
            scheduler.step()

            # Validation Loop
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            n_val_batches = len(val_loader)

            print(f'\n  [VALIDATION] Commencing validation on {len(val_df)} unseen test videos...')
            sys.stdout.flush()

            with torch.no_grad():
                for vb_idx, batch in enumerate(val_loader, 1):
                    spatial = batch['spatial'].to(device, non_blocking=True)
                    temporal = batch['temporal'].to(device, non_blocking=True)
                    labels = batch['label'].to(device, non_blocking=True)

                    if use_amp:
                        with torch.cuda.amp.autocast():
                            logits = model(spatial, temporal)
                            v_loss = criterion(logits, labels)
                    else:
                        logits = model(spatial, temporal)
                        v_loss = criterion(logits, labels)

                    val_loss += v_loss.item()
                    probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    val_preds.extend(probs.tolist())
                    val_targets.extend(labels.cpu().numpy().tolist())

                    pct_v = (vb_idx / n_val_batches) * 100
                    filled_v = int(bar_len * vb_idx / n_val_batches)
                    bar_v = '█' * filled_v + '░' * (bar_len - filled_v)
                    print(f'  [VAL]   Batch {vb_idx:02d}/{n_val_batches:02d} [{bar_v}] {pct_v:5.1f}% | Val Loss: {v_loss.item():.4f}')
                    sys.stdout.flush()

                    del spatial, temporal, labels, logits, v_loss

            val_loss /= max(n_val_batches, 1)
            val_preds_arr = np.array(val_preds)
            val_targets_arr = np.array(val_targets)

            binary_preds = (val_preds_arr >= 0.5).astype(int)
            val_acc = accuracy_score(val_targets_arr, binary_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(val_targets_arr, binary_preds, average='binary', zero_division=0)
            try:
                val_auc = roc_auc_score(val_targets_arr, val_preds_arr)
            except Exception:
                val_auc = 0.5

            epoch_time = time.time() - t_epoch_start
            epoch_rec = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_auc': val_auc,
                'val_acc': val_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'lr': current_lr,
                'epoch_sec': epoch_time
            }
            history.append(epoch_rec)
            pd.DataFrame(history).to_csv(metrics_log_path, index=False)

            print(f'\n' + '-' * 85)
            print(f'[✓] EPOCH {epoch:02d}/{args.epochs:02d} SUMMARY ({epoch_time:.1f}s)')
            print(f'  • Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            print(f'  • Val ROC-AUC: {val_auc:.4f} | Accuracy: {val_acc * 100:.2f}% | F1-Score: {f1:.4f}')
            print(f'  • Precision : {precision:.4f} | Recall: {recall:.4f}')
            print('-' * 85)

            # Atomic Checkpoint Persistence
            latest_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_auc': max(best_val_auc, val_auc),
                'history': history
            }
            if use_amp:
                latest_state['scaler_state_dict'] = scaler.state_dict()
            torch.save(latest_state, latest_checkpoint_path)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), best_checkpoint_path)
                print(f'  >>> [★] NEW HIGHEST VALIDATION ROC-AUC ({best_val_auc:.4f})! Saved weights to {best_checkpoint_path} <<<')

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print('\n' + '=' * 85)
        print(f'     TRAINING CAMPAIGN CONCLUDED CLEANLY! PEAK VAL ROC-AUC: {best_val_auc:.4f}')
        print('=' * 85)

    except KeyboardInterrupt:
        print('\n[!] Session interrupted by user (KeyboardInterrupt)!')
        interrupted_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_auc': best_val_auc,
            'history': history
        }
        if use_amp:
            interrupted_state['scaler_state_dict'] = scaler.state_dict()
        torch.save(interrupted_state, latest_checkpoint_path)
        print(f'[✓] Clean recovery checkpoint saved to {latest_checkpoint_path}.')
        print('    You can rerun "python train.py" anytime and it will resume right here!')

    # Publication Figures Generation
    if len(history) > 0:
        hist_df = pd.DataFrame(history)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. Loss Progression
        axes[0].plot(hist_df['epoch'], hist_df['train_loss'], label='Train Loss', color='#1f77b4', lw=2, marker='o')
        axes[0].plot(hist_df['epoch'], hist_df['val_loss'], label='Val Loss', color='#d62728', lw=2, marker='s')
        axes[0].set_title('Cross-Entropy Loss Progression', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].legend()

        # 2. ROC-AUC and Accuracy Trajectory
        axes[1].plot(hist_df['epoch'], hist_df['val_auc'], label='Val ROC-AUC', color='#2ca02c', lw=2, marker='^')
        axes[1].plot(hist_df['epoch'], hist_df['val_acc'], label='Val Accuracy', color='#9467bd', lw=2, marker='d')
        axes[1].set_title('Validation Discrimination Trajectory', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].grid(True, linestyle='--', alpha=0.6)
        axes[1].legend()

        # 3. Final ROC Curve
        if len(val_targets_arr) > 0 and len(np.unique(val_targets_arr)) > 1:
            fpr, tpr, _ = roc_curve(val_targets_arr, val_preds_arr)
            roc_score = auc(fpr, tpr)
            axes[2].plot(fpr, tpr, color='#ff7f0e', lw=2.5, label=f'ST2-SyncNet (AUC = {roc_score:.4f})')
            axes[2].plot([0, 1], [0, 1], color='navy', linestyle='--', alpha=0.7)
            axes[2].set_title('Validation ROC Curve', fontsize=12, fontweight='bold')
            axes[2].set_xlabel('False Positive Rate')
            axes[2].set_ylabel('True Positive Rate')
            axes[2].grid(True, linestyle='--', alpha=0.6)
            axes[2].legend()

        plt.tight_layout()
        viz_dir = Path('visualizations')
        viz_dir.mkdir(parents=True, exist_ok=True)
        plot_save_path = viz_dir / 'real_multimodal_training_curves.png'
        plt.savefig(plot_save_path, dpi=300)
        print(f'[✓] Publication-quality training curves saved to: {plot_save_path}')

    # Sample Quick Verification
    if len(val_df) > 0 and best_checkpoint_path.exists():
        demo_video = val_df.iloc[0]['video_path']
        print(f'\n[*] Performing quick forensic verification on sample: {demo_video}')
        inf_model = ST2SyncNet().to(device)
        weights = torch.load(best_checkpoint_path, map_location=device)
        if isinstance(weights, dict) and 'model_state_dict' in weights:
            inf_model.load_state_dict(weights['model_state_dict'])
        else:
            inf_model.load_state_dict(weights)
        inf_model.eval()

        temp_df = pd.DataFrame([{'video_path': str(demo_video), 'label': 0, 'dataset': 'Test', 'actor_group': 'test'}])
        temp_ds = MultimodalDeepfakeDataset(temp_df, num_frames=args.num_frames, is_train=False)
        sample = temp_ds[0]

        spatial = sample['spatial'].unsqueeze(0).to(device)
        temporal = sample['temporal'].unsqueeze(0).to(device)

        with torch.no_grad():
            logits = inf_model(spatial, temporal)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        prob_real = probs[0] * 100
        prob_fake = probs[1] * 100
        verdict = 'REAL (Authentic Human Capture)' if prob_real >= 50.0 else 'FAKE (Deepfake / Synthetic Manipulation)'
        confidence = max(prob_real, prob_fake)

        print('=' * 75)
        print(f'          DEEPFAKE-AI MULTIMODAL FORENSIC VERDICT')
        print('=' * 75)
        print(f'  • Final Verdict     : {verdict}')
        print(f'  • Confidence        : {confidence:.2f}%')
        print(f'  • Probability REAL  : {prob_real:.2f}%')
        print(f'  • Probability FAKE  : {prob_fake:.2f}%')
        print('=' * 75)


if __name__ == '__main__':
    main()
