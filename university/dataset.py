"""Dual benchmark ingestion, leak-free actor-isolated splitting, and multimodal frame decoding."""

import json
import random
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset

from university.config import TrainingConfig


class MultimodalDeepfakeDataset(Dataset):
    """Production Multimodal Video Dataset with on-the-fly OpenCV frame tube decoding.
    
    Extracts:
    1. Spatial Representative Face/Frame (3, 224, 224)
    2. Temporal Sequential Tube of N Frames (N, 3, 224, 224)
    3. Synchronous spatial and temporal augmentations during training
    """

    def __init__(self, df: pd.DataFrame, num_frames: int = 8, img_size: int = 224, is_train: bool = True):
        self.df = df
        self.num_frames = num_frames
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self) -> int:
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

    def __getitem__(self, idx: int) -> Dict[str, Any]:
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


def discover_datasets(config: TrainingConfig) -> Tuple[pd.DataFrame, pd.DataFrame, torch.Tensor]:
    """Auto-discovers Google/Jigsaw DFD and Meta DFDC corpora and partitions leak-free.
    
    Returns:
        (train_df, val_df, class_weights_tensor)
    """
    print('\n[*] Scanning environment for Google/Jigsaw DFD and Meta DFDC video corpora...')
    records = []

    # 1. Ingest Google & Jigsaw DFD Benchmark
    dfd_found = None
    for cand in config.dfd_search_roots:
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
    for cand in config.dfdc_search_roots:
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
    np.random.seed(config.random_seed)
    np.random.shuffle(groups)

    val_group_count = max(1, int(len(groups) * config.val_split_ratio))
    val_groups = set(groups[:val_group_count])
    train_groups = set(groups[val_group_count:])

    train_df = corpus_df[corpus_df['actor_group'].isin(train_groups)].reset_index(drop=True)
    val_df = corpus_df[corpus_df['actor_group'].isin(val_groups)].reset_index(drop=True)

    print(f'\n[✓] Actor-Isolated Partitioning Completed:')
    print(f'  • Train Split : {len(train_df):5d} videos ({len(train_groups)} distinct actor clusters)')
    print(f'  • Val Split   : {len(val_df):5d} videos ({len(val_groups)} distinct actor clusters)')

    # Inverse Class Weights
    n_train = len(train_df)
    n_real = max(1, (train_df['label'] == 0).sum())
    n_fake = max(1, (train_df['label'] == 1).sum())
    weight_real = n_train / (2.0 * n_real)
    weight_fake = n_train / (2.0 * n_fake)
    class_weights = torch.tensor([weight_real, weight_fake], dtype=torch.float32)

    print(f'  • Class Balance: {n_real} REAL vs {n_fake} FAKE')
    print(f'  • Dynamic Loss Weights: REAL = {weight_real:.3f} | FAKE = {weight_fake:.3f}')

    return train_df, val_df, class_weights
