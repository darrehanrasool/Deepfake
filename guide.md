# DeepFake-AI: User and Developer Guide

Welcome to **DeepFake-AI**, a state-of-the-art framework for deepfake detection, forensic artifact analysis, and multimodal media verification.

## 1. Quick Start Commands

```bash
# Setup environment and directories
make setup

# Download / verify datasets
python3 scripts/download_datasets.py --mode sample
python3 scripts/verify_datasets.py

# Prepare data (face detection, alignment, frequency transforms)
python3 scripts/prepare_data.py

# Train models
python3 scripts/train_all.py --config configs/training.yaml

# Run evaluation benchmarks
python3 scripts/evaluate_all.py

# Launch interactive forensics dashboard
python3 scripts/run_dashboard.py
```

## 2. Supported Datasets & Benchmarks
- **Google & Jigsaw DFD**: 22.5 GB / 3,431 high-definition videos with original and manipulated sequences
- **DFDC (Deepfake Detection Challenge)**: 400 multimodal video corpus with official metadata and actor groups
- **FaceForensics++ (FF++)**: Deepfakes, Face2Face, FaceSwap, NeuralTextures, FaceShifter (c23 / c40)
- **Celeb-DF v2**: High-quality face swapping with subtle boundaries
- **FakeAVCeleb**: Multimodal audio-visual manipulation and lipsync anomalies
- **GenImage**: Diffusion-generated faces (Stable Diffusion, Midjourney, DALL-E) vs GANs

## 3. Master Multimodal Training Pipeline
The master training pipeline is available as both a direct standalone Python script and a Jupyter notebook:
- **Direct Python Script (Recommended for University PC):** `train.py` (Run via `python train.py`)
- **Interactive Jupyter Notebook:** `notebooks/01_master_multimodal_training.ipynb`

### Features:
- **Universal Hardware Autonomy**: Automatically detects NVIDIA CUDA (with FP16 AMP & VRAM profiling), Apple Silicon MPS, or CPU multi-threading.
- **Dual Benchmark Training**: Discovers and ingests both Google/Jigsaw DFD and Meta DFDC with strict actor-isolated partitioning.
- **ST2-SyncNet Multimodal Forensics**: Tri-stream architecture modeling Spatial boundaries, 12-subband 2D Haar Wavelet frequencies, and 8-frame Bi-GRU biological continuity.
- **Anti-Overfitting & Anti-Underfitting Guarantee**: Balanced class re-weighting, Label Smoothing (0.05), AdamW Decoupled Weight Decay (1e-4), Dropout (0.4), Gradient Norm Clipping (1.0), and Cosine Annealing Learning Rate decay with warmup.
- **Interactive Console Progress**: Step-by-step progress logging with running loss, step latency, and ETA.
- **Checkpointing & Resumability**: Auto-saves `best_model.pt` and `latest_checkpoint.pt` for interruption-free recovery.

## 4. Supported Model Architectures
- **Multimodal SOTA**: ST2-SyncNet (Spatial-Spectral-Temporal Synchronous Attention Network)
- **Spatial Deep Models**: MesoNet (Meso4, MesoInception4), XceptionNet, EfficientNet-B4
- **Frequency Domain**: Dual-stream Spatial-Frequency Network (RGB + 2D-DCT high-pass residual)
- **Spatio-Temporal**: Spatio-temporal ResNet-LSTM / 3D-CNN for temporal flickering & unnatural blinking
- **Diffusion Forensics**: DIRE-inspired latent reconstruction error classifier
- **Ensemble**: Soft/Hard voting & Stacking ensemble

## 5. Research Attribution & Academic Leadership
- **Dar Rehan Rasool** (`M.Tech Student, Centre for Artificial Intelligence, IUST`)
- **Dr. Riyaz Ahmed** (`M.Tech Student, Centre for Artificial Intelligence, IUST`)
- **Dr. M. Abdul Jawad** (`M.Tech Student, Centre for Artificial Intelligence, IUST`)

*Copyright (c) 2026 Dar Rehan Rasool (@darrehanrasool). All rights reserved.*

