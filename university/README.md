# DeepFake-AI: University Multimodal Training Package

This directory houses the modular, production-grade training architecture designed for university workstations, computing clusters, and personal computers.

## Architectural Module Layout

```
university/
├── __init__.py           # Package exports & public API
├── config.py             # Hyperparameters, directory paths, dataclasses
├── hardware.py           # Universal hardware detection (CUDA, MPS, CPU), VRAM & sensors
├── dataset.py            # Dual-benchmark discovery (DFD & DFDC), leak-free splitting, OpenCV tubes
├── model.py              # ST2-SyncNet (Wavelet 2D DWT, Spatial CNN, Bi-GRU, Cross-Attention)
├── trainer.py            # SOTA training engine, label smoothing loss, dynamic class weights, ASCII progress
├── checkpoints.py        # Atomic checkpoint persistence & automatic session resumption
├── visualization.py      # Publication-grade performance plots (Loss, ROC-AUC, ROC Curve)
├── verify.py             # Single-sample forensic verification inference
├── main.py               # Master CLI entrypoint
└── README.md             # Package documentation
```

## How to Execute

### Option 1: From the Project Root (Easiest)
```bash
python train.py
```

### Option 2: As a Python Module
```bash
python -m university.main
```

### Option 3: Directly via Main Script
```bash
python university/main.py --epochs 10 --batch-size 8
```

## Features & Guarantees
1. **Decoupled Architecture**: Follows clean code principles where modeling, data loading, hardware detection, training, and metrics logging are strictly separated.
2. **Zero-Leakage Guarantee**: Uses strict actor-isolated group partitioning across both Google DFD and Meta DFDC benchmarks.
3. **Hardware Autonomy**: Automatically configures NVIDIA CUDA with FP16 Automatic Mixed Precision (AMP), queries VRAM and `nvidia-smi` sensors, or falls back gracefully to MPS or multi-threaded CPU.
4. **Interruption-Proof Resumption**: Saves `latest_checkpoint.pt` after every epoch. Rerunning resumes seamlessly from the interrupted epoch without restarting.
