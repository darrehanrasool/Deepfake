"""Universal hardware detection, telemetry, and compute optimization."""

import os
import sys
import subprocess
from typing import Tuple, Dict, Any
import torch
import psutil


def detect_hardware(requested_device: str = 'auto') -> Tuple[torch.device, bool, Dict[str, Any]]:
    """Detects available compute acceleration (CUDA, MPS, CPU) and system telemetry.
    
    Args:
        requested_device: 'auto', 'cuda', 'mps', or 'cpu'
        
    Returns:
        (device, use_amp, telemetry_dict)
    """
    telemetry: Dict[str, Any] = {
        'os': f'{sys.platform.upper()} ({os.name})',
        'python': sys.version.split()[0],
        'pytorch': torch.__version__,
        'cpu_cores': os.cpu_count() or 4,
    }
    
    # System RAM Telemetry
    ram_info = psutil.virtual_memory()
    telemetry['ram_total_gb'] = ram_info.total / (1024 ** 3)
    telemetry['ram_available_gb'] = ram_info.available / (1024 ** 3)
    telemetry['ram_percent'] = ram_info.percent

    use_amp = False
    
    # NVIDIA CUDA Accelerator
    if requested_device == 'cuda' or (requested_device == 'auto' and torch.cuda.is_available()):
        device = torch.device('cuda')
        use_amp = True
        gpu_name = torch.cuda.get_device_name(0)
        vram_bytes = torch.cuda.get_device_properties(0).total_memory
        vram_gb = vram_bytes / (1024 ** 3)
        cuda_ver = torch.version.cuda
        
        telemetry['accelerator'] = f'NVIDIA CUDA GPU ({gpu_name})'
        telemetry['vram_gb'] = vram_gb
        telemetry['cuda_version'] = cuda_ver
        telemetry['amp_enabled'] = True
        
        torch.backends.cudnn.benchmark = True
        
        # Query GPU core sensors via nvidia-smi if available
        try:
            smi_out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=temperature.gpu,fan.speed,utilization.gpu,power.draw', '--format=csv,noheader,nounits'],
                text=True, stderr=subprocess.DEVNULL
            ).strip().split(',')
            if len(smi_out) >= 4:
                temp, fan, util, power = [x.strip() for x in smi_out[:4]]
                telemetry['gpu_sensors'] = {
                    'temperature_c': temp,
                    'fan_percent': fan,
                    'core_utilization_percent': util,
                    'power_draw_w': power
                }
        except Exception:
            telemetry['gpu_sensors'] = None

    # Apple Silicon Metal Performance Shaders (MPS)
    elif requested_device == 'mps' or (requested_device == 'auto' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        device = torch.device('mps')
        telemetry['accelerator'] = 'Apple Silicon Metal Performance Shaders (MPS)'
        telemetry['amp_enabled'] = False
        
    # High-Performance Multi-Threaded CPU Fallback
    else:
        device = torch.device('cpu')
        telemetry['accelerator'] = 'High-Performance Multi-Threaded CPU'
        telemetry['amp_enabled'] = False
        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

    telemetry['active_device'] = str(device)
    return device, use_amp, telemetry


def print_hardware_banner(telemetry: Dict[str, Any]):
    """Prints a standardized, professional system telemetry banner."""
    print('=' * 85)
    print('             DEEPFAKE-AI: SYSTEM HARDWARE & COMPUTE TELEMETRY')
    print('=' * 85)
    print(f" • Operating System      : {telemetry['os']}")
    print(f" • Python Runtime        : {telemetry['python']}")
    print(f" • PyTorch Version       : {telemetry['pytorch']}")
    print(f" • CPU Core Count        : {telemetry['cpu_cores']} logical cores")
    print(f" • Host System RAM       : {telemetry['ram_total_gb']:.2f} GB Total | {telemetry['ram_available_gb']:.2f} GB Available ({telemetry['ram_percent']}% used)")
    
    if 'accelerator' in telemetry:
        print(f" • Primary Accelerator   : {telemetry['accelerator']}")
    if 'vram_gb' in telemetry:
        print(f" • Total VRAM Allocated  : {telemetry['vram_gb']:.2f} GB")
    if 'cuda_version' in telemetry:
        print(f" • CUDA Toolkit Version  : {telemetry['cuda_version']}")
    
    amp_status = 'ENABLED (FP16 Accelerated)' if telemetry.get('amp_enabled') else 'Standard FP32'
    print(f" • Mixed Precision (AMP) : {amp_status}")
    
    sensors = telemetry.get('gpu_sensors')
    if sensors:
        print(f" • GPU Sensor Telemetry  : Temp: {sensors['temperature_c']}°C | Fan: {sensors['fan_percent']}% | Core Load: {sensors['core_utilization_percent']}% | Power: {sensors['power_draw_w']}W")
        
    print(f" • Active Compute Target : {telemetry['active_device']}")
    print('=' * 85)
