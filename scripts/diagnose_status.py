#!/usr/bin/env python3
"""GPU and training diagnostics."""

import subprocess
import sys
import psutil
from pathlib import Path

def check_gpu():
    """Check GPU availability and status."""
    print("🖥️  GPU STATUS")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', 
                                '--format=csv,noheader'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
        else:
            print("  [ERROR] nvidia-smi failed")
    except Exception as e:
        print(f"  [ERROR] {e}")


def check_cpu():
    """Check CPU and memory."""
    print("\n💾 SYSTEM STATUS")
    cpu_percent = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    print(f"  CPU:      {cpu_percent}%")
    print(f"  RAM:      {mem.used / 1e9:.1f}GB / {mem.total / 1e9:.1f}GB ({mem.percent}%)")
    

def check_python_env():
    """Check Python environment."""
    print("\n🐍 PYTHON ENVIRONMENT")
    print(f"  Python:   {sys.version.split()[0]}")
    
    # Check key packages
    packages = ['torch', 'transformers', 'peft', 'pillow']
    for pkg in packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  {pkg:15s}: {version}")
        except ImportError:
            print(f"  {pkg:15s}: NOT INSTALLED")
    
    # Check CUDA
    try:
        import torch
        print(f"  CUDA:      {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Ver:  {torch.version.cuda}")
            print(f"  Device:    {torch.cuda.get_device_name(0)}")
    except:
        pass


def check_training_status():
    """Check if training script is running."""
    print("\n⚙️  TRAINING PROCESS")
    try:
        result = subprocess.run(['tasklist'], capture_output=True, text=True)
        if 'python.exe' in result.stdout:
            # More details
            result2 = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/V'], 
                                   capture_output=True, text=True)
            print("  Python processes found:")
            for line in result2.stdout.split('\n')[3:]:  # Skip headers
                if line.strip():
                    print(f"    {line[:80]}")
        else:
            print("  No Python process found (may have completed)")
    except Exception as e:
        print(f"  [ERROR] {e}")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("DIAGNOSTIC CHECK")
    print("="*70)
    
    check_gpu()
    check_cpu()
    check_python_env()
    check_training_status()
    
    print("\n" + "="*70)
    print("\n✅ Diagnostics complete")
