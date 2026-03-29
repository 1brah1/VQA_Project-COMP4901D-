#!/usr/bin/env python3
"""
Jetson preflight check: validates environment before running VQA + VibeVoice pipeline.

Exit code 0: all checks passed
Exit code 1: one or more critical checks failed
"""
import json
import os
import platform
import shutil
import sys
from pathlib import Path

def check_python_version() -> dict:
    """Check Python version compatibility."""
    major, minor, micro = sys.version_info[:3]
    version_str = f"{major}.{minor}.{micro}"
    is_ok = major == 3 and minor >= 8
    return {
        "check": "Python version",
        "expected": "3.8+",
        "actual": version_str,
        "passed": is_ok,
        "message": "Python 3.8 compatible" if is_ok else "Python 3.8+ required; found {version_str}"
    }

def check_torch_cuda() -> dict:
    """Check PyTorch and CUDA availability."""
    try:
        import torch
        torch_available = True
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU only"
    except ImportError:
        torch_available = False
        torch_version = "Not installed"
        cuda_available = False
        device_name = "N/A"
    
    is_ok = torch_available and cuda_available
    return {
        "check": "PyTorch + CUDA",
        "expected": "torch installed with CUDA support",
        "actual": f"torch {torch_version}, CUDA: {cuda_available}, device: {device_name}",
        "passed": is_ok,
        "message": "PyTorch CUDA ready" if is_ok else "PyTorch or CUDA support missing"
    }

def check_architecture() -> dict:
    """Check processor architecture."""
    arch = platform.machine()
    is_jetson = arch == "aarch64"
    return {
        "check": "Architecture",
        "expected": "aarch64 (Jetson)",
        "actual": arch,
        "passed": is_jetson,
        "message": "Jetson aarch64 detected" if is_jetson else f"Non-Jetson arch detected: {arch}. VLM may behave differently."
    }

def check_disk_space() -> dict:
    """Check available disk space."""
    try:
        stat = shutil.disk_usage("/")
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        is_ok = free_gb > 2.0  # 2 GB minimum for models + reports
        return {
            "check": "Disk space",
            "expected": "> 2 GB free",
            "actual": f"{free_gb:.1f} GB free / {total_gb:.1f} GB total",
            "passed": is_ok,
            "message": "Sufficient disk space" if is_ok else f"Low disk space: {free_gb:.1f} GB free (need ≥2 GB)"
        }
    except Exception as e:
        return {
            "check": "Disk space",
            "expected": "> 2 GB free",
            "actual": f"Error: {e}",
            "passed": False,
            "message": f"Could not check disk space: {e}"
        }

def check_memory() -> dict:
    """Check available GPU/CPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            is_ok = total_gb >= 12.0  # 12 GB minimum for Jetson Orin NX
            return {
                "check": "GPU Memory",
                "expected": "≥ 12 GB",
                "actual": f"{total_gb:.1f} GB total",
                "passed": is_ok,
                "message": "Sufficient GPU memory" if is_ok else f"GPU memory below threshold: {total_gb:.1f} GB (need ≥12 GB)"
            }
        else:
            return {
                "check": "GPU Memory",
                "expected": "CUDA device",
                "actual": "No CUDA device",
                "passed": False,
                "message": "CUDA not available; cannot run pipeline efficiently"
            }
    except Exception as e:
        return {
            "check": "GPU Memory",
            "expected": "≥ 12 GB",
            "actual": f"Error: {e}",
            "passed": False,
            "message": f"Could not check GPU memory: {e}"
        }

def check_model_paths(repo_root: Path) -> dict:
    """Check that critical model paths exist."""
    paths_to_check = {
        "SigLIP model": repo_root / "src" / "vision" / "siglip_encoder.py",
        "VLM model": repo_root / "src" / "vlm" / "model.py",
        "LLM loader": repo_root / "src" / "vlm" / "llm_loader.py",
        "TTS bridge": repo_root / "src" / "tts" / "streaming_bridge.py",
        "Data labels": repo_root / "data" / "eval" / "labels.json"
    }
    
    missing = [name for name, path in paths_to_check.items() if not path.exists()]
    is_ok = len(missing) == 0
    
    return {
        "check": "Model paths",
        "expected": "All files present",
        "actual": f"{len(paths_to_check) - len(missing)}/{len(paths_to_check)} present",
        "passed": is_ok,
        "message": "All critical model paths found" if is_ok else f"Missing: {', '.join(missing)}"
    }

def check_voices_directory(voices_root: str = None) -> dict:
    """Check that VibeVoice voice presets exist."""
    if voices_root is None:
        voices_root = os.path.expanduser("~/vibevoice_test/voices")
    
    voices_path = Path(voices_root)
    
    if not voices_path.exists():
        return {
            "check": "VibeVoice voices directory",
            "expected": f"{voices_root} with .pt presets",
            "actual": "Directory does not exist",
            "passed": False,
            "message": f"Voices directory not found at {voices_root}. Verify VibeVoice installation on Jetson."
        }
    
    pt_files = list(voices_path.glob("*.pt"))
    is_ok = len(pt_files) > 0
    
    return {
        "check": "VibeVoice voices directory",
        "expected": ".pt voice presets present",
        "actual": f"{len(pt_files)} preset files found",
        "passed": is_ok,
        "message": f"Voice presets ready ({len(pt_files)} presets)" if is_ok else f"No .pt preset files found in {voices_root}"
    }

def check_audio_device() -> dict:
    """Check sounddevice availability and audio device detection."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        device_count = len(devices) if isinstance(devices, list) else 1
        return {
            "check": "Audio device",
            "expected": "sounddevice + audio device detected",
            "actual": f"{device_count} audio device(s) available",
            "passed": device_count > 0,
            "message": f"Audio ready ({device_count} device(s))" if device_count > 0 else "No audio devices detected"
        }
    except ImportError:
        return {
            "check": "Audio device",
            "expected": "sounddevice installed",
            "actual": "sounddevice not installed",
            "passed": False,
            "message": "sounddevice not installed; audio playback will be unavailable"
        }
    except Exception as e:
        return {
            "check": "Audio device",
            "expected": "sounddevice + audio device",
            "actual": f"Error: {e}",
            "passed": False,
            "message": f"Audio check failed: {e}"
        }

def check_dependencies() -> dict:
    """Check critical Python packages."""
    deps = ["transformers", "pillow", "numpy", "tqdm"]
    missing = []
    for dep in deps:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    is_ok = len(missing) == 0
    return {
        "check": "Required packages",
        "expected": "transformers, pillow, numpy, tqdm",
        "actual": f"{len(deps) - len(missing)}/{len(deps)} installed",
        "passed": is_ok,
        "message": "All dependencies installed" if is_ok else f"Missing: {', '.join(missing)}"
    }

def main():
    """Run all preflight checks and report results."""
    # Determine repo root (script is in scripts/ subdirectory)
    repo_root = Path(__file__).parent.parent
    
    print("=" * 70)
    print("JETSON PREFLIGHT CHECK — VQA + VibeVoice Pipeline")
    print("=" * 70)
    
    checks = [
        check_python_version(),
        check_architecture(),
        check_torch_cuda(),
        check_memory(),
        check_disk_space(),
        check_dependencies(),
        check_model_paths(repo_root),
        check_voices_directory(),
        check_audio_device(),
    ]
    
    print("\nRESULTS:")
    print("-" * 70)
    
    passed_count = 0
    failed_checks = []
    
    for check in checks:
        status = "✓ PASS" if check["passed"] else "✗ FAIL"
        print(f"{status:8} | {check['check']:25} | {check['message']}")
        if check["passed"]:
            passed_count += 1
        else:
            failed_checks.append(check)
    
    print("-" * 70)
    print(f"\nSummary: {passed_count}/{len(checks)} checks passed")
    
    if failed_checks:
        print("\nFailed checks:")
        for check in failed_checks:
            print(f"  • {check['check']}: {check['message']}")
        print("\nAction required:")
        print("  1. Review failures above")
        print("  2. Install missing dependencies or fix paths")
        print("  3. Re-run this script to verify")
        return 1
    else:
        print("\n✓ All preflight checks passed. Ready to run VQA + VibeVoice pipeline.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
