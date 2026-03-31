#!/usr/bin/env python3
"""
Jetson preflight check: validates environment before running VQA + TTS pipeline.

Exit code 0: all checks passed
Exit code 1: one or more critical checks failed
"""
import json
import importlib.util
import os
import platform
import shutil
import sys
import argparse
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
        "message": "Python 3.8 compatible" if is_ok else f"Python 3.8+ required; found {version_str}"
    }


def check_transformers_version(min_major: int = 4, min_minor: int = 45) -> dict:
    """Check transformers minimum version required by current Jetson workflow."""
    try:
        import transformers

        version = transformers.__version__
        parts = version.split(".")
        major = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0
        minor_str = "".join(ch for ch in (parts[1] if len(parts) > 1 else "0") if ch.isdigit())
        minor = int(minor_str) if minor_str else 0
        is_ok = (major, minor) >= (min_major, min_minor)
        return {
            "check": "Transformers version",
            "expected": f">= {min_major}.{min_minor}",
            "actual": version,
            "passed": is_ok,
            "message": "Transformers version is compatible" if is_ok else f"transformers {version} too old"
        }
    except Exception as e:
        return {
            "check": "Transformers version",
            "expected": f">= {min_major}.{min_minor}",
            "actual": f"Error: {e}",
            "passed": False,
            "message": f"Could not import transformers: {e}"
        }


def check_accelerate_installed() -> dict:
    """Check accelerate availability and version visibility."""
    try:
        import accelerate
        version = getattr(accelerate, "__version__", "unknown")
        return {
            "check": "Accelerate package",
            "expected": "installed",
            "actual": version,
            "passed": True,
            "message": "accelerate is installed"
        }
    except Exception as e:
        return {
            "check": "Accelerate package",
            "expected": "installed",
            "actual": f"Error: {e}",
            "passed": False,
            "message": "accelerate is missing"
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
        is_ok = free_gb >= 0.5  # practical minimum for reports + short demo runs
        return {
            "check": "Disk space",
            "expected": "> 0.5 GB free",
            "actual": f"{free_gb:.1f} GB free / {total_gb:.1f} GB total",
            "passed": is_ok,
            "message": "Sufficient disk space" if is_ok else f"Low disk space: {free_gb:.1f} GB free (need ≥0.5 GB)"
        }
    except Exception as e:
        return {
            "check": "Disk space",
            "expected": "> 0.5 GB free",
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
        "TTS backends": repo_root / "src" / "tts" / "fallback_backends.py",
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

def check_piper_model(piper_model: str = None) -> dict:
    """Check whether a Piper model is present (optional but recommended)."""
    if piper_model is None:
        piper_model = str(Path.home() / "piper" / "models" / "en_US-lessac-medium.onnx")

    model_path = Path(piper_model)
    cfg_path = Path(str(model_path) + ".json")
    model_ok = model_path.exists()
    cfg_ok = cfg_path.exists()
    is_ok = model_ok and cfg_ok

    return {
        "check": "Piper model files",
        "expected": f"{model_path} and {cfg_path}",
        "actual": f"model={model_ok}, config={cfg_ok}",
        "passed": True,
        "message": "Piper model ready" if is_ok else "Piper model/config missing (optional; silero/pyttsx3 still usable)",
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
            "passed": True,
            "message": "sounddevice not installed; WAV generation still works without playback"
        }
    except Exception as e:
        return {
            "check": "Audio device",
            "expected": "sounddevice + audio device",
            "actual": f"Error: {e}",
            "passed": True,
            "message": f"Audio playback not available ({e}); non-playback TTS still supported"
        }

def check_dependencies() -> dict:
    """Check critical Python packages."""
    deps = ["transformers", "pillow", "numpy", "tqdm"]
    optional_deps = ["pyttsx3"]
    missing = []
    for dep in deps:
        try:
            if dep == "pillow":
                __import__("PIL")
            else:
                __import__(dep)
        except ImportError:
            missing.append(dep)

    missing_optional = []
    for dep in optional_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_optional.append(dep)
    
    is_ok = len(missing) == 0
    return {
        "check": "Required packages",
        "expected": "transformers, pillow, numpy, tqdm",
        "actual": f"{len(deps) - len(missing)}/{len(deps)} installed",
        "passed": is_ok,
        "message": (
            "All required dependencies installed"
            if is_ok and not missing_optional
            else (
                f"All required dependencies installed; optional missing: {', '.join(missing_optional)}"
                if is_ok
                else f"Missing: {', '.join(missing)}"
            )
        )
    }

def check_tts_backend_availability() -> dict:
    """Check that at least one TTS backend is importable."""
    piper_available = (shutil.which("piper") is not None) or (Path.home() / "piper" / "piper" / "piper").exists()
    silero_available = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("numpy") is not None
    pyttsx3_available = importlib.util.find_spec("pyttsx3") is not None
    is_ok = piper_available or pyttsx3_available or silero_available

    return {
        "check": "TTS backend availability",
        "expected": "piper or silero or pyttsx3",
        "actual": f"piper={piper_available}, silero={silero_available}, pyttsx3={pyttsx3_available}",
        "passed": is_ok,
        "message": (
            "At least one TTS backend is available"
            if is_ok
            else "No TTS backend available; install piper or pyttsx3"
        ),
    }


def check_shell_line_endings(repo_root: Path) -> dict:
    """Check Jetson shell scripts for CRLF line endings."""
    scripts_dir = repo_root / "scripts"
    targets = [
        scripts_dir / "jetson_run_benchmark.sh",
        scripts_dir / "jetson_quantize_llm_awq.sh",
        repo_root / "JETSON_RUN_VQA_TTS.sh",
    ]
    found_crlf = []
    checked = 0
    for p in targets:
        if not p.exists():
            continue
        checked += 1
        try:
            raw = p.read_bytes()
            if b"\r\n" in raw:
                found_crlf.append(p.name)
        except Exception:
            found_crlf.append(p.name)

    is_ok = len(found_crlf) == 0
    return {
        "check": "Shell line endings",
        "expected": "LF-only for jetson_*.sh",
        "actual": f"checked={checked}, crlf={len(found_crlf)}",
        "passed": is_ok,
        "message": "Shell scripts are LF-only" if is_ok else f"CRLF detected in: {', '.join(found_crlf)}"
    }


def _infer_expected_hidden_size(model_name_or_path: str):
    low = (model_name_or_path or "").lower()
    if "qwen" not in low:
        return None
    if "0.5b" in low or "0p5b" in low:
        return 1024
    if "1.5b" in low or "1p5b" in low:
        return 1536
    return None


def check_model_identity(model_name_or_path: str, allow_network: bool = False) -> dict:
    """Check expected hidden size against model config for the selected LLM."""
    expected_hidden = _infer_expected_hidden_size(model_name_or_path)
    if expected_hidden is None:
        return {
            "check": "LLM identity",
            "expected": "known hidden size",
            "actual": model_name_or_path,
            "passed": True,
            "message": "Skipped identity check for non-Qwen model name"
        }

    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name_or_path, local_files_only=(not allow_network))
        actual_hidden = getattr(cfg, "hidden_size", None)
        is_ok = actual_hidden == expected_hidden
        return {
            "check": "LLM identity",
            "expected": f"hidden_size={expected_hidden}",
            "actual": f"hidden_size={actual_hidden}",
            "passed": is_ok,
            "message": "LLM hidden_size matches expected" if is_ok else "LLM hidden_size mismatch"
        }
    except Exception as e:
        return {
            "check": "LLM identity",
            "expected": f"hidden_size={expected_hidden}",
            "actual": f"Error: {e}",
            "passed": False,
            "message": f"Could not validate model config for {model_name_or_path}"
        }

def main():
    """Run all preflight checks and report results."""
    parser = argparse.ArgumentParser(description="Jetson preflight checks for VQA pipeline")
    parser.add_argument(
        "--llm",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="LLM model id/path to validate identity against expected hidden size",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow model config downloads during identity check (default: cached/local only)",
    )
    args = parser.parse_args()

    # Determine repo root (script is in scripts/ subdirectory)
    repo_root = Path(__file__).parent.parent
    
    print("=" * 70)
    print("JETSON PREFLIGHT CHECK — VQA + Fallback-TTS Pipeline")
    print("=" * 70)
    
    checks = [
        check_python_version(),
        check_architecture(),
        check_torch_cuda(),
        check_memory(),
        check_disk_space(),
        check_transformers_version(),
        check_accelerate_installed(),
        check_dependencies(),
        check_shell_line_endings(repo_root),
        check_model_identity(args.llm, allow_network=args.allow_network),
        check_tts_backend_availability(),
        check_model_paths(repo_root),
        check_piper_model(),
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
        print("\n✓ All preflight checks passed. Ready to run VQA + fallback-TTS pipeline.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
