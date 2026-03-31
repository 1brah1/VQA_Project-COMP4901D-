#!/usr/bin/env python3
"""
Validate VQA + TTS integration on Jetson.
Tests imports and fallback TTS backend setup.
"""
import sys
import traceback
from pathlib import Path

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def test_imports():
    """Test all required imports."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    try:
        print("[1/7] Importing torch...", end=" ")
        import torch
        print(f"✓ (v{torch.__version__})")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
        
    try:
        print("[2/7] Importing transformers...", end=" ")
        import transformers
        print(f"✓ (v{transformers.__version__})")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
        
    try:
        print("[3/7] Importing src.tts.fallback_backends...", end=" ")
        from src.tts import fallback_backends
        print("✓")
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False
        
    try:
        print("[4/7] Importing src.vision components...", end=" ")
        from src.vision.siglip_encoder import SiglipPatchEncoder
        from src.vision.token_compression import compress_27x27_tokens
        print("✓")
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False
        
    try:
        print("[5/7] Importing src.vlm components...", end=" ")
        from src.vlm.model import SimplePrefixVLM
        from src.vlm.pipelined_vlm import SelfSpeculativeVLM
        print("✓")
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False
        
    try:
        print("[6/7] Importing fallback TTS components...", end=" ")
        from src.tts.fallback_backends import PiperTTSBackend, SileroTTSBackend, Pyttsx3TTSBackend
        print("✓")
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False
        
    try:
        print("[7/7] Checking piper executable...", end=" ")
        import shutil
        print("✓" if shutil.which("piper") else "⚠ Warning: piper not found in PATH")
    except Exception as e:
        print(f"⚠ Warning (non-fatal): {e}")
        
    return True


def test_fallback_backends():
    """Test that fallback backend classes can be instantiated."""
    print("\n" + "=" * 60)
    print("Testing fallback backends...")
    print("=" * 60)
    
    try:
        from src.tts.fallback_backends import PiperTTSBackend, SileroTTSBackend, Pyttsx3TTSBackend
        piper = PiperTTSBackend()
        silero = SileroTTSBackend(device="cpu")
        pyttsx = Pyttsx3TTSBackend()
        print(f"[1/3] Piper backend class: {'✓' if piper is not None else '✗'}")
        print(f"[2/3] Silero backend class: {'✓' if silero is not None else '✗'}")
        print(f"[3/3] pyttsx3 backend class: {'✓' if pyttsx is not None else '✗'}")
        return True
    except Exception as e:
        print(f"✗ Error checking fallback backends: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("\n" + "=" * 60)
    print("VQA + TTS Integration Validation")
    print("=" * 60 + "\n")
    
    # Test imports
    if not test_imports():
        print("\n✗ Import validation FAILED")
        return 1
        
    # Test fallback backend setup
    if not test_fallback_backends():
        print("\n✗ Fallback backend validation FAILED")
        return 1
        
    print("\n" + "=" * 60)
    print("✓ All validation tests PASSED!")
    print("=" * 60)
    print("\nNext: Run 'python scripts/run_pipelined.py' with appropriate args")
    return 0


if __name__ == "__main__":
    sys.exit(main())
