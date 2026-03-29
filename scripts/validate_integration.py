#!/usr/bin/env python3
"""
Validate VQA + TTS integration on Jetson.
Tests imports, compat patches, and basic pipeline setup.
"""
import sys
import traceback

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
        print("[3/7] Importing src.compat_patches...", end=" ")
        from src import compat_patches
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
        print("[6/7] Importing src.tts components...", end=" ")
        from src.tts.streaming_bridge import VibeVoiceTTSService, WordBufferedTTSBridge
        print("✓")
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return False
        
    try:
        print("[7/7] Importing vibevoice package...", end=" ")
        import vibevoice
        print(f"✓ (found at {vibevoice.__file__})")
    except Exception as e:
        print(f"⚠ Warning (expected if not fully installed): {e}")
        # Not fatal - VibeVoice import is optional
        
    return True


def test_compat_patches():
    """Test that compat patches were applied."""
    print("\n" + "=" * 60)
    print("Testing compat patches...")
    print("=" * 60)
    
    try:
        import transformers.modeling_flash_attention_utils as _m
        has_fak = hasattr(_m, 'FlashAttentionKwargs')
        print(f"[1/3] FlashAttentionKwargs shim: {'✓' if has_fak else '✗'}")
        
        import transformers.generation as _gen
        has_bs = hasattr(_gen, 'BaseStreamer')
        print(f"[2/3] BaseStreamer re-export: {'✓' if has_bs else '✗'}")
        
        from transformers import GenerationMixin
        has_pgc = hasattr(GenerationMixin, '_prepare_generation_config')
        print(f"[3/3] GenerationMixin patched: {'✓' if has_pgc else '✗'}")
        
        return has_fak and has_bs and has_pgc
    except Exception as e:
        print(f"✗ Error checking compat patches: {e}")
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
        
    # Test compat patches
    if not test_compat_patches():
        print("\n✗ Compat patch validation FAILED")
        return 1
        
    print("\n" + "=" * 60)
    print("✓ All validation tests PASSED!")
    print("=" * 60)
    print("\nNext: Run 'python scripts/run_pipelined.py' with appropriate args")
    return 0


if __name__ == "__main__":
    sys.exit(main())
