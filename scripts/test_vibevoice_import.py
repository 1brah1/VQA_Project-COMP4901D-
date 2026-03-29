#!/usr/bin/env python3
"""Test vibevoice import."""
import sys
print("[test_vibevoice] Python:", sys.version)
print("[test_vibevoice] Path:", '\n  '.join(sys.path[:3]))

try:
    print("[test_vibevoice] Importing vibevoice...")
    import vibevoice
    print("[test_vibevoice] SUCCESS! vibevoice imported from:", vibevoice.__file__)
except Exception as e:
    print(f"[test_vibevoice] FAILED: {e}")
    import traceback
    traceback.print_exc()
