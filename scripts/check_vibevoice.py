#!/usr/bin/env python3
"""Check and setup VibeVoice on Jetson."""
import sys
import os
import subprocess
from pathlib import Path

print("Checking VibeVoice availability...")

# Check vibevoice_test venv
vibevoice_test_path = Path("/home/comp4901d/vibevoice_test")
if vibevoice_test_path.exists():
    print(f"✓ vibevoice_test directory exists at {vibevoice_test_path}")
   
    # Check if VibeVoice is in site-packages
    site_packages = vibevoice_test_path / ".venv/lib/python3.8/site-packages"
    vibevoice_dirs = list(site_packages.glob("vibevoice*"))
    if vibevoice_dirs:
       print(f"✓ Found vibevoice package(s): {vibevoice_dirs}")
    else:
        print("✗ No vibevoice package found in vibevoice_test venv")
        
    # Check if vibevoice is installed as editable or local
    egg_link = site_packages / "vibevoice.egg-link"
    if egg_link.exists():
        print(f"✓ vibevoice is installed as editable package")
        print(f"  Link contents: {egg_link.read_text()}")
else:
    print(f"✗ vibevoice_test not found at {vibevoice_test_path}")

# Try to find if VibeVoice repo exists anywhere
print("\nSearching for VibeVoice repository...")
result = subprocess.run(["find", "/home/comp4901d", "-maxdepth", "3", "-type", "d", "-name", "VibeVoice"], 
                       capture_output=True, text=True, timeout=10)
if result.stdout:
    print(f"Found VibeVoice directories:\n{result.stdout}")
else:
    print("No VibeVoice directory found")

# Check what's in vibevoice_test
print(f"\nContents of {vibevoice_test_path}:")
if vibevoice_test_path.exists():
    for item in sorted(vibevoice_test_path.iterdir()):
        if item.is_dir():
            print(f"  [DIR] {item.name}/")
        else:
            size_kb = item.stat().st_size / 1024
            print(f"  [FILE] {item.name} ({size_kb:.1f} KB)")
