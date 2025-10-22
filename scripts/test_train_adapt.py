#!/usr/bin/env python3
"""
Test script for train_adapt.py

Runs a TINY training job (100 samples) to verify everything works
"""

import subprocess
import sys
import io
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 80)
print("TESTING train_adapt.py")
print("=" * 80)
print()
print("This will run a TINY training job (100 samples) to test if everything works.")
print("This should take about 2-5 minutes.")
print()
print("=" * 80)
print()

# Run training with 100 samples
cmd = [
    sys.executable,
    "adapter_service/train_adapt.py",
    "--config", "adapter_config.yaml",
    "--max-samples", "100"
]

print(f"Running: {' '.join(cmd)}")
print()

try:
    result = subprocess.run(
        cmd,
        capture_output=False,  # Show output in real-time
        text=True,
        timeout=600  # 10 minute timeout
    )
    
    if result.returncode == 0:
        print()
        print("=" * 80)
        print("TEST PASSED!")
        print("=" * 80)
        print("Training completed successfully with 100 samples.")
        print("Now try with more samples: --max-samples 1000")
        sys.exit(0)
    else:
        print()
        print("=" * 80)
        print("TEST FAILED")
        print("=" * 80)
        print(f"Exit code: {result.returncode}")
        sys.exit(1)
        
except subprocess.TimeoutExpired:
    print()
    print("=" * 80)
    print("TEST TIMEOUT")
    print("=" * 80)
    print("Training took longer than 10 minutes - this might indicate it's stuck.")
    sys.exit(1)
    
except Exception as e:
    print()
    print("=" * 80)
    print("TEST ERROR")
    print("=" * 80)
    print(f"Error: {e}")
    sys.exit(1)

