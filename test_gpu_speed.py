#!/usr/bin/env python3
"""Test GPU speed after fix"""

import requests
import subprocess
import sys
import time
import io
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*80)
print("TESTING GPU SPEED (After Fix)")
print("="*80)

# Start server
print("\nStarting API server with GPU enabled...")
venv_python = Path("venv/Scripts/python.exe")
api_process = subprocess.Popen(
    [str(venv_python), "-m", "uvicorn", "adapter_service.standalone_api:app", 
     "--host", "127.0.0.1", "--port", "8117"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT
)

print("Loading model (should load on GPU)...")
time.sleep(40)

# Test 3 requests
print("\nTesting 3 requests:\n")
times = []

for i in range(3):
    try:
        start = time.time()
        resp = requests.post(
            "http://127.0.0.1:8117/generate-lite",
            json={
                "prompt": f"Translate to Hindi: Test number {i+1}",
                "max_new_tokens": 20,
                "adapter_path": "adapters/gurukul_lite",
                "base_model": "bigscience/bloomz-560m",
                "temperature": 0.5
            },
            timeout=30
        )
        duration = time.time() - start
        times.append(duration)
        
        if resp.status_code == 200:
            output = resp.json().get('generated_text', '')[:50]
            print(f"Test {i+1}: ✅ {duration:.2f}s - {output}")
        else:
            print(f"Test {i+1}: ❌ Error {resp.status_code}")
    except Exception as e:
        duration = time.time() - start
        print(f"Test {i+1}: ❌ {duration:.2f}s - {str(e)[:60]}")

api_process.kill()

if times:
    avg = sum(times) / len(times)
    print(f"\n{'='*80}")
    print(f"RESULTS:")
    print(f"  Average time: {avg:.2f}s")
    print(f"  Expected on CPU: ~15-18s")
    print(f"  Expected on GPU: ~2-4s")
    
    if avg < 6:
        print(f"\n✅ RUNNING ON GPU! ({avg:.2f}s avg - 5x faster!)")
    elif avg < 10:
        print(f"\n⚠️  Faster than before but might still be CPU ({avg:.2f}s avg)")
    else:
        print(f"\n❌ Still on CPU ({avg:.2f}s avg - same as before)")
    print(f"{'='*80}")

