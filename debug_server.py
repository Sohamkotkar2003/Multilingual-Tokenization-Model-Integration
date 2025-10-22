#!/usr/bin/env python3
"""Debug why the server is timing out"""

import requests
import subprocess
import sys
import time
import io
from pathlib import Path

# Fix Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("Starting server for debugging...")

# Start server
venv_python = Path("venv/Scripts/python.exe")
api_process = subprocess.Popen(
    [str(venv_python), "-m", "uvicorn", "adapter_service.standalone_api:app", 
     "--host", "127.0.0.1", "--port", "8116"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True,
    bufsize=1
)

print("Waiting for server to start...")
time.sleep(40)

print("\nTesting 3 requests in a row to see what happens:\n")

for i in range(3):
    print(f"Test {i+1}:")
    try:
        start = time.time()
        resp = requests.post(
            "http://127.0.0.1:8116/generate-lite",
            json={
                "prompt": f"Translate to Hindi: Test {i+1}",
                "max_new_tokens": 20,
                "adapter_path": "adapters/gurukul_lite",
                "base_model": "bigscience/bloomz-560m",
                "temperature": 0.5,
                "repetition_penalty": 1.5
            },
            timeout=30
        )
        duration = time.time() - start
        if resp.status_code == 200:
            output = resp.json().get('generated_text', '')
            print(f"  ✅ Success in {duration:.1f}s: {output[:50]}")
        else:
            print(f"  ❌ Error {resp.status_code}")
    except Exception as e:
        duration = time.time() - start
        print(f"  ❌ Exception after {duration:.1f}s: {str(e)[:80]}")
    
    print()

print("Checking server process...")
if api_process.poll() is None:
    print("✅ Server still running")
    # Try to read server output
    try:
        api_process.terminate()
        stdout, _ = api_process.communicate(timeout=5)
        print("\nServer output:")
        print(stdout[-1000:] if len(stdout) > 1000 else stdout)
    except:
        api_process.kill()
else:
    print("❌ Server crashed!")
    stdout, _ = api_process.communicate()
    print("\nServer output:")
    print(stdout[-1000:] if len(stdout) > 1000 else stdout)

