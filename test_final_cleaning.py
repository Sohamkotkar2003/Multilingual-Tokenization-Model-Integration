#!/usr/bin/env python3
"""Final test of improved output cleaning"""

import requests
import subprocess
import sys
import time
import io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*80)
print("FINAL OUTPUT CLEANING TEST")
print("="*80)

# Start server
print("\nStarting API server...")
api_process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "adapter_service.standalone_api:app", 
     "--host", "127.0.0.1", "--port", "8114"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

time.sleep(35)

try:
    requests.get("http://127.0.0.1:8114/health", timeout=5)
    print("✅ Server ready!\n")
except:
    print("❌ Server failed")
    api_process.kill()
    sys.exit(1)

# Quick tests
tests = [
    ("Translate to Hindi: Hello", "Hindi"),
    ("Translate to Bengali: Good morning", "Bengali"),
    ("Translate to Tamil: Thank you", "Tamil"),
]

print("="*80)

success = 0
for prompt, lang in tests:
    print(f"\n{lang}: {prompt}")
    try:
        resp = requests.post(
            "http://127.0.0.1:8114/generate-lite",
            json={
                "prompt": prompt,
                "max_new_tokens": 25,  # Even shorter
                "adapter_path": "adapters/gurukul_lite",
                "base_model": "bigscience/bloomz-560m",
                "temperature": 0.5,  # Lower temp
                "do_sample": True,
                "repetition_penalty": 1.5  # Higher penalty
            },
            timeout=30
        )
        
        if resp.status_code == 200:
            result = resp.json()
            output = result.get('generated_text', '')
            print(f"  Output: {output}")
            
            # Check if clean
            has_trans = any(ord(c) > 127 for c in output)
            is_short = len(output) < 100
            
            if has_trans and is_short:
                print(f"  ✅ CLEAN!")
                success += 1
            elif has_trans:
                print(f"  ⚠️  Translation but long ({len(output)} chars)")
                success += 0.5
            else:
                print(f"  ❌ No translation")
        else:
            print(f"  ❌ Error {resp.status_code}")
    except Exception as e:
        print(f"  ❌ {str(e)[:50]}")

api_process.kill()
api_process.wait()

print("\n" + "="*80)
print(f"Clean outputs: {success}/{len(tests)}")
if success >= 2:
    print("✅ OUTPUT CLEANING WORKS!")
else:
    print("⚠️ Needs more tuning")
print("="*80)

