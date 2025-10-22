#!/usr/bin/env python3
"""Quick focused test with shorter outputs"""

import requests
import json
import subprocess
import sys
import time
import io

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*80)
print("QUICK ADAPTER TEST (Short outputs)")
print("="*80)

# Start API server
print("\nStarting API server...")
api_process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "adapter_service.standalone_api:app", 
     "--host", "127.0.0.1", "--port", "8112"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

time.sleep(35)

# Test with VERY short outputs
test_prompts = [
    {
        "name": "Hindi (short)",
        "prompt": "Translate to Hindi: Hello",
        "max_new_tokens": 10
    },
    {
        "name": "Bengali (short)", 
        "prompt": "Translate to Bengali: Good morning",
        "max_new_tokens": 10
    },
    {
        "name": "Tamil (short)",
        "prompt": "Translate to Tamil: Thank you",
        "max_new_tokens": 10
    }
]

print("\n" + "="*80)
print("TESTING")
print("="*80)

for i, test in enumerate(test_prompts, 1):
    print(f"\n{i}. {test['name']}")
    print(f"   Input: {test['prompt']}")
    
    try:
        start_time = time.time()
        response = requests.post(
            "http://127.0.0.1:8112/generate-lite",
            json={
                "prompt": test['prompt'],
                "max_new_tokens": test['max_new_tokens'],
                "adapter_path": "adapters/gurukul_lite",
                "base_model": "bigscience/bloomz-560m",
                "temperature": 0.3,  # Lower temp for more focused output
                "do_sample": False,  # Greedy decoding for best quality
                "repetition_penalty": 1.5
            },
            timeout=30
        )
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            generated = result.get('generated_text', 'No output')
            
            print(f"   Output: {generated}")
            print(f"   Time: {duration:.2f}s")
        else:
            print(f"   Failed: {response.status_code} - {response.text[:100]}")
            
    except Exception as e:
        print(f"   Error: {e}")

# Cleanup
print("\n" + "="*80)
api_process.kill()
api_process.wait()
print("Done!")

