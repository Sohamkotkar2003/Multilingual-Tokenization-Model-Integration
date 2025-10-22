#!/usr/bin/env python3
"""
Test the Colab-trained adapter

This tests the adapter you just trained on Google Colab!
"""

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
print("TESTING COLAB-TRAINED ADAPTER")
print("="*80)

# Check if adapter exists
from pathlib import Path

adapter_path = Path("adapters/gurukul_lite")
if not adapter_path.exists():
    print("\nERROR: Adapter not found at adapters/gurukul_lite/")
    print("   Did you extract the files there?")
    sys.exit(1)

# Check for required files
required_files = ["adapter_config.json", "adapter_model.safetensors"]
missing = []
for file in required_files:
    if not (adapter_path / file).exists():
        missing.append(file)

if missing:
    print(f"\nWARNING: Missing files: {missing}")
    print("   The adapter might not work correctly.")
else:
    print("\nAdapter files found!")

# List all files in adapter directory
print("\nFiles in adapters/gurukul_lite/:")
for file in sorted(adapter_path.iterdir()):
    if file.is_file():
        size_kb = file.stat().st_size / 1024
        print(f"  - {file.name:40s} ({size_kb:,.1f} KB)")

# Start API server
print("\nStarting API server...")
print("   (This will take ~30 seconds to load the model)\n")

api_process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "adapter_service.standalone_api:app", 
     "--host", "127.0.0.1", "--port", "8111"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for server to start
print("Waiting for server to start...")
time.sleep(35)

# Check if server is running
try:
    response = requests.get("http://127.0.0.1:8111/health", timeout=5)
    print("Server is running!\n")
except Exception as e:
    print(f"Server failed to start: {e}")
    api_process.kill()
    sys.exit(1)

# Test prompts in multiple languages
test_prompts = [
    {
        "name": "Hindi Translation",
        "prompt": "Translate to Hindi: Hello friend, how are you?",
        "max_new_tokens": 50
    },
    {
        "name": "Bengali Translation", 
        "prompt": "Translate to Bengali: Good morning, have a nice day.",
        "max_new_tokens": 50
    },
    {
        "name": "Tamil Translation",
        "prompt": "Translate to Tamil: Thank you very much for your help.",
        "max_new_tokens": 50
    },
    {
        "name": "Telugu Translation",
        "prompt": "Translate to Telugu: Welcome to our school.",
        "max_new_tokens": 50
    },
    {
        "name": "Gujarati Translation",
        "prompt": "Translate to Gujarati: How can I help you?",
        "max_new_tokens": 50
    }
]

print("="*80)
print("TESTING WITH ADAPTER")
print("="*80)

results = []

for i, test in enumerate(test_prompts, 1):
    print(f"\n{i}. {test['name']}")
    print(f"   Prompt: {test['prompt']}")
    
    try:
        start_time = time.time()
        response = requests.post(
            "http://127.0.0.1:8111/generate-lite",
            json={
                "prompt": test['prompt'],
                "max_new_tokens": test['max_new_tokens'],
                "adapter_path": "adapters/gurukul_lite",
                "base_model": "bigscience/bloomz-560m",  # Match the adapter's base model!
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.2
            },
            timeout=60
        )
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            generated = result.get('generated_text', 'No output')
            
            # Check if it's just echoing the prompt
            is_echo = generated.strip() == test['prompt'].strip()
            
            print(f"   Output: {generated[:200]}")
            print(f"   Duration: {duration:.2f}s")
            
            if is_echo:
                print("   WARNING: Model is echoing the prompt!")
                results.append("ECHO")
            else:
                print("   Success!")
                results.append("OK")
        else:
            print(f"   Failed: {response.status_code}")
            print(f"   {response.text[:200]}")
            results.append("FAIL")
            
    except Exception as e:
        print(f"   Error: {e}")
        results.append("ERROR")

# Cleanup
print("\n" + "="*80)
print("Stopping server...")
api_process.kill()
api_process.wait()

print("\n" + "="*80)
print("TEST RESULTS SUMMARY")
print("="*80)

success_count = results.count("OK")
echo_count = results.count("ECHO")
fail_count = results.count("FAIL") + results.count("ERROR")

print(f"\nTotal tests: {len(results)}")
print(f"  Success: {success_count}")
print(f"  Echo (needs improvement): {echo_count}")
print(f"  Failed: {fail_count}")

if success_count == len(results):
    print("\n🎉 ALL TESTS PASSED! Your adapter is working perfectly!")
elif success_count > 0:
    print(f"\n✅ {success_count}/{len(results)} tests passed. Adapter is working!")
    if echo_count > 0:
        print("   Note: Some outputs are echoing. Try training with more samples/epochs.")
elif echo_count == len(results):
    print("\n⚠️ Adapter is loaded but only echoing prompts.")
    print("   Recommendations:")
    print("   1. Train with more samples (10k-20k instead of 5k)")
    print("   2. Train for more epochs (5 instead of 3)")
    print("   3. Check your training data quality")
else:
    print("\n❌ Adapter tests failed. Check errors above.")

print("\n" + "="*80)

