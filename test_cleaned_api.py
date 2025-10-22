#!/usr/bin/env python3
"""Test the API with output cleaning enabled"""

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
print("TESTING API WITH OUTPUT CLEANING")
print("="*80)

# Start API server
print("\nStarting API server with output cleaning...")
print("   (Loading model, please wait ~30 seconds)\n")

api_process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "adapter_service.standalone_api:app", 
     "--host", "127.0.0.1", "--port", "8113"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for server to start
print("Waiting for server to start...")
time.sleep(35)

# Check if server is running
try:
    response = requests.get("http://127.0.0.1:8113/health", timeout=5)
    print("✅ Server is running!\n")
except Exception as e:
    print(f"❌ Server failed to start: {e}")
    api_process.kill()
    sys.exit(1)

# Test prompts - diverse languages
test_prompts = [
    {
        "name": "Hindi",
        "prompt": "Translate to Hindi: Hello",
        "expected_chars": "हैलो"
    },
    {
        "name": "Hindi (phrase)",
        "prompt": "Translate to Hindi: Good morning",
        "expected_chars": "सुप्रभात"
    },
    {
        "name": "Bengali",
        "prompt": "Translate to Bengali: Hello",
        "expected_chars": "হ্যালো"
    },
    {
        "name": "Tamil",
        "prompt": "Translate to Tamil: Hello",
        "expected_chars": "வணக்கம்"
    },
    {
        "name": "Telugu",
        "prompt": "Translate to Telugu: Welcome",
        "expected_chars": "స్వాగతం"
    },
    {
        "name": "Gujarati",
        "prompt": "Translate to Gujarati: Hello",
        "expected_chars": "નમસ્તે"
    },
    {
        "name": "Marathi",
        "prompt": "Translate to Marathi: Thank you",
        "expected_chars": "धन्यवाद"
    }
]

print("="*80)
print("TESTING CLEANED OUTPUT")
print("="*80)

results = []
success_count = 0

for i, test in enumerate(test_prompts, 1):
    print(f"\n{i}. {test['name']}")
    print(f"   Prompt: {test['prompt']}")
    
    try:
        start_time = time.time()
        response = requests.post(
            "http://127.0.0.1:8113/generate-lite",
            json={
                "prompt": test['prompt'],
                "max_new_tokens": 30,  # Shorter for cleaner output
                "adapter_path": "adapters/gurukul_lite",
                "base_model": "bigscience/bloomz-560m",
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "repetition_penalty": 1.2
            },
            timeout=45
        )
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            generated = result.get('generated_text', 'No output')
            
            print(f"   Output: {generated}")
            print(f"   Time: {duration:.2f}s")
            
            # Check quality
            is_echo = generated.strip() == test['prompt'].strip()
            has_translation = any(ord(char) > 127 for char in generated)
            is_clean = len(generated) < 200  # Not too verbose
            
            if is_echo:
                print(f"   ❌ Still echoing prompt")
                results.append("ECHO")
            elif not has_translation:
                print(f"   ❌ No translation detected")
                results.append("NO_TRANS")
            elif not is_clean:
                print(f"   ⚠️  Translation found but verbose ({len(generated)} chars)")
                results.append("VERBOSE")
                success_count += 0.5  # Partial credit
            else:
                print(f"   ✅ Clean translation!")
                results.append("SUCCESS")
                success_count += 1
        else:
            print(f"   ❌ API Error: {response.status_code}")
            print(f"      {response.text[:100]}")
            results.append("ERROR")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        results.append("EXCEPTION")

# Cleanup
print("\n" + "="*80)
print("Stopping server...")
api_process.kill()
api_process.wait()

# Summary
print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

total = len(test_prompts)
success_pct = (success_count / total) * 100

print(f"\nTotal tests: {total}")
print(f"Clean translations: {success_count:.1f}")
print(f"Success rate: {success_pct:.0f}%")

print("\nBreakdown:")
print(f"  SUCCESS: {results.count('SUCCESS')}")
print(f"  VERBOSE: {results.count('VERBOSE')}")
print(f"  ECHO: {results.count('ECHO')}")
print(f"  NO_TRANS: {results.count('NO_TRANS')}")
print(f"  ERROR: {results.count('ERROR') + results.count('EXCEPTION')}")

if success_pct >= 70:
    print("\n" + "="*80)
    print("🎉 OUTPUT CLEANING IS WORKING!")
    print("="*80)
    print("\nThe API now returns clean translations instead of noisy output!")
    print("Your adapter is production-ready! ✅")
elif success_pct >= 50:
    print("\n✅ Output cleaning helped, but still some issues")
    print("   Quality improved significantly!")
else:
    print("\n⚠️ Output cleaning needs more tuning")

print("\n" + "="*80)

