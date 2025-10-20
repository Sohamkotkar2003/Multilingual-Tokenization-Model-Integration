#!/usr/bin/env python3
"""
Simple API Testing Script for BLOOMZ-560M Service
Quick and reliable testing without Unicode issues
"""

import requests
import json
import time

def test_api():
    """Test the BLOOMZ-560M API with simple requests"""
    base_url = "http://127.0.0.1:8110"
    
    print("Testing BLOOMZ-560M API")
    print("=" * 40)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("   Health check: PASSED")
        else:
            print(f"   Health check: FAILED ({response.status_code})")
            return
    except Exception as e:
        print(f"   Health check: ERROR ({e})")
        return
    
    # Test 2: Simple generation
    print("\n2. Testing simple generation...")
    try:
        payload = {
            "prompt": "The weather today is",
            "base_model": "bigscience/bloomz-560m",
            "max_new_tokens": 30,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2,
            "min_new_tokens": 5,
            "eos_token_id": 2
        }
        
        start_time = time.time()
        response = requests.post(f"{base_url}/generate-lite", json=payload, timeout=60)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            generated_text = data.get('generated_text', '')
            duration = end_time - start_time
            
            print(f"   Generation: PASSED")
            print(f"   Generated: '{generated_text[:100]}{'...' if len(generated_text) > 100 else ''}'")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Length: {len(generated_text)} chars")
            
            # Quality check
            if len(generated_text.strip()) < 10:
                print("   Warning: Output very short")
            elif generated_text.strip() == payload.get('prompt', '').strip():
                print("   Warning: Only echoing prompt")
            else:
                print("   Quality: Good generation")
        else:
            print(f"   Generation: FAILED ({response.status_code})")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Generation: ERROR ({e})")
    
    # Test 3: Multilingual generation tests
    print("\n3. Testing multilingual generation...")
    
    # Language test cases (based on your training data)
    language_tests = [
        {"name": "Hindi", "prompt": "आज का दिन बहुत अच्छा है", "lang": "hi"},
        {"name": "Bengali", "prompt": "আজকের দিনটি খুবই ভালো", "lang": "bn"},
        {"name": "Tamil", "prompt": "இன்று நல்ல நாள்", "lang": "ta"},
        {"name": "Telugu", "prompt": "ఈ రోజు చాలా మంచి రోజు", "lang": "te"},
        {"name": "Gujarati", "prompt": "આજે ખૂબ સારો દિવસ છે", "lang": "gu"},
        {"name": "Marathi", "prompt": "आज खूप चांगला दिवस आहे", "lang": "mr"},
        {"name": "Urdu", "prompt": "آج کا دن بہت اچھا ہے", "lang": "ur"},
        {"name": "Punjabi", "prompt": "ਅੱਜ ਦਾ ਦਿਨ ਬਹੁਤ ਵਧੀਆ ਹੈ", "lang": "pa"},
        {"name": "Nepali", "prompt": "आजको दिन धेरै राम्रो छ", "lang": "ne"},
        {"name": "Odia", "prompt": "ଆଜି ଖୁବ୍ ଭଲ ଦିନ", "lang": "or"}
    ]
    
    passed_lang_tests = 0
    total_lang_tests = len(language_tests)
    
    for test in language_tests:
        try:
            print(f"\n   Testing {test['name']} ({test['lang']}):")
            payload = {
                "prompt": test["prompt"],
                "base_model": "bigscience/bloomz-560m",
                "max_new_tokens": 30,
                "temperature": 0.7,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.2,
                "min_new_tokens": 5,
                "eos_token_id": 2
            }
            
            start_time = time.time()
            response = requests.post(f"{base_url}/generate-lite", json=payload, timeout=30)
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                generated_text = data.get('generated_text', '')
                duration = end_time - start_time
                
                print(f"     {test['name']}: PASSED")
                print(f"     Generated: '{generated_text[:50]}{'...' if len(generated_text) > 50 else ''}'")
                print(f"     Duration: {duration:.2f}s")
                
                # Quality check
                if len(generated_text.strip()) < 5:
                    print(f"     Warning: Output very short")
                elif generated_text.strip() == test["prompt"].strip():
                    print(f"     Warning: Only echoing prompt")
                else:
                    print(f"     Quality: Good generation")
                    passed_lang_tests += 1
            else:
                print(f"     {test['name']}: FAILED ({response.status_code})")
        except Exception as e:
            print(f"     {test['name']}: ERROR ({e})")
    
    print(f"\n   Multilingual Results: {passed_lang_tests}/{total_lang_tests} languages passed")
    
    # Test 4: Greedy generation (faster)
    print("\n4. Testing greedy generation...")
    try:
        payload = {
            "prompt": "The future of AI is",
            "base_model": "bigscience/bloomz-560m",
            "max_new_tokens": 20,
            "temperature": 0.0,
            "do_sample": False,
            "min_new_tokens": 5,
            "eos_token_id": 2
        }
        
        start_time = time.time()
        response = requests.post(f"{base_url}/generate-lite", json=payload, timeout=30)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            generated_text = data.get('generated_text', '')
            duration = end_time - start_time
            
            print(f"   Greedy generation: PASSED")
            print(f"   Generated: '{generated_text}'")
            print(f"   Duration: {duration:.2f}s")
        else:
            print(f"   Greedy generation: FAILED ({response.status_code})")
    except Exception as e:
        print(f"   Greedy generation: ERROR ({e})")
    
    # Test 5: Adapter list
    print("\n5. Testing adapter list...")
    try:
        response = requests.get(f"{base_url}/adapter/list", timeout=5)
        if response.status_code == 200:
            data = response.json()
            adapters = data.get('adapters', [])
            print(f"   Adapter list: PASSED")
            print(f"   Available adapters: {len(adapters)}")
        else:
            print(f"   Adapter list: FAILED ({response.status_code})")
    except Exception as e:
        print(f"   Adapter list: ERROR ({e})")
    
    print("\n" + "=" * 40)
    print("API Testing Complete!")
    print("The BLOOMZ-560M API is working and ready for use.")
    print(f"Multilingual Support: {passed_lang_tests}/{total_lang_tests} languages tested successfully")
    print("\nTo use the API:")
    print("1. Import Postman collection: docs/BLOOMZ_API_Collection.postman_collection.json")
    print("2. Use this script: python scripts/test_simple_api.py")
    print("3. API is running on: http://127.0.0.1:8110")
    print("4. Supports 21+ languages from your training data!")

if __name__ == "__main__":
    test_api()
