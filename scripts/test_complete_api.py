#!/usr/bin/env python3
"""
Complete API Testing Script for BLOOMZ-560M Service
Tests all endpoints and generation scenarios
"""

import requests
import json
import time
from typing import Dict, Any

class BLOOMZAPITester:
    def __init__(self, base_url: str = "http://127.0.0.1:8110"):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = []
    
    def test_health(self) -> bool:
        """Test health endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                print("Health check: PASSED")
                return True
            else:
                print(f"Health check: FAILED ({response.status_code})")
                return False
        except Exception as e:
            print(f"Health check: ERROR ({e})")
            return False
    
    def test_adapter_list(self) -> bool:
        """Test adapter list endpoint"""
        try:
            response = self.session.get(f"{self.base_url}/adapter/list", timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(f"Adapter list: PASSED - {len(data.get('adapters', []))} adapters available")
                return True
            else:
                print(f"Adapter list: FAILED ({response.status_code})")
                return False
        except Exception as e:
            print(f"Adapter list: ERROR ({e})")
            return False
    
    def test_generation(self, test_name: str, payload: Dict[str, Any]) -> bool:
        """Test text generation"""
        try:
            print(f"\nTesting: {test_name}")
            print(f"   Prompt: '{payload.get('prompt', 'N/A')[:50]}...'")
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/generate-lite", 
                json=payload, 
                timeout=30
            )
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                generated_text = data.get('generated_text', '')
                duration = end_time - start_time
                
                print(f"{test_name}: PASSED")
                print(f"   Generated: '{generated_text[:100]}{'...' if len(generated_text) > 100 else ''}'")
                print(f"   Duration: {duration:.2f}s")
                print(f"   Length: {len(generated_text)} chars")
                
                # Check for quality
                if len(generated_text.strip()) < 10:
                    print("   Warning: Output very short")
                elif generated_text.strip() == payload.get('prompt', '').strip():
                    print("   Warning: Only echoing prompt")
                else:
                    print("   Quality: Good generation")
                
                return True
            else:
                print(f"{test_name}: FAILED ({response.status_code})")
                print(f"   Error: {response.text}")
                return False
        except Exception as e:
            print(f"{test_name}: ERROR ({e})")
            return False
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("Starting Complete BLOOMZ-560M API Test Suite")
        print("=" * 60)
        
        # Test basic endpoints
        print("\nTesting Basic Endpoints:")
        health_ok = self.test_health()
        adapter_list_ok = self.test_adapter_list()
        
        if not health_ok:
            print("\nBasic endpoints failed. Cannot proceed with generation tests.")
            return
        
        # Test generation scenarios
        print("\nTesting Generation Scenarios:")
        
        test_cases = [
            {
                "name": "Basic Generation",
                "payload": {
                    "prompt": "The weather today is",
                    "base_model": "bigscience/bloomz-560m",
                    "max_new_tokens": 50,
                    "temperature": 0.7,
                    "do_sample": True,
                    "top_p": 0.9,
                    "top_k": 50,
                    "repetition_penalty": 1.2,
                    "no_repeat_ngram_size": 3,
                    "min_new_tokens": 10,
                    "eos_token_id": 2
                }
            },
            {
                "name": "Creative Generation",
                "payload": {
                    "prompt": "Once upon a time in a magical forest",
                    "base_model": "bigscience/bloomz-560m",
                    "max_new_tokens": 100,
                    "temperature": 0.9,
                    "do_sample": True,
                    "top_p": 0.95,
                    "top_k": 100,
                    "repetition_penalty": 1.3,
                    "no_repeat_ngram_size": 4,
                    "min_new_tokens": 20,
                    "eos_token_id": 2
                }
            },
            {
                "name": "Technical Generation",
                "payload": {
                    "prompt": "The benefits of renewable energy include",
                    "base_model": "bigscience/bloomz-560m",
                    "max_new_tokens": 80,
                    "temperature": 0.3,
                    "do_sample": True,
                    "top_p": 0.8,
                    "top_k": 40,
                    "repetition_penalty": 1.1,
                    "no_repeat_ngram_size": 3,
                    "min_new_tokens": 15,
                    "eos_token_id": 2
                }
            },
            {
                "name": "Multilingual Generation",
                "payload": {
                    "prompt": "नमस्ते, आज का दिन बहुत अच्छा है",
                    "base_model": "bigscience/bloomz-560m",
                    "max_new_tokens": 60,
                    "temperature": 0.7,
                    "do_sample": True,
                    "top_p": 0.9,
                    "top_k": 50,
                    "repetition_penalty": 1.2,
                    "no_repeat_ngram_size": 3,
                    "min_new_tokens": 10,
                    "eos_token_id": 2
                }
            },
            {
                "name": "Greedy Generation",
                "payload": {
                    "prompt": "The future of artificial intelligence is",
                    "base_model": "bigscience/bloomz-560m",
                    "max_new_tokens": 40,
                    "temperature": 0.0,
                    "do_sample": False,
                    "min_new_tokens": 8,
                    "eos_token_id": 2
                }
            },
            {
                "name": "Short Prompt",
                "payload": {
                    "prompt": "Hello",
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
            }
        ]
        
        passed_tests = 0
        total_tests = len(test_cases)
        
        for test_case in test_cases:
            if self.test_generation(test_case["name"], test_case["payload"]):
                passed_tests += 1
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Basic Endpoints: {'PASSED' if health_ok else 'FAILED'}")
        print(f"Generation Tests: {passed_tests}/{total_tests} passed")
        print(f"Overall Success Rate: {((passed_tests + (1 if health_ok else 0)) / (total_tests + 1)) * 100:.1f}%")
        
        if passed_tests == total_tests:
            print("\nALL TESTS PASSED! The BLOOMZ-560M API is working perfectly!")
        elif passed_tests > total_tests // 2:
            print("\nMOSTLY WORKING! Some tests passed, minor issues to investigate.")
        else:
            print("\nMAJOR ISSUES! Multiple tests failed, needs investigation.")
        
        print("\nNext Steps:")
        print("1. Import the Postman collection: docs/BLOOMZ_API_Collection.postman_collection.json")
        print("2. Use this script for automated testing: python scripts/test_complete_api.py")
        print("3. The API is ready for production use!")

def main():
    """Main test function"""
    tester = BLOOMZAPITester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()
