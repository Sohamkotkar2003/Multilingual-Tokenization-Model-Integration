#!/usr/bin/env python3
"""
Test script to verify all API endpoints are working correctly
"""

import requests
import json
import time
import sys

BASE_URL = "http://localhost:8000"

def test_endpoint(method, endpoint, data=None, description=""):
    """Test a single API endpoint"""
    try:
        url = f"{BASE_URL}{endpoint}"
        print(f"\n🧪 Testing {description}")
        print(f"   {method} {url}")
        
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        elif method == "DELETE":
            response = requests.delete(url, timeout=10)
        
        if response.status_code == 200:
            print(f"   ✅ SUCCESS (200)")
            try:
                result = response.json()
                if isinstance(result, dict) and len(str(result)) < 200:
                    print(f"   📄 Response: {result}")
                else:
                    print(f"   📄 Response: {type(result)} with {len(str(result))} chars")
            except:
                print(f"   📄 Response: {response.text[:100]}...")
            return True
        else:
            print(f"   ❌ FAILED ({response.status_code})")
            print(f"   📄 Error: {response.text[:200]}...")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"   ❌ CONNECTION ERROR - Server not running?")
        return False
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

def wait_for_server(max_wait=60):
    """Wait for server to be ready"""
    print("⏳ Waiting for server to start...")
    for i in range(max_wait):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return True
        except:
            pass
        time.sleep(1)
        if i % 10 == 0 and i > 0:
            print(f"   Still waiting... ({i}s)")
    
    print("❌ Server did not start within timeout")
    return False

def main():
    """Test all API endpoints"""
    print("🚀 Testing Multilingual Tokenization & Inference API")
    print("=" * 60)
    
    # Wait for server
    if not wait_for_server():
        print("\n❌ Cannot proceed without server. Please start the API first:")
        print("   python app.py")
        return False
    
    # Test results
    results = []
    
    # 1. Health check
    results.append(test_endpoint("GET", "/health", description="Health Check"))
    
    # 2. Root endpoint
    results.append(test_endpoint("GET", "/", description="Root Endpoint"))
    
    # 3. Configuration
    results.append(test_endpoint("GET", "/config", description="Configuration"))
    
    # 4. Language detection
    test_data = {"text": "नमस्ते, आप कैसे हैं?"}
    results.append(test_endpoint("POST", "/language-detect", test_data, "Language Detection"))
    
    # 5. Tokenization
    results.append(test_endpoint("POST", "/tokenize", test_data, "Tokenization"))
    
    # 6. Text generation
    results.append(test_endpoint("POST", "/generate", test_data, "Text Generation"))
    
    # 7. Knowledge Base Q&A
    kb_data = {"text": "What is the capital of India?", "language": "english"}
    results.append(test_endpoint("POST", "/qa", kb_data, "Knowledge Base Q&A"))
    
    # 8. Multilingual conversation
    conv_data = {
        "text": "मुझे हिंदी में जवाब दें", 
        "generate_response": True,
        "session_id": "test_session_001"
    }
    results.append(test_endpoint("POST", "/multilingual-conversation", conv_data, "Multilingual Conversation"))
    
    # 9. Language switching test
    switch_data = {"text": "Tell me about India", "session_id": "test_session_001"}
    results.append(test_endpoint("POST", "/test-language-switching", switch_data, "Language Switching Test"))
    
    # 10. Statistics
    results.append(test_endpoint("GET", "/stats", description="Statistics"))
    
    # 11. Conversation history
    results.append(test_endpoint("GET", "/conversation/test_session_001/history", description="Conversation History"))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The API is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
