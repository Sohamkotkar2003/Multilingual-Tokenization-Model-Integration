#!/usr/bin/env python3
"""
Complete System Demo for Multilingual Tokenization Model Integration v2.0

This script demonstrates the complete functionality of the updated system:
1. 20+ Indian language support
2. MCP preprocessing pipeline
3. Enhanced language detection
4. Multilingual tokenization and generation
5. Evaluation metrics
6. Integration readiness

Usage:
    python demo_complete_system.py
"""

import os
import sys
import time
import requests
import json
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def print_header(title: str):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{title}")
    print("-" * len(title))

def test_language_detection():
    """Test language detection for all 20+ languages"""
    print_section("Testing Language Detection for 20+ Languages")
    
    test_cases = [
        ("नमस्ते, आप कैसे हैं?", "hindi"),
        ("नमस्कारः, भवान् कथं वर्तते?", "sanskrit"),
        ("नमस्कार, तुम्ही कसे आहात?", "marathi"),
        ("வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?", "tamil"),
        ("నమస్కారం, మీరు ఎలా ఉన్నారు?", "telugu"),
        ("ನಮಸ್ಕಾರ, ನೀವು ಹೇಗಿದ್ದೀರಿ?", "kannada"),
        ("নমস্কার, আপনি কেমন আছেন?", "bengali"),
        ("નમસ્કાર, તમે કેમ છો?", "gujarati"),
        ("ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ?", "punjabi"),
        ("ନମସ୍କାର, ଆପଣ କିପରି ଅଛନ୍ତି?", "odia"),
        ("നമസ്കാരം, നിങ്ങൾ എങ്ങനെയുണ്ട്?", "malayalam"),
        ("নমস্কাৰ, আপুনি কেনেকৈ আছা?", "assamese"),
        ("السلام علیکم، آپ کیسے ہیں؟", "urdu"),
        ("नमस्कार, तपाईं कसरी हुनुहुन्छ?", "nepali"),
        ("Hello, how are you?", "english")
    ]
    
    api_base = "http://localhost:8000"
    
    for text, expected_lang in test_cases:
        try:
            response = requests.post(
                f"{api_base}/language-detect",
                json={"text": text},
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                detected_lang = result["language"]
                confidence = result["confidence"]
                
                status = "✓" if detected_lang == expected_lang else "✗"
                print(f"{status} {text[:30]}... → {detected_lang} (conf: {confidence:.2f}) [Expected: {expected_lang}]")
            else:
                print(f"✗ {text[:30]}... → Error: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"✗ {text[:30]}... → Connection Error: {e}")

def test_text_generation():
    """Test text generation for multiple languages"""
    print_section("Testing Text Generation")
    
    prompts = [
        ("Hello, how are you?", "english"),
        ("नमस्ते, आप कैसे हैं?", "hindi"),
        ("வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?", "tamil"),
        ("నమస్కారం, మీరు ఎలా ఉన్నారు?", "telugu"),
        ("নমস্কার, আপনি কেমন আছেন?", "bengali")
    ]
    
    api_base = "http://localhost:8000"
    
    for prompt, language in prompts:
        try:
            response = requests.post(
                f"{api_base}/generate",
                json={"text": prompt, "language": language},
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result["generated_text"]
                detected_lang = result["language"]
                
                print(f"\nLanguage: {language}")
                print(f"Prompt: {prompt}")
                print(f"Generated: {generated_text[:100]}...")
                print(f"Detected Language: {detected_lang}")
            else:
                print(f"✗ Error generating text for {language}: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"✗ Connection error for {language}: {e}")

def test_multilingual_conversation():
    """Test multilingual conversation with KB integration"""
    print_section("Testing Multilingual Conversation")
    
    conversations = [
        ("भारत के बारे में बताइए", "hindi"),
        ("Tell me about India", "english"),
        ("இந்தியா பற்றி சொல்லுங்கள்", "tamil"),
        ("భారతదేశం గురించి చెప్పండి", "telugu")
    ]
    
    api_base = "http://localhost:8000"
    session_id = f"demo_session_{int(time.time())}"
    
    for query, language in conversations:
        try:
            response = requests.post(
                f"{api_base}/multilingual-conversation",
                json={
                    "text": query,
                    "language": language,
                    "session_id": session_id,
                    "generate_response": True,
                    "max_response_length": 100
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"\nLanguage: {language}")
                print(f"Query: {query}")
                print(f"KB Answer: {result.get('kb_answer', 'N/A')[:100]}...")
                print(f"Generated Response: {result.get('generated_response', 'N/A')[:100]}...")
                print(f"Confidence: {result.get('confidence', 0):.2f}")
                print(f"Processing Time: {result.get('processing_time', 0):.2f}s")
            else:
                print(f"✗ Error in conversation for {language}: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"✗ Connection error for {language}: {e}")

def test_language_switching():
    """Test mid-conversation language switching"""
    print_section("Testing Language Switching")
    
    api_base = "http://localhost:8000"
    
    try:
        response = requests.post(
            f"{api_base}/test-language-switching",
            json={
                "text": "Hello, how are you?",
                "session_id": "switching_test"
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"Test Type: {result.get('test_type', 'N/A')}")
            print(f"Session ID: {result.get('session_id', 'N/A')}")
            print(f"Switching Successful: {result.get('switching_successful', False)}")
            
            print("\nResults:")
            for i, test_result in enumerate(result.get('results', []), 1):
                print(f"\nTest {i}:")
                print(f"  Query: {test_result.get('query', 'N/A')}")
                print(f"  Expected Language: {test_result.get('expected_language', 'N/A')}")
                print(f"  Detected Language: {test_result.get('detected_language', 'N/A')}")
                print(f"  Answer: {test_result.get('answer', 'N/A')[:50]}...")
                print(f"  Confidence: {test_result.get('confidence', 0):.2f}")
        else:
            print(f"✗ Error in language switching test: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"✗ Connection error in language switching test: {e}")

def test_api_health():
    """Test API health and statistics"""
    print_section("Testing API Health and Statistics")
    
    api_base = "http://localhost:8000"
    
    # Test basic health
    try:
        response = requests.get(f"{api_base}/", timeout=120)
        if response.status_code == 200:
            result = response.json()
            print("✓ Basic Health Check:")
            print(f"  Status: {result.get('status', 'N/A')}")
            print(f"  Tokenizer Loaded: {result.get('tokenizer_loaded', False)}")
            print(f"  Model Loaded: {result.get('model_loaded', False)}")
            print(f"  API Version: {result.get('api_version', 'N/A')}")
            print(f"  Supported Languages: {len(result.get('supported_languages', []))}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Health check connection error: {e}")
    
    # Test detailed health
    try:
        response = requests.get(f"{api_base}/health", timeout=120)
        if response.status_code == 200:
            result = response.json()
            print("\n✓ Detailed Health Check:")
            print(f"  Status: {result.get('status', 'N/A')}")
            
            components = result.get('components', {})
            for component, status in components.items():
                print(f"  {component}: {status.get('status', 'N/A')}")
        else:
            print(f"✗ Detailed health check failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Detailed health check connection error: {e}")
    
    # Test statistics
    try:
        response = requests.get(f"{api_base}/stats", timeout=120)
        if response.status_code == 200:
            result = response.json()
            print("\n✓ API Statistics:")
            print(f"  Status: {result.get('status', 'N/A')}")
            print(f"  API Version: {result.get('api_version', 'N/A')}")
            print(f"  Supported Languages: {len(result.get('supported_languages', []))}")
            
            kb_integration = result.get('kb_integration', {})
            if kb_integration:
                print(f"  KB Integration: {kb_integration.get('status', 'N/A')}")
        else:
            print(f"✗ Statistics check failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"✗ Statistics check connection error: {e}")

def run_evaluation_demo():
    """Run evaluation metrics demo"""
    print_section("Running Evaluation Metrics Demo")
    
    try:
        # This would run the actual evaluation script
        print("To run full evaluation metrics:")
        print("python src/evaluation/metrics.py --languages hindi tamil telugu")
        print("\nThis will evaluate:")
        print("- BLEU/ROUGE scores")
        print("- Perplexity metrics")
        print("- Tokenization accuracy")
        print("- Fluency scores")
        print("- Latency metrics")
        print("- Language switching success")
    except Exception as e:
        print(f"Evaluation demo error: {e}")

def main():
    """Main demo function"""
    print_header("Multilingual Tokenization Model Integration v2.0 - Complete System Demo")
    
    print("This demo showcases the complete functionality of the updated system:")
    print("• 20+ Indian language support")
    print("• MCP preprocessing pipeline")
    print("• Enhanced language detection")
    print("• Multilingual tokenization and generation")
    print("• Knowledge Base integration")
    print("• Language switching capabilities")
    print("• Evaluation metrics")
    print("• Integration readiness")
    
    print("\nNote: Make sure the API is running on http://localhost:8000")
    print("Start the API with: python main.py")
    
    input("\nPress Enter to continue with the demo...")
    
    # Run all tests
    test_api_health()
    test_language_detection()
    test_text_generation()
    test_multilingual_conversation()
    test_language_switching()
    run_evaluation_demo()
    
    print_header("Demo Complete!")
    
    print("\nNext Steps:")
    print("1. Review the integration guide: docs/INTEGRATION_GUIDE.md")
    print("2. Run full evaluation: python src/evaluation/metrics.py")
    print("3. Deploy with Docker: docker-compose up --build")
    print("4. Integrate with Vaani TTS and Indigenous NLP")
    
    print("\nFor more information:")
    print("• API Documentation: http://localhost:8000/docs")
    print("• Integration Guide: docs/INTEGRATION_GUIDE.md")
    print("• README: README.md")

if __name__ == "__main__":
    main()
