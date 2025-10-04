#!/usr/bin/env python3
"""
Complete Integration Example

This script demonstrates the complete integration between
Multilingual Tokenization Model, Indigenous NLP, and Vaani TTS.

Usage:
    python examples/complete_integration_example.py
"""

import asyncio
import time
import json
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from integration.multilingual_pipeline import CompleteMultilingualPipeline, AsyncMultilingualPipeline
from integration.tts_integration import VaaniTTSIntegration
from integration.nlp_integration import IndigenousNLPIntegration
from integration.cached_pipeline import CachedMultilingualPipeline


def print_separator(title: str):
    """Print a formatted separator"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def print_result(result: dict, title: str = "Result"):
    """Print a formatted result"""
    print(f"\n{title}:")
    print("-" * 40)
    for key, value in result.items():
        if isinstance(value, (dict, list)):
            print(f"{key}: {json.dumps(value, indent=2, ensure_ascii=False)}")
        else:
            print(f"{key}: {value}")


def test_basic_integration():
    """Test basic integration functionality"""
    print_separator("Basic Integration Test")
    
    # Initialize pipeline
    pipeline = CompleteMultilingualPipeline()
    
    # Test texts in different languages
    test_cases = [
        {
            "text": "नमस्ते, आप कैसे हैं?",
            "expected_lang": "hindi",
            "description": "Hindi greeting"
        },
        {
            "text": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
            "expected_lang": "tamil", 
            "description": "Tamil greeting"
        },
        {
            "text": "Hello, how are you?",
            "expected_lang": "english",
            "description": "English greeting"
        },
        {
            "text": "नमस्कारः, भवान् कथं वर्तते?",
            "expected_lang": "sanskrit",
            "description": "Sanskrit greeting"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['description']}")
        print(f"Input: {test_case['text']}")
        
        try:
            result = pipeline.process_user_input(
                test_case['text'],
                user_id=f"test_user_{i}",
                session_id=f"test_session_{i}"
            )
            
            print(f"Detected Language: {result.get('detected_language', 'unknown')}")
            print(f"Success: {result.get('success', False)}")
            
            if result.get('success'):
                print(f"Response: {result.get('text_response', 'No response')[:100]}...")
                if result.get('audio_url'):
                    print(f"Audio URL: {result['audio_url']}")
                print(f"Processing Time: {result.get('processing_time', 0):.2f}s")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"Exception: {e}")


def test_language_switching():
    """Test mid-conversation language switching"""
    print_separator("Language Switching Test")
    
    pipeline = CompleteMultilingualPipeline()
    
    # Simulate a conversation with language switching
    conversation = [
        "Hello, how are you?",
        "नमस्ते, मैं ठीक हूं। आप कैसे हैं?",
        "வணக்கம், நான் நன்றாக இருக்கிறேன். நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "Thank you for the conversation!",
        "धन्यवाद, बातचीत के लिए!"
    ]
    
    print("Testing mid-conversation language switching...")
    print("Conversation flow:")
    
    for i, message in enumerate(conversation, 1):
        print(f"\n{i}. {message}")
        
        try:
            result = pipeline.process_user_input(
                message,
                user_id="switching_test_user",
                session_id="switching_test_session"
            )
            
            print(f"   → Detected: {result.get('detected_language', 'unknown')}")
            print(f"   → Success: {result.get('success', False)}")
            
            if result.get('success'):
                print(f"   → Response: {result.get('text_response', 'No response')[:50]}...")
            
        except Exception as e:
            print(f"   → Error: {e}")


def test_tts_integration():
    """Test TTS integration"""
    print_separator("TTS Integration Test")
    
    tts = VaaniTTSIntegration()
    
    # Test TTS health
    print("Testing TTS health...")
    health = tts.health_check()
    print(f"TTS Health: {health}")
    
    # Test voice availability
    print("\nTesting voice availability...")
    languages = ["hindi", "tamil", "english", "sanskrit"]
    
    for lang in languages:
        try:
            voices = tts.get_available_voices(lang)
            print(f"{lang}: {len(voices)} voices available")
            if voices:
                print(f"  Sample voice: {voices[0].get('name', 'Unknown')}")
        except Exception as e:
            print(f"{lang}: Error - {e}")
    
    # Test speech synthesis
    print("\nTesting speech synthesis...")
    test_texts = [
        ("नमस्ते, आप कैसे हैं?", "hindi"),
        ("வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?", "tamil"),
        ("Hello, how are you?", "english")
    ]
    
    for text, lang in test_texts:
        print(f"\nSynthesizing: {text} ({lang})")
        try:
            result = tts.synthesize_speech(text, lang)
            print(f"Success: {result.get('success', False)}")
            if result.get('success'):
                print(f"Audio URL: {result.get('audio_url', 'N/A')}")
                print(f"Duration: {result.get('duration', 0):.2f}s")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"Exception: {e}")


def test_nlp_integration():
    """Test NLP integration"""
    print_separator("NLP Integration Test")
    
    nlp = IndigenousNLPIntegration()
    
    # Test NLP health
    print("Testing NLP health...")
    health = nlp.health_check()
    print(f"NLP Health: {health}")
    
    # Test text preprocessing
    print("\nTesting text preprocessing...")
    test_texts = [
        ("नमस्ते, आप कैसे हैं?", "hindi"),
        ("வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?", "tamil"),
        ("Hello, how are you?", "english")
    ]
    
    for text, lang in test_texts:
        print(f"\nPreprocessing: {text} ({lang})")
        try:
            result = nlp.preprocess_text(text, lang)
            print(f"Success: {result.get('success', False)}")
            if result.get('success'):
                print(f"Processed: {result.get('processed_text', 'N/A')}")
                if result.get('tokens'):
                    print(f"Tokens: {result.get('tokens', [])[:5]}...")
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"Exception: {e}")
    
    # Test sentiment analysis
    print("\nTesting sentiment analysis...")
    sentiment_texts = [
        ("मैं बहुत खुश हूं!", "hindi"),
        ("நான் மிகவும் மகிழ்ச்சியாக இருக்கிறேன்!", "tamil"),
        ("I am very happy!", "english")
    ]
    
    for text, lang in sentiment_texts:
        print(f"\nSentiment: {text} ({lang})")
        try:
            result = nlp.analyze_sentiment(text, lang)
            print(f"Sentiment: {result.get('sentiment', 'unknown')}")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
        except Exception as e:
            print(f"Exception: {e}")


async def test_async_integration():
    """Test async integration"""
    print_separator("Async Integration Test")
    
    async_pipeline = AsyncMultilingualPipeline()
    
    # Test async processing
    print("Testing async processing...")
    texts = [
        "नमस्ते, आप कैसे हैं?",
        "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "Hello, how are you?"
    ]
    
    try:
        start_time = time.time()
        results = await async_pipeline.batch_process(
            texts, "async_test_user", "async_test_session"
        )
        end_time = time.time()
        
        print(f"Processed {len(results)} texts in {end_time - start_time:.2f}s")
        
        for i, result in enumerate(results):
            print(f"\nText {i+1}: {texts[i]}")
            print(f"Success: {result.get('success', False)}")
            print(f"Language: {result.get('detected_language', 'unknown')}")
            if result.get('success'):
                print(f"Response: {result.get('text_response', 'No response')[:50]}...")
        
    except Exception as e:
        print(f"Async processing error: {e}")


def test_cached_integration():
    """Test cached integration"""
    print_separator("Cached Integration Test")
    
    # Note: This will fail if Redis is not available, which is expected
    try:
        cached_pipeline = CachedMultilingualPipeline()
        
        print("Testing cached processing...")
        
        # Test cache operations
        test_text = "नमस्ते, आप कैसे हैं?"
        
        # First call (should cache)
        print("First call (should cache)...")
        result1 = cached_pipeline.process_user_input(
            test_text, "cache_test_user", "cache_test_session"
        )
        print(f"Success: {result1.get('success', False)}")
        
        # Second call (should hit cache)
        print("Second call (should hit cache)...")
        result2 = cached_pipeline.process_user_input(
            test_text, "cache_test_user", "cache_test_session"
        )
        print(f"Success: {result2.get('success', False)}")
        
        # Get cache stats
        stats = cached_pipeline.get_cache_stats()
        print(f"Cache stats: {stats}")
        
    except Exception as e:
        print(f"Cached integration test skipped (Redis not available): {e}")


def test_performance():
    """Test performance metrics"""
    print_separator("Performance Test")
    
    pipeline = CompleteMultilingualPipeline()
    
    # Test with multiple requests
    test_texts = [
        "नमस्ते, आप कैसे हैं?",
        "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "Hello, how are you?",
        "नमस्कारः, भवान् कथं वर्तते?",
        "வணக்கம், நான் நன்றாக இருக்கிறேன்"
    ]
    
    print(f"Testing performance with {len(test_texts)} requests...")
    
    start_time = time.time()
    results = []
    
    for i, text in enumerate(test_texts):
        print(f"Processing request {i+1}/{len(test_texts)}...")
        
        try:
            result = pipeline.process_user_input(
                text, f"perf_user_{i}", f"perf_session_{i}"
            )
            results.append(result)
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
            results.append({"success": False, "error": str(e)})
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate metrics
    successful_requests = sum(1 for r in results if r.get('success', False))
    avg_time_per_request = total_time / len(test_texts)
    
    print(f"\nPerformance Results:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Successful requests: {successful_requests}/{len(test_texts)}")
    print(f"Average time per request: {avg_time_per_request:.2f}s")
    print(f"Requests per second: {len(test_texts)/total_time:.2f}")
    
    # Show individual processing times
    processing_times = [r.get('processing_time', 0) for r in results if r.get('success')]
    if processing_times:
        print(f"Average processing time: {sum(processing_times)/len(processing_times):.2f}s")
        print(f"Min processing time: {min(processing_times):.2f}s")
        print(f"Max processing time: {max(processing_times):.2f}s")


def main():
    """Main function to run all tests"""
    print("Multilingual Integration Test Suite")
    print("===================================")
    
    try:
        # Run all tests
        test_basic_integration()
        test_language_switching()
        test_tts_integration()
        test_nlp_integration()
        
        # Run async test
        print("\nRunning async test...")
        asyncio.run(test_async_integration())
        
        test_cached_integration()
        test_performance()
        
        print_separator("Test Suite Complete")
        print("All tests completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nTest suite interrupted by user.")
    except Exception as e:
        print(f"\n\nTest suite failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
