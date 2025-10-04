"""
Integration Tests for Multilingual Pipeline

This module contains comprehensive tests for the integration
between Multilingual Tokenization Model, Indigenous NLP, and Vaani TTS.
"""

import unittest
import asyncio
import requests
import time
from unittest.mock import Mock, patch
import json

# Import integration modules
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.integration.multilingual_pipeline import CompleteMultilingualPipeline, AsyncMultilingualPipeline
from src.integration.tts_integration import VaaniTTSIntegration
from src.integration.nlp_integration import IndigenousNLPIntegration
from src.integration.cached_pipeline import CachedMultilingualPipeline


class TestMultilingualPipelineIntegration(unittest.TestCase):
    """Test cases for the complete multilingual pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pipeline = CompleteMultilingualPipeline()
        self.test_texts = {
            "hindi": "नमस्ते, आप कैसे हैं?",
            "tamil": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
            "english": "Hello, how are you?",
            "sanskrit": "नमस्कारः, भवान् कथं वर्तते?"
        }
    
    def test_language_detection(self):
        """Test language detection accuracy"""
        for expected_lang, text in self.test_texts.items():
            with self.subTest(language=expected_lang):
                result = self.pipeline.detect_language(text)
                
                self.assertIn('language', result)
                self.assertIn('confidence', result)
                self.assertGreater(result['confidence'], 0.0)
    
    @patch('requests.post')
    def test_process_user_input_success(self, mock_post):
        """Test successful user input processing"""
        # Mock API responses
        mock_responses = [
            # Language detection response
            Mock(json=lambda: {"language": "hindi", "confidence": 0.9}),
            # NLP preprocessing response
            Mock(json=lambda: {"processed_text": "नमस्ते, आप कैसे हैं?", "success": True}),
            # KB/Generation response
            Mock(json=lambda: {
                "generated_response": "मैं ठीक हूं, धन्यवाद!",
                "kb_answer": "मैं ठीक हूं, धन्यवाद!",
                "processing_time": 1.5
            }),
            # TTS response
            Mock(json=lambda: {
                "audio_url": "http://example.com/audio.wav",
                "duration": 2.5,
                "success": True
            })
        ]
        
        mock_post.side_effect = mock_responses
        
        result = self.pipeline.process_user_input(
            "नमस्ते, आप कैसे हैं?",
            user_id="test_user",
            session_id="test_session"
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(result['detected_language'], 'hindi')
        self.assertIn('text_response', result)
        self.assertIn('audio_url', result)
        self.assertIn('processing_time', result)
    
    def test_handle_conversation(self):
        """Test multi-turn conversation handling"""
        messages = [
            {"text": "नमस्ते"},
            {"text": "आप कैसे हैं?"},
            {"text": "धन्यवाद"}
        ]
        
        with patch.object(self.pipeline, 'process_user_input') as mock_process:
            mock_process.return_value = {
                "success": True,
                "detected_language": "hindi",
                "text_response": "Response",
                "audio_url": "http://example.com/audio.wav"
            }
            
            results = self.pipeline.handle_conversation(
                messages, "test_user", "test_session"
            )
            
            self.assertEqual(len(results), 3)
            self.assertTrue(all(r['success'] for r in results))
    
    def test_language_switching(self):
        """Test mid-conversation language switching"""
        texts = [
            "Hello, how are you?",
            "नमस्ते, आप कैसे हैं?",
            "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"
        ]
        
        with patch.object(self.pipeline, 'process_user_input') as mock_process:
            mock_responses = [
                {"success": True, "detected_language": "english"},
                {"success": True, "detected_language": "hindi"},
                {"success": True, "detected_language": "tamil"}
            ]
            mock_process.side_effect = mock_responses
            
            result = self.pipeline.test_language_switching(
                texts, "test_user", "test_session"
            )
            
            self.assertTrue(result['switching_successful'])
            self.assertEqual(len(result['detected_languages']), 3)
            self.assertIn('english', result['detected_languages'])
            self.assertIn('hindi', result['detected_languages'])
            self.assertIn('tamil', result['detected_languages'])


class TestTTSIntegration(unittest.TestCase):
    """Test cases for Vaani TTS integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.tts = VaaniTTSIntegration()
    
    @patch('requests.get')
    def test_health_check(self, mock_get):
        """Test TTS health check"""
        mock_get.return_value.json.return_value = {"status": "healthy"}
        mock_get.return_value.raise_for_status.return_value = None
        
        result = self.tts.health_check()
        
        self.assertEqual(result['status'], 'healthy')
    
    @patch('requests.get')
    def test_get_available_voices(self, mock_get):
        """Test getting available voices"""
        mock_voices = [
            {"id": "voice1", "name": "Hindi Female", "language": "hindi"},
            {"id": "voice2", "name": "Hindi Male", "language": "hindi"}
        ]
        mock_get.return_value.json.return_value = mock_voices
        mock_get.return_value.raise_for_status.return_value = None
        
        voices = self.tts.get_available_voices("hindi")
        
        self.assertEqual(len(voices), 2)
        self.assertEqual(voices[0]['language'], 'hindi')
    
    @patch('requests.post')
    def test_synthesize_speech(self, mock_post):
        """Test speech synthesis"""
        mock_response = {
            "audio_url": "http://example.com/audio.wav",
            "duration": 2.5,
            "success": True
        }
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status.return_value = None
        
        result = self.tts.synthesize_speech("नमस्ते", "hindi")
        
        self.assertTrue(result['success'])
        self.assertIn('audio_url', result)
        self.assertIn('duration', result)
    
    def test_quality_score_calculation(self):
        """Test voice quality score calculation"""
        # Test with good result
        good_result = {
            "success": True,
            "duration": 2.5,
            "warnings": []
        }
        score = self.tts._calculate_quality_score(good_result)
        self.assertGreater(score, 0.8)
        
        # Test with poor result
        poor_result = {
            "success": False,
            "duration": 0.1,
            "warnings": ["Low quality audio"]
        }
        score = self.tts._calculate_quality_score(poor_result)
        self.assertEqual(score, 0.0)


class TestNLPIntegration(unittest.TestCase):
    """Test cases for Indigenous NLP integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.nlp = IndigenousNLPIntegration()
    
    @patch('requests.post')
    def test_preprocess_text(self, mock_post):
        """Test text preprocessing"""
        mock_response = {
            "processed_text": "नमस्ते, आप कैसे हैं?",
            "tokens": ["नमस्ते", ",", "आप", "कैसे", "हैं", "?"],
            "success": True
        }
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status.return_value = None
        
        result = self.nlp.preprocess_text("नमस्ते, आप कैसे हैं?", "hindi")
        
        self.assertTrue(result['success'])
        self.assertIn('processed_text', result)
        self.assertIn('tokens', result)
    
    @patch('requests.post')
    def test_analyze_sentiment(self, mock_post):
        """Test sentiment analysis"""
        mock_response = {
            "sentiment": "positive",
            "confidence": 0.8,
            "success": True
        }
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status.return_value = None
        
        result = self.nlp.analyze_sentiment("मैं खुश हूं", "hindi")
        
        self.assertTrue(result['success'])
        self.assertEqual(result['sentiment'], 'positive')
        self.assertGreater(result['confidence'], 0.0)
    
    @patch('requests.post')
    def test_extract_entities(self, mock_post):
        """Test entity extraction"""
        mock_response = {
            "entities": [
                {"text": "भारत", "type": "LOCATION", "confidence": 0.9},
                {"text": "दिल्ली", "type": "LOCATION", "confidence": 0.8}
            ],
            "success": True
        }
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.raise_for_status.return_value = None
        
        result = self.nlp.extract_entities("भारत की राजधानी दिल्ली है", "hindi")
        
        self.assertTrue(result['success'])
        self.assertEqual(len(result['entities']), 2)
        self.assertEqual(result['entities'][0]['type'], 'LOCATION')


class TestCachedPipeline(unittest.TestCase):
    """Test cases for cached multilingual pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock Redis client
        self.mock_redis = Mock()
        self.pipeline = CachedMultilingualPipeline()
        self.pipeline.redis_client = self.mock_redis
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        key = self.pipeline._get_cache_key(
            "test_operation", "test_text", "user123", "session456"
        )
        
        self.assertTrue(key.startswith("multilingual_pipeline:test_operation:"))
        self.assertIn("test_text", key)
    
    def test_cache_set_and_get(self):
        """Test cache set and get operations"""
        test_data = {"test": "data", "success": True}
        cache_key = "test_key"
        
        # Test cache set
        self.pipeline._set_cache(cache_key, test_data)
        self.mock_redis.setex.assert_called_once()
        
        # Test cache get
        self.mock_redis.get.return_value = json.dumps(test_data)
        result = self.pipeline._get_from_cache(cache_key)
        
        self.assertEqual(result, test_data)
    
    @patch.object(CachedMultilingualPipeline, 'detect_language')
    def test_cached_language_detection(self, mock_detect):
        """Test cached language detection"""
        mock_detect.return_value = {"language": "hindi", "confidence": 0.9}
        
        # First call should hit the actual method
        result1 = self.pipeline.detect_language("नमस्ते")
        self.mock_redis.get.assert_called()
        self.mock_redis.setex.assert_called()
        
        # Reset mocks
        self.mock_redis.reset_mock()
        
        # Second call should hit cache
        self.mock_redis.get.return_value = json.dumps({"language": "hindi", "confidence": 0.9})
        result2 = self.pipeline.detect_language("नमस्ते")
        
        # Should not call the actual method
        mock_detect.assert_called_once()  # Only called once (first time)
        self.mock_redis.get.assert_called()


class TestAsyncPipeline(unittest.TestCase):
    """Test cases for async multilingual pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.async_pipeline = AsyncMultilingualPipeline()
    
    @patch.object(CompleteMultilingualPipeline, 'process_user_input')
    def test_async_process_user_input(self, mock_process):
        """Test async user input processing"""
        mock_process.return_value = {
            "success": True,
            "detected_language": "hindi",
            "text_response": "Response"
        }
        
        async def run_test():
            result = await self.async_pipeline.process_user_input_async(
                "नमस्ते", "user123", "session456"
            )
            return result
        
        result = asyncio.run(run_test())
        
        self.assertTrue(result['success'])
        self.assertEqual(result['detected_language'], 'hindi')
    
    @patch.object(CompleteMultilingualPipeline, 'process_user_input')
    def test_batch_process(self, mock_process):
        """Test batch processing"""
        mock_process.return_value = {
            "success": True,
            "detected_language": "hindi",
            "text_response": "Response"
        }
        
        texts = ["नमस्ते", "आप कैसे हैं?", "धन्यवाद"]
        
        async def run_test():
            results = await self.async_pipeline.batch_process(
                texts, "user123", "session456"
            )
            return results
        
        results = asyncio.run(run_test())
        
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r['success'] for r in results))


class TestIntegrationEndToEnd(unittest.TestCase):
    """End-to-end integration tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pipeline = CompleteMultilingualPipeline()
    
    @patch('requests.post')
    @patch('requests.get')
    def test_complete_pipeline_flow(self, mock_get, mock_post):
        """Test complete pipeline flow with all services"""
        # Mock all API responses
        mock_responses = [
            # Language detection
            Mock(json=lambda: {"language": "hindi", "confidence": 0.9}),
            # NLP preprocessing
            Mock(json=lambda: {"processed_text": "नमस्ते, आप कैसे हैं?", "success": True}),
            # KB/Generation
            Mock(json=lambda: {
                "generated_response": "मैं ठीक हूं, धन्यवाद!",
                "processing_time": 1.5
            }),
            # TTS synthesis
            Mock(json=lambda: {
                "audio_url": "http://example.com/audio.wav",
                "duration": 2.5,
                "success": True
            })
        ]
        
        mock_post.side_effect = mock_responses
        
        result = self.pipeline.process_user_input(
            "नमस्ते, आप कैसे हैं?",
            user_id="test_user",
            session_id="test_session"
        )
        
        # Verify result structure
        self.assertTrue(result['success'])
        self.assertEqual(result['detected_language'], 'hindi')
        self.assertIn('text_response', result)
        self.assertIn('audio_url', result)
        self.assertIn('processing_time', result)
        self.assertEqual(result['user_id'], 'test_user')
        self.assertEqual(result['session_id'], 'test_session')
    
    def test_error_handling(self):
        """Test error handling in pipeline"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = requests.exceptions.ConnectionError("Service unavailable")
            
            result = self.pipeline.process_user_input("Test text")
            
            self.assertFalse(result['success'])
            self.assertIn('error', result)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
