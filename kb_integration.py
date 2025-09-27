"""
Knowledge Base Integration Module

This module handles integration with the Black Hole DB and provides
a complete Q&A pipeline: User → Multilingual LM → KB → LM → Response

Features:
- Async HTTP client for KB communication
- Query preprocessing and response post-processing
- Multilingual query handling
- Error handling and fallback responses
- Caching for frequent queries
"""

import asyncio
import aiohttp
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
from core import settings

logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries supported by the KB"""
    FACTUAL = "factual"
    CONVERSATIONAL = "conversational"
    EDUCATIONAL = "educational"
    CULTURAL = "cultural"
    TRANSLATION = "translation"

@dataclass
class KBQuery:
    """Knowledge Base query structure"""
    text: str
    language: str
    query_type: QueryType
    context: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class KBResponse:
    """Knowledge Base response structure"""
    answer: str
    confidence: float
    sources: List[str]
    language: str
    query_type: QueryType
    processing_time: float
    cached: bool = False

class KBIntegrationError(Exception):
    """Custom exception for KB integration errors"""
    pass

class KnowledgeBaseClient:
    """Async client for Knowledge Base operations"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.cache: Dict[str, Dict] = {}  # Simple in-memory cache
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.max_cache_size = 1000
        
        # KB endpoints from settings
        self.kb_endpoint = settings.KB_ENDPOINT
        self.kb_timeout = settings.KB_TIMEOUT
        
        # Stats
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "average_response_time": 0.0
        }

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.kb_timeout),
            headers={"Content-Type": "application/json"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def _get_cache_key(self, query: KBQuery) -> str:
        """Generate cache key for query"""
        key_string = f"{query.text}_{query.language}_{query.query_type.value}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid"""
        return time.time() - cache_entry["timestamp"] < self.cache_ttl

    def _clean_cache(self):
        """Remove old cache entries"""
        current_time = time.time()
        keys_to_remove = []
        
        for key, entry in self.cache.items():
            if current_time - entry["timestamp"] > self.cache_ttl:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cache[key]
        
        # Limit cache size
        if len(self.cache) > self.max_cache_size:
            # Remove oldest entries
            sorted_items = sorted(self.cache.items(), key=lambda x: x[1]["timestamp"])
            excess_count = len(self.cache) - self.max_cache_size
            for key, _ in sorted_items[:excess_count]:
                del self.cache[key]

    async def query_knowledge_base(self, query: KBQuery) -> KBResponse:
        """
        Query the knowledge base with multilingual support
        
        Args:
            query: KBQuery object with query details
            
        Returns:
            KBResponse object with answer and metadata
        """
        start_time = time.time()
        self.stats["total_queries"] += 1
        
        # Check cache first
        cache_key = self._get_cache_key(query)
        if cache_key in self.cache and self._is_cache_valid(self.cache[cache_key]):
            self.stats["cache_hits"] += 1
            cached_response = self.cache[cache_key]["response"]
            cached_response.cached = True
            cached_response.processing_time = time.time() - start_time
            logger.debug(f"Cache hit for query: {query.text[:50]}...")
            return cached_response
        
        self.stats["cache_misses"] += 1
        
        try:
            # If no KB endpoint configured, use mock response
            if not self.kb_endpoint:
                logger.warning("No KB endpoint configured, using mock response")
                return await self._get_mock_response(query, start_time)
            
            # Prepare request payload for multilingual-conversation endpoint
            payload = {
                "text": query.text,
                "language": query.language,
                "user_id": query.user_id,
                "session_id": query.session_id,
                "generate_response": True,  # Enable response generation
                "max_response_length": 256
            }
            
            # Make request to multilingual-conversation endpoint
            if not self.session:
                raise KBIntegrationError("HTTP session not initialized")
            
            async with self.session.post(
                f"{self.kb_endpoint}/multilingual-conversation",
                json=payload
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    kb_response = self._parse_kb_response(data, query, start_time)
                    
                    # Cache the response
                    self.cache[cache_key] = {
                        "response": kb_response,
                        "timestamp": time.time()
                    }
                    self._clean_cache()
                    
                    return kb_response
                    
                else:
                    error_text = await response.text()
                    logger.error(f"KB API error {response.status}: {error_text}")
                    raise KBIntegrationError(f"KB API returned status {response.status}")
        
        except asyncio.TimeoutError:
            logger.error("KB request timed out")
            self.stats["errors"] += 1
            return await self._get_fallback_response(query, start_time, "timeout")
            
        except Exception as e:
            logger.error(f"KB integration error: {e}")
            self.stats["errors"] += 1
            return await self._get_fallback_response(query, start_time, str(e))

    def _parse_kb_response(self, data: Dict, query: KBQuery, start_time: float) -> KBResponse:
        """Parse response from multilingual-conversation endpoint"""
        processing_time = time.time() - start_time
        
        # Extract the KB answer (primary response from multilingual-conversation)
        kb_answer = data.get("kb_answer", data.get("answer", "No answer available"))
        
        # If there's a generated response, prefer it over KB answer
        generated_response = data.get("generated_response")
        if generated_response:
            kb_answer = generated_response
        
        # Extract metadata
        metadata = data.get("metadata", {})
        
        return KBResponse(
            answer=kb_answer,
            confidence=data.get("confidence", metadata.get("kb_confidence", 0.8)),
            sources=data.get("sources", metadata.get("kb_sources", ["Multilingual Conversation API"])),
            language=data.get("language", query.language),
            query_type=query.query_type,
            processing_time=processing_time
        )

    async def _get_mock_response(self, query: KBQuery, start_time: float) -> KBResponse:
        """Generate mock response for testing when KB is not available"""
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Generate contextual mock responses based on query content
        mock_responses = {
            "hindi": {
                "greeting": "नमस्ते! मैं आपकी सहायता के लिए यहां हूं।",
                "general": f"आपके प्रश्न '{query.text}' के बारे में मुझे खुशी होगी कि मैं आपकी सहायता कर सकूं।",
                "educational": "यह एक शैक्षिक प्रश्न है। मुझे अधिक जानकारी चाहिए।"
            },
            "sanskrit": {
                "greeting": "नमस्कारः! अहं भवतः सहायतार्थं अत्र अस्मि।",
                "general": f"भवतः प्रश्नस्य '{query.text}' विषये अहं सहायतां कर्तुं शक्नोमि।",
                "educational": "एषा शैक्षिका प्रश्ना अस्ति। मह्यं अधिकं ज्ञानं आवश्यकम्।"
            },
            "marathi": {
                "greeting": "नमस्कार! मी तुमची मदत करण्यासाठी येथे आहे।",
                "general": f"तुमच्या प्रश्नाबद्दल '{query.text}' मला तुमची मदत करायला आवडेल।",
                "educational": "हा एक शैक्षणिक प्रश्न आहे. मला अधिक माहिती हवी आहे."
            },
            "english": {
                "greeting": "Hello! I'm here to help you with your questions.",
                "general": f"Regarding your question about '{query.text}', I'd be happy to assist you.",
                "educational": "This appears to be an educational question. I'd need more information to provide a comprehensive answer."
            }
        }
        
        # Determine response type
        if any(word in query.text.lower() for word in ["hello", "hi", "namaste", "नमस्ते", "नमस्कार"]):
            response_key = "greeting"
        elif query.query_type == QueryType.EDUCATIONAL:
            response_key = "educational"
        else:
            response_key = "general"
        
        lang_responses = mock_responses.get(query.language, mock_responses["english"])
        mock_answer = lang_responses[response_key]
        
        return KBResponse(
            answer=mock_answer,
            confidence=0.8,
            sources=["Mock Knowledge Base"],
            language=query.language,
            query_type=query.query_type,
            processing_time=time.time() - start_time
        )

    async def _get_fallback_response(self, query: KBQuery, start_time: float, error_reason: str) -> KBResponse:
        """Generate fallback response when KB is unavailable"""
        
        fallback_messages = {
            "hindi": "क्षमा करें, मैं अभी आपके प्रश्न का उत्तर नहीं दे सकता। कृपया बाद में कोशिश करें।",
            "sanskrit": "क्षम्यताम्, अहं सम्प्रति भवतः प्रश्नस्य उत्तरं दातुं न शक्नोमि। कृपया पुनः प्रयासं कुर्वन्तु।",
            "marathi": "माफ करा, मी सध्या तुमच्या प्रश्नाचे उत्तर देऊ शकत नाही. कृपया नंतर प्रयत्न करा.",
            "english": "I apologize, but I'm unable to answer your question right now. Please try again later."
        }
        
        fallback_answer = fallback_messages.get(query.language, fallback_messages["english"])
        
        return KBResponse(
            answer=fallback_answer,
            confidence=0.1,
            sources=["Fallback System"],
            language=query.language,
            query_type=query.query_type,
            processing_time=time.time() - start_time
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get integration statistics"""
        cache_hit_rate = (self.stats["cache_hits"] / max(self.stats["total_queries"], 1)) * 100
        
        return {
            "total_queries": self.stats["total_queries"],
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "errors": self.stats["errors"],
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.cache),
            "error_rate": (self.stats["errors"] / max(self.stats["total_queries"], 1)) * 100
        }

class QueryClassifier:
    """Classify queries to determine appropriate KB routing"""
    
    def __init__(self):
        self.query_patterns = {
            QueryType.FACTUAL: [
                "what is", "who is", "when", "where", "how many",
                "क्या है", "कौन है", "कब", "कहाँ", "कितने",
                "किम् अस्ति", "कः अस्ति", "कदा", "कुत्र",
                "काय आहे", "कोण आहे", "केव्हा", "कुठे"
            ],
            QueryType.CONVERSATIONAL: [
                "hello", "hi", "how are you", "good morning",
                "नमस्ते", "हैलो", "कैसे हैं", "सुप्रभात",
                "नमस्कार", "कथम्", "सुप्रभातम्",
                "नमस्कार", "कसे आहात", "सुप्रभात"
            ],
            QueryType.EDUCATIONAL: [
                "explain", "teach", "learn", "study", "definition",
                "समझाओ", "सिखाओ", "सीखना", "अध्ययन", "परिभाषा",
                "व्याख्या", "शिक्षा", "अध्ययनम्", "ज्ञानम्",
                "समजावून सांगा", "शिकवा", "शिकणे", "अभ्यास"
            ],
            QueryType.CULTURAL: [
                "tradition", "festival", "culture", "history", "mythology",
                "परंपरा", "त्योहार", "संस्कृति", "इतिहास", "पुराण",
                "संस्कृतिः", "उत्सवः", "इतिहासः", "पुराणम्",
                "परंपरा", "सण", "संस्कृती", "इतिहास"
            ]
        }

    def classify_query(self, text: str, language: str) -> QueryType:
        """
        Classify the query type based on content patterns
        
        Args:
            text: Query text
            language: Detected language
            
        Returns:
            QueryType enum value
        """
        text_lower = text.lower()
        
        # Score each query type
        scores = {query_type: 0 for query_type in QueryType}
        
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    scores[query_type] += 1
        
        # Return the highest scoring type, default to CONVERSATIONAL
        if max(scores.values()) == 0:
            return QueryType.CONVERSATIONAL
        
        return max(scores, key=scores.get)

class MultilingualQAOrchestrator:
    """
    Main orchestrator for the multilingual Q&A pipeline
    Handles: User → Multilingual LM → KB → LM → Response
    """
    
    def __init__(self):
        self.kb_client = None
        self.classifier = QueryClassifier()
        self.conversation_history = {}  # Simple conversation tracking
        
    async def process_multilingual_query(
        self, 
        query_text: str, 
        detected_language: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Process a multilingual query through the complete pipeline
        
        Args:
            query_text: User's query
            detected_language: Language detected by the API
            user_id: Optional user identifier
            session_id: Optional session identifier
            context: Optional additional context
            
        Returns:
            Tuple of (final_response, metadata)
        """
        
        start_time = time.time()
        
        # Step 1: Classify the query
        query_type = self.classifier.classify_query(query_text, detected_language)
        
        # Step 2: Create KB query object
        kb_query = KBQuery(
            text=query_text,
            language=detected_language,
            query_type=query_type,
            context=self._get_conversation_context(session_id) if session_id else context,
            user_id=user_id,
            session_id=session_id
        )
        
        # Step 3: Query knowledge base
        async with KnowledgeBaseClient() as kb_client:
            kb_response = await kb_client.query_knowledge_base(kb_query)
        
        # Step 4: Post-process response (could involve another LM call here)
        final_response = await self._post_process_response(
            kb_response, query_text, detected_language
        )
        
        # Step 5: Update conversation history
        if session_id:
            self._update_conversation_history(session_id, query_text, final_response, detected_language)
        
        # Prepare metadata
        metadata = {
            "query_type": query_type.value,
            "kb_confidence": kb_response.confidence,
            "kb_sources": kb_response.sources,
            "processing_time": time.time() - start_time,
            "kb_processing_time": kb_response.processing_time,
            "cached_response": kb_response.cached,
            "language": detected_language
        }
        
        return final_response, metadata

    def _get_conversation_context(self, session_id: str) -> Optional[str]:
        """Get recent conversation context for better responses"""
        if session_id in self.conversation_history:
            history = self.conversation_history[session_id]
            # Return last 3 exchanges as context
            recent_exchanges = history[-6:]  # Last 3 Q&A pairs
            context_parts = []
            for i in range(0, len(recent_exchanges), 2):
                if i + 1 < len(recent_exchanges):
                    context_parts.append(f"Q: {recent_exchanges[i]}")
                    context_parts.append(f"A: {recent_exchanges[i+1]}")
            return "\n".join(context_parts)
        return None

    def _update_conversation_history(self, session_id: str, query: str, response: str, language: str):
        """Update conversation history for context"""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        
        self.conversation_history[session_id].extend([query, response])
        
        # Keep only last 20 exchanges (40 items)
        if len(self.conversation_history[session_id]) > 40:
            self.conversation_history[session_id] = self.conversation_history[session_id][-40:]

    async def _post_process_response(
        self, 
        kb_response: KBResponse, 
        original_query: str, 
        target_language: str
    ) -> str:
        """
        Post-process KB response for better user experience
        This could involve:
        - Language consistency checks
        - Response formatting
        - Additional context injection
        """
        
        response = kb_response.answer
        
        # Add source attribution if available and confidence is high
        if kb_response.sources and kb_response.confidence > 0.7:
            source_text = {
                "hindi": "स्रोत",
                "sanskrit": "स्रोतः",
                "marathi": "स्रोत",
                "english": "Source"
            }.get(target_language, "Source")
            
            if len(kb_response.sources) == 1:
                response += f"\n\n({source_text}: {kb_response.sources[0]})"
            elif len(kb_response.sources) > 1:
                response += f"\n\n({source_text}: {', '.join(kb_response.sources[:3])})"
        
        # Add confidence indicator for low-confidence responses
        if kb_response.confidence < 0.5:
            uncertainty_text = {
                "hindi": "मुझे इस उत्तर के बारे में पूरा भरोसा नहीं है।",
                "sanskrit": "अस्य उत्तरस्य विषये मम पूर्णं विश्वासः नास्ति।",
                "marathi": "या उत्तराबद्दल मला पूर्ण खात्री नाही.",
                "english": "I'm not entirely confident about this answer."
            }.get(target_language, "I'm not entirely confident about this answer.")
            
            response = f"{response}\n\n({uncertainty_text})"
        
        return response

# Global orchestrator instance
qa_orchestrator = MultilingualQAOrchestrator()

async def process_qa_query(
    query_text: str, 
    detected_language: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to process Q&A queries
    
    Usage in FastAPI:
        response, metadata = await process_qa_query(
            request.text, 
            detected_language,
            user_id="user123",
            session_id="session456"
        )
    """
    return await qa_orchestrator.process_multilingual_query(
        query_text, detected_language, user_id, session_id
    )

def get_kb_stats() -> Dict[str, Any]:
    """Get KB integration statistics"""
    # This would typically be called from the API's /stats endpoint
    return {
        "conversation_sessions": len(qa_orchestrator.conversation_history),
        "total_exchanges": sum(len(history) for history in qa_orchestrator.conversation_history.values())
    }