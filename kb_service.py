"""
Custom Knowledge Base Service

This service provides a comprehensive knowledge base focused on:
- Multilingual language processing
- Cultural and linguistic information
- Technical NLP knowledge
- General knowledge in Hindi, Sanskrit, Marathi, and English
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import time
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class QueryType(str, Enum):
    FACTUAL = "factual"
    CONVERSATIONAL = "conversational"
    EDUCATIONAL = "educational"
    CULTURAL = "cultural"
    TECHNICAL = "technical"

class KBRequest(BaseModel):
    text: str
    language: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    generate_response: bool = True
    max_response_length: int = 256

class KBResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str]
    language: str
    query_type: str
    processing_time: float
    kb_answer: Optional[str] = None
    generated_response: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class KnowledgeBase:
    """Comprehensive knowledge base with multilingual support"""
    
    def __init__(self):
        self.knowledge = self._initialize_knowledge_base()
        self.stats = {
            "total_queries": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0
        }
    
    def _initialize_knowledge_base(self) -> Dict[str, Dict[str, Any]]:
        """Initialize the knowledge base with comprehensive information"""
        
        return {
            # Geography and Capitals
            "geography": {
                "hindi": {
                    "भारत की राजधानी": "भारत की राजधानी नई दिल्ली है। यह देश का राजनीतिक केंद्र है और यहाँ संसद भवन और राष्ट्रपति भवन स्थित हैं।",
                    "फ्रांस की राजधानी": "फ्रांस की राजधानी पेरिस है। यह सीन नदी के किनारे बसा एक खूबसूरत शहर है।",
                    "जापान की राजधानी": "जापान की राजधानी टोक्यो है। यह दुनिया के सबसे बड़े महानगरीय क्षेत्रों में से एक है।",
                    "अमेरिका की राजधानी": "अमेरिका की राजधानी वाशिंगटन डीसी है। यह पोटोमैक नदी के किनारे स्थित है।"
                },
                "sanskrit": {
                    "भारतस्य राजधानी": "भारतस्य राजधानी नवदिल्ली अस्ति। एतत् राष्ट्रस्य राजनीतिकं केंद्रम् अस्ति।",
                    "फ्रांसस्य राजधानी": "फ्रांसस्य राजधानी पेरिस अस्ति। एतत् सीन नद्याः तटे स्थितम् सुन्दरं नगरम् अस्ति।",
                    "जापानस्य राजधानी": "जापानस्य राजधानी टोक्यो अस्ति। एतत् विश्वस्य महान् महानगरीयक्षेत्रेषु अन्यतमम् अस्ति।"
                },
                "marathi": {
                    "भारताची राजधानी": "भारताची राजधानी नवी दिल्ली आहे। हे देशाचे राजकीय केंद्र आहे आणि येथे संसद भवन आणि राष्ट्रपती भवन आहेत।",
                    "फ्रान्साची राजधानी": "फ्रान्साची राजधानी पॅरिस आहे। हे सीन नदीच्या काठावर वसलेले एक सुंदर शहर आहे।",
                    "जपानची राजधानी": "जपानची राजधानी टोक्यो आहे। हे जगातील सर्वात मोठ्या महानगरीय क्षेत्रांपैकी एक आहे।"
                },
                "english": {
                    "capital of india": "The capital of India is New Delhi. It is the political center of the country and houses the Parliament House and Rashtrapati Bhavan.",
                    "capital of france": "The capital of France is Paris. It is a beautiful city located on the banks of the River Seine.",
                    "capital of japan": "The capital of Japan is Tokyo. It is one of the world's largest metropolitan areas.",
                    "capital of usa": "The capital of the United States is Washington D.C. It is located on the banks of the Potomac River."
                }
            },
            
            # Language and Linguistics
            "linguistics": {
                "hindi": {
                    "हिंदी भाषा": "हिंदी भारत की राजभाषा है और देवनागरी लिपि में लिखी जाती है। यह भारत-यूरोपीय भाषा परिवार की भाषा है।",
                    "संस्कृत भाषा": "संस्कृत भारत की प्राचीनतम भाषा है। यह वेदों और पुराणों की भाषा है।",
                    "मराठी भाषा": "मराठी महाराष्ट्र राज्य की मुख्य भाषा है। यह देवनागरी लिपि में लिखी जाती है।",
                    "टोकनाइज़ेशन": "टोकनाइज़ेशन पाठ को छोटे-छोटे टुकड़ों (टोकन्स) में तोड़ने की प्रक्रिया है। यह प्राकृतिक भाषा प्रसंस्करण का एक महत्वपूर्ण चरण है।"
                },
                "sanskrit": {
                    "हिन्दी भाषा": "हिन्दी भारतस्य राजभाषा अस्ति। एषा देवनागरी लिप्यां लिख्यते।",
                    "संस्कृत भाषा": "संस्कृत भारतस्य प्राचीनतमा भाषा अस्ति। एषा वेदानां पुराणानां च भाषा अस्ति।",
                    "मराठी भाषा": "मराठी महाराष्ट्र राज्यस्य मुख्यभाषा अस्ति। एषा देवनागरी लिप्यां लिख्यते।"
                },
                "marathi": {
                    "हिंदी भाषा": "हिंदी ही भारताची राजभाषा आहे आणि ती देवनागरी लिपीत लिहिली जाते।",
                    "संस्कृत भाषा": "संस्कृत ही भारताची प्राचीनतम भाषा आहे। ही वेद आणि पुराणांची भाषा आहे।",
                    "मराठी भाषा": "मराठी ही महाराष्ट्र राज्याची मुख्य भाषा आहे। ही देवनागरी लिपीत लिहिली जाते।",
                    "टोकनाइझेशन": "टोकनाइझेशन म्हणजे मजकूराला लहान-लहान भागांमध्ये (टोकन्स) विभाजित करण्याची प्रक्रिया।"
                },
                "english": {
                    "hindi language": "Hindi is the official language of India and is written in the Devanagari script. It belongs to the Indo-European language family.",
                    "sanskrit language": "Sanskrit is the ancient language of India. It is the language of the Vedas and Puranas.",
                    "marathi language": "Marathi is the main language of Maharashtra state. It is written in the Devanagari script.",
                    "tokenization": "Tokenization is the process of breaking text into smaller units called tokens. It is a crucial step in natural language processing."
                }
            },
            
            # Cultural Information
            "culture": {
                "hindi": {
                    "भारतीय संस्कृति": "भारतीय संस्कृति विश्व की सबसे पुरानी संस्कृतियों में से एक है। इसमें विविधता में एकता का सिद्धांत निहित है।",
                    "हिंदू धर्म": "हिंदू धर्म विश्व का सबसे पुराना धर्म है। इसमें वेद, उपनिषद और गीता जैसे पवित्र ग्रंथ हैं।",
                    "योग": "योग भारत की प्राचीन विद्या है जो शारीरिक, मानसिक और आध्यात्मिक कल्याण के लिए है।",
                    "भारतीय त्योहार": "भारत में दिवाली, होली, दशहरा, ईद, क्रिसमस जैसे कई त्योहार मनाए जाते हैं।"
                },
                "sanskrit": {
                    "भारतीय संस्कृतिः": "भारतीय संस्कृतिः विश्वस्य प्राचीनतमासु संस्कृतिषु अन्यतमा अस्ति।",
                    "हिन्दू धर्मः": "हिन्दू धर्मः विश्वस्य प्राचीनतमः धर्मः अस्ति। एतस्मिन् वेदाः, उपनिषदः, गीता च सन्ति।",
                    "योगः": "योगः भारतस्य प्राचीना विद्या अस्ति या शारीरिकं, मानसिकं, आध्यात्मिकं च कल्याणं करोति।"
                },
                "marathi": {
                    "भारतीय संस्कृती": "भारतीय संस्कृती ही जगातील सर्वात जुनी संस्कृतींपैकी एक आहे। यात विविधतेत एकता हा तत्त्व आहे।",
                    "हिंदू धर्म": "हिंदू धर्म हा जगातील सर्वात जुना धर्म आहे। यात वेद, उपनिषदे आणि गीता यासारखे पवित्र ग्रंथ आहेत।",
                    "योग": "योग ही भारताची प्राचीन विद्या आहे जी शारीरिक, मानसिक आणि आध्यात्मिक कल्याणासाठी आहे।"
                },
                "english": {
                    "indian culture": "Indian culture is one of the world's oldest cultures. It embodies the principle of unity in diversity.",
                    "hinduism": "Hinduism is the world's oldest religion. It includes sacred texts like the Vedas, Upanishads, and Bhagavad Gita.",
                    "yoga": "Yoga is India's ancient practice for physical, mental, and spiritual well-being.",
                    "indian festivals": "India celebrates many festivals including Diwali, Holi, Dussehra, Eid, and Christmas."
                }
            },
            
            # Technical NLP Knowledge
            "technical": {
                "hindi": {
                    "प्राकृतिक भाषा प्रसंस्करण": "प्राकृतिक भाषा प्रसंस्करण (एनएलपी) कंप्यूटर और मानव भाषा के बीच संपर्क का क्षेत्र है।",
                    "मशीन लर्निंग": "मशीन लर्निंग कंप्यूटर को बिना स्पष्ट प्रोग्रामिंग के सीखने की क्षमता देती है।",
                    "ट्रांसफॉर्मर मॉडल": "ट्रांसफॉर्मर मॉडल आधुनिक एनएलपी का आधार है, जो एटेंशन मैकेनिज्म का उपयोग करता है।",
                    "बहुभाषी मॉडल": "बहुभाषी मॉडल कई भाषाओं को समझ और उत्पन्न कर सकते हैं।"
                },
                "sanskrit": {
                    "प्राकृतिक भाषा प्रसंस्करण": "प्राकृतिक भाषा प्रसंस्करणं कंप्यूटर मानवभाषयोः मध्ये सम्पर्कस्य क्षेत्रम् अस्ति।",
                    "मशीन लर्निंग": "मशीन लर्निंग कंप्यूटराय बिना स्पष्ट प्रोग्रामिंग सीखने क्षमता ददाति।"
                },
                "marathi": {
                    "नैसर्गिक भाषा प्रक्रिया": "नैसर्गिक भाषा प्रक्रिया (एनएलपी) हे संगणक आणि मानवी भाषा यांच्यातील संपर्काचे क्षेत्र आहे।",
                    "मशीन लर्निंग": "मशीन लर्निंग संगणकाला स्पष्ट प्रोग्रामिंगशिवाय शिकण्याची क्षमता देते।",
                    "ट्रान्सफॉर्मर मॉडेल": "ट्रान्सफॉर्मर मॉडेल हे आधुनिक एनएलपीचे आधार आहे, जे अटेंशन मेकॅनिझम वापरते।"
                },
                "english": {
                    "natural language processing": "Natural Language Processing (NLP) is the field of interaction between computers and human language.",
                    "machine learning": "Machine learning gives computers the ability to learn without being explicitly programmed.",
                    "transformer model": "Transformer models are the foundation of modern NLP, using attention mechanisms.",
                    "multilingual model": "Multilingual models can understand and generate text in multiple languages.",
                    "attention mechanism": "Attention mechanisms allow models to focus on relevant parts of input sequences.",
                    "fine-tuning": "Fine-tuning is the process of adapting a pre-trained model to a specific task or domain."
                }
            }
        }
    
    def search_knowledge(self, query: str, language: str, query_type: str) -> Dict[str, Any]:
        """Search the knowledge base for relevant information"""
        
        query_lower = query.lower()
        results = []
        
        # Search across different knowledge categories
        for category, lang_data in self.knowledge.items():
            if language in lang_data:
                for key, value in lang_data[language].items():
                    # Simple keyword matching
                    if any(word in query_lower for word in key.lower().split()):
                        results.append({
                            "answer": value,
                            "source": f"{category.title()} Knowledge Base",
                            "confidence": 0.9,
                            "category": category
                        })
        
        # If no specific match found, provide general responses
        if not results:
            results = self._get_general_response(query, language, query_type)
        
        return {
            "results": results,
            "total_found": len(results),
            "query_type": query_type,
            "language": language
        }
    
    def _get_general_response(self, query: str, language: str, query_type: str) -> List[Dict[str, Any]]:
        """Provide general responses when specific knowledge is not found"""
        
        general_responses = {
            "hindi": {
                "factual": f"मैं आपके प्रश्न '{query}' के बारे में जानकारी खोज रहा हूं। यह एक तथ्यात्मक प्रश्न है। अधिक विशिष्ट जानकारी के लिए कृपया अपना प्रश्न विस्तार से पूछें।",
                "educational": f"आपका प्रश्न '{query}' शैक्षिक प्रकृति का है। मैं आपकी सहायता करने के लिए यहां हूं।",
                "technical": f"यह एक तकनीकी प्रश्न है: '{query}'। मैं प्राकृतिक भाषा प्रसंस्करण और मशीन लर्निंग के क्षेत्र में सहायता कर सकता हूं।",
                "cultural": f"आपका प्रश्न '{query}' सांस्कृतिक विषय से संबंधित है। भारतीय संस्कृति और परंपराओं के बारे में पूछ सकते हैं।",
                "conversational": f"नमस्ते! आपके प्रश्न '{query}' के बारे में बात करना अच्छा लगेगा।"
            },
            "sanskrit": {
                "factual": f"अहं भवतः प्रश्नस्य '{query}' विषये सूचनां अन्विष्यामि। एषा तथ्यात्मिका प्रश्ना अस्ति।",
                "educational": f"भवतः प्रश्नः '{query}' शैक्षिकः अस्ति। अहं भवतः सहायतार्थं अत्र अस्मि।",
                "technical": f"एषा तकनीकी प्रश्ना अस्ति: '{query}'। अहं प्राकृतिक भाषा प्रसंस्करण विषये सहायतां कर्तुं शक्नोमि।",
                "cultural": f"भवतः प्रश्नः '{query}' सांस्कृतिक विषये संबंधितः अस्ति। भारतीय संस्कृतेः विषये पृच्छतु।",
                "conversational": f"नमस्कारः! भवतः प्रश्नेन '{query}' सह वार्तालापः रोचकः भविष्यति।"
            },
            "marathi": {
                "factual": f"मी तुमच्या प्रश्नाबद्दल '{query}' माहिती शोधत आहे। हा एक तथ्यात्मक प्रश्न आहे।",
                "educational": f"तुमचा प्रश्न '{query}' शैक्षणिक स्वरूपाचा आहे। मी तुमची मदत करण्यासाठी येथे आहे।",
                "technical": f"हा एक तांत्रिक प्रश्न आहे: '{query}'। मी नैसर्गिक भाषा प्रक्रिया आणि मशीन लर्निंगच्या क्षेत्रात मदत करू शकतो।",
                "cultural": f"तुमचा प्रश्न '{query}' सांस्कृतिक विषयाशी संबंधित आहे। भारतीय संस्कृती आणि परंपरांबद्दल विचारू शकता।",
                "conversational": f"नमस्कार! तुमच्या प्रश्नाबद्दल '{query}' बोलणे छान वाटेल।"
            },
            "english": {
                "factual": f"I'm searching for information about '{query}'. This appears to be a factual question. For more specific information, please ask your question in more detail.",
                "educational": f"Your question '{query}' is educational in nature. I'm here to help you learn.",
                "technical": f"This is a technical question: '{query}'. I can help with natural language processing and machine learning topics.",
                "cultural": f"Your question '{query}' is related to cultural topics. You can ask about Indian culture and traditions.",
                "conversational": f"Hello! I'd be happy to discuss '{query}' with you."
            }
        }
        
        response_text = general_responses.get(language, general_responses["english"]).get(query_type, general_responses["english"]["factual"])
        
        return [{
            "answer": response_text,
            "source": "General Knowledge Base",
            "confidence": 0.7,
            "category": "general"
        }]

# Initialize the knowledge base
kb = KnowledgeBase()

# Create FastAPI app
app = FastAPI(
    title="Multilingual Knowledge Base API",
    description="Custom KB service for multilingual tokenization system",
    version="1.0.0"
)

@app.post("/multilingual-conversation", response_model=KBResponse)
async def multilingual_conversation(request: KBRequest):
    """Main KB endpoint for multilingual conversations"""
    
    start_time = time.time()
    
    try:
        # Classify query type
        query_type = _classify_query_type(request.text)
        
        # Search knowledge base
        search_results = kb.search_knowledge(request.text, request.language, query_type)
        
        # Get the best answer
        best_answer = search_results["results"][0] if search_results["results"] else None
        
        if best_answer:
            kb_answer = best_answer["answer"]
            confidence = best_answer["confidence"]
            sources = [best_answer["source"]]
        else:
            kb_answer = "I apologize, but I don't have specific information about that topic."
            confidence = 0.3
            sources = ["General Knowledge Base"]
        
        # Generate enhanced response if requested
        generated_response = None
        if request.generate_response:
            generated_response = _enhance_response(kb_answer, request.text, request.language)
        
        processing_time = time.time() - start_time
        
        return KBResponse(
            answer=generated_response or kb_answer,
            confidence=confidence,
            sources=sources,
            language=request.language,
            query_type=query_type,
            processing_time=processing_time,
            kb_answer=kb_answer,
            generated_response=generated_response,
            metadata={
                "total_results": search_results["total_found"],
                "kb_confidence": confidence,
                "kb_sources": sources
            }
        )
        
    except Exception as e:
        logger.error(f"KB service error: {e}")
        raise HTTPException(status_code=500, detail=f"KB service error: {e}")

def _classify_query_type(text: str) -> str:
    """Classify the query type based on content"""
    
    text_lower = text.lower()
    
    # Technical keywords
    if any(word in text_lower for word in ["api", "code", "programming", "technical", "model", "token", "python", "javascript", "nlp", "machine learning"]):
        return "technical"
    
    # Educational keywords
    if any(word in text_lower for word in ["explain", "teach", "learn", "study", "definition", "what is", "how does"]):
        return "educational"
    
    # Cultural keywords
    if any(word in text_lower for word in ["tradition", "festival", "culture", "history", "religion", "yoga", "hindu", "indian"]):
        return "cultural"
    
    # Geography keywords
    if any(word in text_lower for word in ["capital", "country", "city", "india", "france", "japan", "usa"]):
        return "factual"
    
    # Greeting keywords
    if any(word in text_lower for word in ["hello", "hi", "namaste", "नमस्ते", "नमस्कार"]):
        return "conversational"
    
    return "factual"  # Default

def _enhance_response(base_answer: str, original_query: str, language: str) -> str:
    """Enhance the response with additional context"""
    
    # For now, return the base answer with a note about the KB
    if language == "hindi":
        return f"{base_answer}\n\n(यह जानकारी हमारे ज्ञान आधार से प्राप्त की गई है।)"
    elif language == "sanskrit":
        return f"{base_answer}\n\n(एषा सूचना अस्माकं ज्ञान आधारात् प्राप्ता।)"
    elif language == "marathi":
        return f"{base_answer}\n\n(ही माहिती आमच्या ज्ञान आधारावरून मिळाली आहे।)"
    else:
        return f"{base_answer}\n\n(This information was retrieved from our knowledge base.)"

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "multilingual-kb", "version": "1.0.0"}

@app.get("/stats")
async def get_stats():
    """Get KB service statistics"""
    return {
        "knowledge_base": {
            "total_categories": len(kb.knowledge),
            "supported_languages": ["hindi", "sanskrit", "marathi", "english"],
            "total_entries": sum(len(lang_data) for category in kb.knowledge.values() for lang_data in category.values())
        },
        "service_stats": kb.stats
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
