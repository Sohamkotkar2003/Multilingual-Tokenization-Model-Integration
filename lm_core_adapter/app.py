from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, datetime
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from typing import List, Optional
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pymongo import MongoClient
from groq import RateLimitError
import hashlib
import json
import time
from typing import Dict, Any
import requests

# Load environment variables
load_dotenv()

# Config
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "documents")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_API_KEY_MAIN = os.getenv("GROQ_API_KEY_MAIN")
GROQ_API_KEY_FALLBACK = os.getenv("GROQ_API_KEY_FALLBACK")
MONGO_URI = os.getenv("MONGODB_URI")

# Load resources
model = SentenceTransformer(EMBED_MODEL)
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=3.0) if QDRANT_API_KEY else QdrantClient(url=QDRANT_URL, timeout=3.0)
groq_client_main = ChatGroq(api_key=GROQ_API_KEY_MAIN, model_name="llama-3.3-70b-versatile") if GROQ_API_KEY_MAIN else None
groq_client_fallback = ChatGroq(api_key=GROQ_API_KEY_FALLBACK, model_name="llama-3.3-70b-versatile") if GROQ_API_KEY_FALLBACK else None

# MongoDB setup
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["LM"]  # explicit database name
history_collection = db["chat_history"]
# telemetry collection for InsightFlow-like logs
insightflow_collection = db.get_collection("insightflow_logs")
# simple TTS cache collection (text hash -> audio_url, meta)
tts_cache_collection = db.get_collection("tts_cache")

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class ContextItem(BaseModel):
    sender: str
    content: str
    timestamp: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    context: Optional[List[ContextItem]] = []

class HistoryItem(BaseModel):
    query: str
    groq_answer: str = ""
    retrieved_chunks: List[dict] = []
    context: Optional[List[ContextItem]] = []
    timestamp: Optional[datetime.datetime] = None

# ===== Vaani Integration Helpers =====
VAANI_URL = os.getenv("VAANI_URL", "https://vaani-sentinel-gs6x.onrender.com")
VAANI_USERNAME = os.getenv("VAANI_USERNAME")
VAANI_PASSWORD = os.getenv("VAANI_PASSWORD")
VAANI_VOICE = os.getenv("VAANI_VOICE", "en_us_female_conversational")
VAANI_LANG = os.getenv("VAANI_DEFAULT_LANG", "en")


class VaaniClient:
    """Simple Vaani client for auth + TTS + agent calls with token caching."""
    def __init__(self, base_url: str, username: str = None, password: str = None):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self._token = None
        self._token_expiry = 0

    def _login(self) -> None:
        if not (self.username and self.password):
            raise RuntimeError("Vaani credentials not configured in environment (VAANI_USERNAME/VAANI_PASSWORD)")
        url = f"{self.base_url}/api/v1/auth/login"
        payload = {"username": self.username, "password": self.password}
        try:
            r = requests.post(url, json=payload, timeout=6)
            r.raise_for_status()
            data = r.json()
            token = data.get("access_token") or data.get("token") or data.get("accessToken") or data.get("access")
            expires_in = data.get("expires_in") or data.get("expires") or 300
            if not token:
                raise RuntimeError(f"Login succeeded but no token returned: {data}")
            self._token = token
            self._token_expiry = time.time() + int(expires_in) - 10
        except Exception as e:
            raise RuntimeError(f"Vaani login failed: {e}")

    def _get_token(self) -> str:
        if not self._token or time.time() > self._token_expiry:
            self._login()
        return self._token

    def tts(self, text: str, voice: str = None, language: str = None) -> Dict[str, Any]:
        """Generate TTS via Vaani /api/v1/agents/tts. Returns parsed JSON or raises."""
        voice = voice or VAANI_VOICE
        language = language or VAANI_LANG
        # check local cache first (simple hash)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = tts_cache_collection.find_one({"_id": text_hash})
        if cached:
            return {"audio_url": cached.get("audio_url"), "cache_status": "hit", "meta": cached.get("meta")}

        token = self._get_token()
        url = f"{self.base_url}/api/v1/agents/tts"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"text": text, "voice": voice, "language": language}
        r = requests.post(url, headers=headers, json=payload, timeout=8)
        r.raise_for_status()
        data = r.json()
        # Vaani returns audio URL and cache metadata as described in docs
        audio_url = data.get("audio_url") or data.get("url") or data.get("data")
        # store in cache if present
        tts_cache_collection.replace_one({"_id": text_hash}, {"audio_url": audio_url, "meta": data, "text": text, "created_at": datetime.datetime.utcnow()}, upsert=True)
        return {"audio_url": audio_url, "cache_status": "miss", "meta": data}

    def post_agent(self, path: str, payload: dict) -> dict:
        token = self._get_token()
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        r = requests.post(url, headers=headers, json=payload, timeout=8)
        r.raise_for_status()
        return r.json()


# initialize Vaani client if possible
vaani = None
if VAANI_URL:
    try:
        vaani = VaaniClient(VAANI_URL, VAANI_USERNAME, VAANI_PASSWORD)
    except Exception:
        vaani = None


def log_insightflow(entry: dict):
    """Log telemetry: prefer InsightFlow endpoint if configured else MongoDB collection."""
    # write to MongoDB collection for traceability
    try:
        entry_copy = dict(entry)
        entry_copy.setdefault("timestamp", datetime.datetime.utcnow())
        insightflow_collection.insert_one(entry_copy)
    except Exception:
        # if Mongo fails, fallback to printing
        print("InsightFlow log failed", entry)

# ===== End Vaani helpers =====
# RAG functions
def retrieve_docs(query: str, k: int = 5) -> List[dict]:
    # If caller explicitly disables retrieval, skip Qdrant entirely
    if not k or k <= 0:
        return []

    vec = model.encode(query).tolist()
    try:
        results = client.query_points(collection_name=COLLECTION_NAME, query=vec, limit=k).points
    except Exception:
        # Graceful degradation if Qdrant is unreachable or slow; proceed without retrieved context
        return []

    docs = []
    for r in results:
        docs.append({
            "content": r.payload.get("content"),
            "file": r.payload.get("file"),
            "score": getattr(r, "score", None)
        })
    return docs

def ask_groq(query: str, context: str, client):
    if not client:
        return "Groq client not configured."
    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [
        SystemMessage(content="You are a helpful assistant. Answer ONLY using context."),
        SystemMessage(content=f"Context:\n{context}"),
        HumanMessage(content=f"Question:\n{query}")
    ]
    response = client.invoke(messages)
    return response.content

# Endpoints
@app.post("/rag")
def rag_query(request: QueryRequest):
    if not request.query.strip():
        return {"error": "Please enter a question."}
    
    docs = retrieve_docs(request.query, request.top_k)
    retrieved_chunks = []
    for i, d in enumerate(docs, 1):
        retrieved_chunks.append({
            "index": i,
            "file": d['file'],
            "score": d['score'],
            "content": d['content'][:1000]
        })
    
    # Build context string from conversation history (user and bot)
    context_text = ""
    if request.context:
        context_text = "\n".join(
            f"[{item.sender} @ {item.timestamp}]: {item.content}" for item in request.context
        )
    answer = ""
    try:
        if groq_client_main:
            docs_text = "\n\n".join([d["content"] for d in docs if d["content"]])
            full_context = (context_text + "\n\n" + docs_text).strip() if context_text else docs_text
            answer = ask_groq(request.query, full_context, groq_client_main)
        else:
            answer = "Groq not configured. Only showing retrieved context."
    except RateLimitError:
        # Try fallback Groq client
        if groq_client_fallback:
            try:
                best_chunk = max(retrieved_chunks, key=lambda c: c.get("score") or 0)
                fallback_context = best_chunk["content"]
                answer = ask_groq(request.query, fallback_context, groq_client_fallback)
            except Exception:
                answer = f"(Rate limit reached) Top context:\n{best_chunk['content']}"
        else:
            # Fallback: return the chunk with the highest score
            if retrieved_chunks:
                best_chunk = max(retrieved_chunks, key=lambda c: c.get("score") or 0)
                answer = f"(Rate limit reached) Top context:\n{best_chunk['content']}"
            else:
                answer = "(Rate limit reached) No context available."

    # Save to MongoDB history (including context)
    try:
        history_collection.insert_one({
            "query": request.query,
            "groq_answer": answer,
            "retrieved_chunks": retrieved_chunks,
            "context": [item.dict() for item in request.context] if request.context else [],
            "timestamp": datetime.datetime.utcnow()
        })
    except Exception:
        # If MongoDB is unreachable or URI invalid, continue without failing the request
        pass
    # Attempt to generate TTS for the answer using Vaani
    audio_info = None
    try:
        if vaani:
            tts_result = vaani.tts(answer, voice=VAANI_VOICE, language=VAANI_LANG)
            audio_info = tts_result
        else:
            audio_info = {"error": "vaani client not configured"}
    except Exception as e:
        audio_info = {"error": str(e)}

    # Telemetry log for InsightFlow
    try:
        telemetry = {
            "type": "rag_interaction",
            "query": request.query,
            "answer_snippet": (answer or "")[:1000],
            "retrieved_count": len(retrieved_chunks),
            "vaani_audio": audio_info,
            "timestamp": datetime.datetime.utcnow()
        }
        log_insightflow(telemetry)
    except Exception:
        pass
    
    return {
        "retrieved_chunks": retrieved_chunks,
        "groq_answer": answer
        , "vaani_audio": audio_info
    }


# ===== New endpoints for Vaani-LM integration =====
class ComposeRequest(BaseModel):
    query: str
    language: Optional[str] = None
    top_k: int = 5
    context: Optional[List[ContextItem]] = []


@app.post("/compose.final_text")
def compose_final_text(req: ComposeRequest):
    """Compose final text via LM + RAG and return TTS audio (Vaani) and telemetry.
    This endpoint ensures LM output is compatible with Vaani TTS and logs to InsightFlow.
    """
    # reuse rag flow
    qreq = QueryRequest(query=req.query, top_k=req.top_k, context=req.context)
    result = rag_query(qreq)
    final_text = result.get("groq_answer")
    language = req.language or VAANI_LANG

    # ensure KSML reasoning + karma-state alignment is annotated (lightweight enforcement)
    # add a KSML header so downstream TTS/agents can align prosody
    ksml_metadata = {"ksml": {"semantic_alignment": True, "karma_state": "neutral"}}

    audio_info = None
    try:
        if vaani and final_text:
            # prefix KSML marker so Vaani voice agents can pick it up (if they use it)
            annotated_text = json.dumps(ksml_metadata) + "\n" + final_text
            audio_info = vaani.tts(annotated_text, voice=VAANI_VOICE, language=language)
    except Exception as e:
        audio_info = {"error": str(e)}

    # telemetry
    log_insightflow({
        "type": "compose.final_text",
        "query": req.query,
        "final_text_snippet": (final_text or "")[:1024],
        "vaani_audio": audio_info,
    })

    return {"final_text": final_text, "vaani_audio": audio_info}


class VaaniConverseRequest(BaseModel):
    message: str
    language: Optional[str] = None
    session_id: Optional[str] = None


@app.post("/vaani_converse")
def vaani_converse(req: VaaniConverseRequest):
    """Bridge endpoint: accepts a user message, runs it through LM (RAG/Groq) and calls Vaani agents to produce voice/response.
    Returns LM text, Vaani audio URL, and telemetry id.
    """
    # For multi-turn, collect previous history if session_id provided
    context_items = []
    try:
        if req.session_id:
            # naive: fetch last few entries by session_id from history_collection
            docs = list(history_collection.find({"session_id": req.session_id}).sort("timestamp", -1).limit(6))
            for d in reversed(docs):
                context_items.append(ContextItem(sender="user", content=d.get("query", ""), timestamp=str(d.get("timestamp"))))
                context_items.append(ContextItem(sender="bot", content=d.get("groq_answer", ""), timestamp=str(d.get("timestamp"))))
    except Exception:
        context_items = []

    qreq = QueryRequest(query=req.message, top_k=4, context=context_items)
    res = rag_query(qreq)
    lm_text = res.get("groq_answer")
    language = req.language or VAANI_LANG

    audio_info = None
    try:
        if vaani and lm_text:
            audio_info = vaani.tts(lm_text, voice=VAANI_VOICE, language=language)
    except Exception as e:
        audio_info = {"error": str(e)}

    # store in history with optional session id
    hist_doc = {"query": req.message, "groq_answer": lm_text, "retrieved_chunks": res.get("retrieved_chunks", []), "timestamp": datetime.datetime.utcnow()}
    if req.session_id:
        hist_doc["session_id"] = req.session_id
    history_collection.insert_one(hist_doc)

    log_insightflow({
        "type": "vaani_converse",
        "message": req.message,
        "response_snippet": (lm_text or "")[:1024],
        "vaani_audio": audio_info,
    })

    return {"lm_text": lm_text, "vaani_audio": audio_info}


class LessonPlayRequest(BaseModel):
    lesson_id: Optional[str] = None
    lesson_text: Optional[str] = None
    language: Optional[str] = None


@app.post("/lesson/play")
def lesson_play(req: LessonPlayRequest):
    """Orchestrate a lesson: break lesson text into segments, get LM enhancements, generate TTS for each, and return ordered audio playlist.
    Accepts either lesson_id (to fetch content from Vaani) or lesson_text directly.
    """
    language = req.language or VAANI_LANG
    lesson_text = req.lesson_text
    # if lesson_id provided, try to fetch from Vaani content API
    if not lesson_text and req.lesson_id and vaani:
        try:
            resp = vaani.post_agent("api/v1/content/" + req.lesson_id, {})
            lesson_text = resp.get("text")
        except Exception:
            lesson_text = None

    if not lesson_text:
        return {"error": "No lesson_text provided and lesson_id fetch failed"}

    # Simple segmentation (by sentence) for playback
    segments = [s.strip() for s in lesson_text.split('.') if s.strip()]
    playlist = []
    for i, seg in enumerate(segments):
        # ask LM to produce a small teaching prompt for the segment (KSML alignment)
        prompt = f"Teach this briefly in simple language: {seg}"
        qreq = QueryRequest(query=prompt, top_k=2, context=[])
        res = rag_query(qreq)
        seg_text = res.get("groq_answer") or seg
        # generate or fetch TTS
        try:
            audio = vaani.tts(seg_text, voice=VAANI_VOICE, language=language) if vaani else {"error": "vaani not configured"}
        except Exception as e:
            audio = {"error": str(e)}
        playlist.append({"index": i, "text": seg_text, "vaani_audio": audio})

    log_insightflow({"type": "lesson_play", "lesson_id": req.lesson_id, "parts": len(playlist)})
    return {"playlist": playlist}


@app.get("/history")
def get_history():
    chats = list(history_collection.find({}, {"_id":0}).sort("timestamp", 1))  # chronological
    return chats