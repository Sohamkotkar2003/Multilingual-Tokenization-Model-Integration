# Sovereign LM Bridge + Multilingual KSML Core

**Author:** Soham Kotkar  
**Duration:** Oct 28 – Nov 2  
**Status:** 🚀 **Foundation Complete - Ready for Integration**

## 🎯 **Overview**

The Sovereign LM Bridge + Multilingual KSML Core is a comprehensive system that connects the LM Core, Vaani TTS, and Gurukul/Uniguru front-end through a sophisticated multilingual reasoning bridge with KSML semantic alignment.

## 🏗️ **Architecture**

```
sovereign_core/
├── api.py                    # Main FastAPI application
├── ksml/
│   ├── aligner.py           # KSML semantic alignment engine
│   └── ksml_roots.json      # Sanskrit root lookup
├── mcp/
│   └── feedback_stream.py   # MCP feedback collection
├── rl/
│   └── policy_updater.py    # RL self-improvement loop
├── vaani/
│   └── speech_composer.py   # Vaani compatibility layer
├── bridge/
│   └── reasoner.py          # Multilingual reasoning bridge
└── requirements.txt         # Dependencies
```

## 🚀 **Core Features**

### 1. **KSML Semantic Alignment Engine** (`/align.ksml`)
- **Intent Classification**: Question, statement, command, greeting, explanation, translation, educational, conversational
- **Language Detection**: 21 Indian languages + English with Unicode range analysis
- **Karma State Classification**: Sattva (pure/harmonious), Rajas (active/passionate), Tamas (inert/destructive)
- **Sanskrit Root Tagging**: Lightweight Sanskrit-root tagging via `ksml_roots.json`
- **Confidence Scoring**: Multi-factor confidence calculation

### 2. **MCP-Driven Feedback Stream** (`/rl.feedback`)
- **Live Feedback Collection**: Real-time user prompts + corrections
- **Auto-Storage**: Automatic storage to `/data/feedback_stream.jsonl`
- **MCP Connectors**: HuggingFace, S3, HTTP, Qdrant integration
- **Real-Time Updates**: Q-table or bandit-style policy updates
- **Background Processing**: Asynchronous feedback processing

### 3. **RL Self-Improvement Loop** (`/rl.feedback`)
- **Policy Updates**: Local adapter delta or policy table updates
- **Reward Processing**: { prompt, output, reward } feedback processing
- **Periodic Adjustments**: Reward-based adjustments without full retraining
- **S3 Sync**: Automatic sync to `s3://bhiv/rl_feedback/sovereign_core/`
- **Q-Learning**: Q-table updates with learning rate adaptation

### 4. **Vaani Compatibility Layer** (`/compose.speech_ready`)
- **Prosody Optimization**: Converts aligned text to prosody-optimized JSON
- **Tone Detection**: Calm, excited, serious, friendly, authoritative, gentle
- **Prosody Hints**: Gentle_low, confident_mid, energetic_high, calm_steady, etc.
- **Language Support**: Hindi, Sanskrit, Tamil, Bengali, English
- **Voice Selection**: Language and tone-specific voice selection

### 5. **Multilingual Reasoning Bridge** (`/bridge.reason`)
- **End-to-End Pipeline**: Complete orchestration of all components
- **LM Core Integration**: Connects to the `/compose.final_text` API
- **Unified Output**: Text + KSML + prosody in one response
- **Trace Logging**: Complete processing trace with `trace_id`
- **Performance Monitoring**: <2s end-to-end latency target

## 📊 **Performance Targets**

- **Latency**: <2s end-to-end processing
- **Memory**: <4GB VRAM on RTX 4050
- **Accuracy**: 90-95% KSML alignment confidence
- **Throughput**: 100+ requests/minute
- **Reliability**: 99.9% uptime target

## 🛠️ **Installation & Setup**

### 1. **Install Dependencies**
```bash
cd sovereign_core
pip install -r requirements.txt
```

### 2. **Initialize Data Directories**
```bash
mkdir -p data logs
```

### 3. **Start the API**
```bash
python api.py
```

The API will be available at `http://localhost:8116`

## 🔌 **API Endpoints**

### **Core Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/align.ksml` | POST | KSML semantic alignment |
| `/rl.feedback` | POST | RL feedback processing |
| `/compose.speech_ready` | POST | Vaani speech composition |
| `/bridge.reason` | POST | Complete reasoning bridge |

### **System Endpoints**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/stats` | GET | System statistics |
| `/docs` | GET | API documentation |

## 📝 **Usage Examples**

### **KSML Alignment**
```python
import requests

response = requests.post("http://localhost:8116/align.ksml", json={
    "text": "What is the meaning of dharma?",
    "source_lang": "en",
    "target_lang": "hi"
})

result = response.json()
print(f"Intent: {result['intent']}")
print(f"Karma State: {result['karma_state']}")
print(f"Semantic Roots: {result['semantic_roots']}")
```

### **RL Feedback**
```python
response = requests.post("http://localhost:8116/rl.feedback", json={
    "prompt": "Explain yoga",
    "output": "Yoga is a spiritual practice",
    "reward": 0.8,
    "user_id": "user123"
})
```

### **Speech Composition**
```python
response = requests.post("http://localhost:8116/compose.speech_ready", json={
    "text": "Yoga is a spiritual practice",
    "language": "hindi",
    "tone": "calm"
})

result = response.json()
print(f"Prosody Hint: {result['prosody_hint']}")
print(f"Audio Metadata: {result['audio_metadata']}")
```

### **Complete Bridge**
```python
response = requests.post("http://localhost:8116/bridge.reason", json={
    "text": "What is the meaning of dharma?",
    "user_id": "user123",
    "include_audio": True
})

result = response.json()
print(f"Aligned Text: {result['aligned_text']}")
print(f"KSML Metadata: {result['ksml_metadata']}")
print(f"Speech Ready: {result['speech_ready']}")
```

## 🔧 **Configuration**

### **Environment Variables**
```bash
export SOVEREIGN_HOST=127.0.0.1
export SOVEREIGN_PORT=8116
export S3_BUCKET=bhiv
export S3_PREFIX=rl_feedback/sovereign_core/
```

### **KSML Roots Customization**
Edit `ksml/ksml_roots.json` to add/modify Sanskrit roots:
```json
{
  "धातु": {
    "meaning": "root, foundation",
    "category": "fundamental",
    "karma_state": "sattva",
    "intent": "educational"
  }
}
```

## 📊 **Monitoring & Logging**

### **Log Files**
- `logs/ksml_bridge.jsonl`: Complete processing traces
- `data/feedback_stream.jsonl`: MCP feedback entries
- `data/rl_policy.json`: RL policy state
- `data/rl_sync.json`: S3 sync data

### **Health Monitoring**
```bash
curl http://localhost:8116/health
curl http://localhost:8116/stats
```

## 🚀 **Integration Points**

### **LM Core**
- **Endpoint**: `http://localhost:8000/compose.final_text`
- **Integration**: HTTP client with fallback handling
- **Data Flow**: Input → LM Core → KSML Alignment → Output

### **Vaani TTS**
- **Integration**: Via existing `src/integration/tts_integration.py`
- **Data Flow**: Aligned text → Speech composition → Prosody JSON
- **Output**: Audio metadata for TTS engine

### **MCP Connectors**
- **HuggingFace**: Model and dataset integration
- **S3**: Feedback and policy sync
- **HTTP**: External API integration
- **Qdrant**: Vector database for feedback

## 🔄 **Development Workflow**

### **1. Component Development**
Each component can be developed independently:
- `ksml/aligner.py`: KSML alignment logic
- `mcp/feedback_stream.py`: MCP integration
- `rl/policy_updater.py`: RL policy updates
- `vaani/speech_composer.py`: TTS composition
- `bridge/reasoner.py`: Pipeline orchestration

### **2. Testing**
```bash
# Test individual components
python -m pytest tests/test_ksml.py
python -m pytest tests/test_mcp.py
python -m pytest tests/test_rl.py
python -m pytest tests/test_vaani.py
python -m pytest tests/test_bridge.py

# Test complete integration
python -m pytest tests/test_integration.py
```

### **3. Deployment**
```bash
# Production deployment
uvicorn api:app --host 0.0.0.0 --port 8116 --workers 4
```

## 📈 **Future Enhancements**

### **Phase 2: Advanced Features**
- **Multi-Modal Support**: Image and audio input processing
- **Advanced RL**: Deep RL with neural policy networks
- **Real-Time Streaming**: WebSocket support for live feedback
- **Advanced Analytics**: Detailed performance metrics and dashboards

### **Phase 3: Scale & Optimization**
- **Distributed Processing**: Multi-node deployment
- **Caching Layer**: Redis-based response caching
- **Load Balancing**: Horizontal scaling support
- **Advanced Monitoring**: Prometheus/Grafana integration

## 🤝 **Contributing**

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **LM Core Team**: LM Core integration and API design
- **Karthikeya**: Vaani TTS system and prosody optimization
- **Gurukul/Uniguru**: Front-end integration requirements
- **MCP Community**: Multi-Cloud Protocol standards and connectors

---

**Status**: 🚀 **Foundation Complete - Ready for Integration**  
**Next Steps**: Integration testing, performance optimization, and production deployment
