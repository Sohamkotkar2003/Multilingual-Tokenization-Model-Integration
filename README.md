# Lightweight Adapter + MCP Pipeline 🚀

## Overview
**Production-ready multilingual generation system** with MCP (Multi-Cloud Protocol) streaming support, optimized for RTX 4050.

### ✅ **Status: 90% Complete & Deployable**

- ✅ Multilingual generation API (21+ languages)
- ✅ RL episode collection pipeline
- ✅ MCP streaming infrastructure (HF + S3 + HTTP + Qdrant)
- ✅ Comprehensive testing & documentation
- ⚠️ Adapter training (optional, not working)

## 🚀 Quick Start

```bash
# 1. Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 2. Start API server
python -m uvicorn adapter_service.standalone_api:app --host 127.0.0.1 --port 8110

# 3. Test generation (21+ languages supported!)
curl -X POST http://localhost:8110/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Translate to Hindi: Hello friend", "max_new_tokens": 50}'

# 4. Run tests
python scripts/test_simple_api.py
python scripts/test_rl_pipeline.py  
python scripts/test_mcp_streaming.py
```

## Project Structure

```
├── adapter_service/                  # Core streaming and inference
│   ├── standalone_api.py            # ✅ FastAPI server (8 endpoints)
│   ├── mcp_streaming.py             # ✅ MCP streaming (HF+S3+HTTP+Qdrant)
│   ├── train_with_mcp.py            # ✅ Training integration example
│   └── requirements-api.txt         # ✅ Production dependencies
├── rl/                              # Reinforcement Learning pipeline
│   ├── collect.py                   # ✅ Episode collection + cloud upload
│   └── rl_config.yaml               # ✅ RL configuration
├── scripts/                         # Testing suite
│   ├── test_simple_api.py           # ✅ API tests (10 languages)
│   ├── test_complete_api.py         # ✅ Comprehensive tests
│   ├── test_rl_pipeline.py          # ✅ RL pipeline tests
│   └── test_mcp_streaming.py        # ✅ MCP streaming tests
├── docs/                            # Documentation
│   ├── MCP_STREAMING_GUIDE.md       # ✅ MCP user guide
│   ├── MCP_IMPLEMENTATION_SUMMARY.md # ✅ Implementation details
│   ├── RL_PIPELINE_SUMMARY.md       # ✅ RL pipeline docs
│   ├── API_USAGE_GUIDE.md           # ✅ API documentation
│   └── HOW_TO.md                    # ✅ Usage guide
├── mcp_connectors.yml               # ✅ MCP data sources
└── smoke_results.md                 # ✅ Smoke test results
```

## ✨ Key Features

### 🌍 **Multilingual Generation (21+ Languages)**
- Hindi, Bengali, Tamil, Telugu, Gujarati, Marathi, Urdu, Punjabi
- Kannada, Malayalam, Assamese, Odia, Sanskrit, Nepali, Sindhi
- Kashmiri, Maithili, Bodo, Meitei, Santali, English
- Base model: `bigscience/bloomz-560m`

### 📡 **MCP Streaming (No Large Downloads)**
- **HuggingFace datasets** - Stream from 50,000+ datasets
- **S3/Cloud storage** - AWS S3, Azure Blob, GCS
- **HTTP APIs** - RESTful streaming endpoints
- **Qdrant Vector DB** - Vector database integration
- **Automatic fallback** - Falls back to local data
- **Memory efficient** - <100MB data buffer

### 🎮 **RL Episode Collection**
- Local episode logging (JSONL format)
- Cloud upload (S3 + HTTP)
- Reward calculation (length + quality + diversity)
- Multilingual support

### 🚀 **Production API (FastAPI)**
- `POST /generate` - Multilingual text generation
- `POST /adapter/train-lite` - Start training job
- `GET /adapter/status/{job_id}` - Check training status
- `GET /adapter/logs/{job_id}` - Get training logs
- `GET /adapter/list` - List available adapters
- `POST /rl/collect` - Collect RL episodes
- `GET /health` - Health check

## 📊 Performance

| Metric | Value |
|--------|-------|
| **Languages** | 21+ |
| **Generation speed** | 18-25s per request |
| **Memory usage** | ~300MB (model + buffer) |
| **MCP throughput** | ~1.1 samples/sec |
| **API response** | <1s (excluding generation) |

## 📚 Documentation

- **[MCP Streaming Guide](docs/MCP_STREAMING_GUIDE.md)** - Complete MCP usage
- **[MCP Implementation Summary](docs/MCP_IMPLEMENTATION_SUMMARY.md)** - Technical details
- **[RL Pipeline Summary](docs/RL_PIPELINE_SUMMARY.md)** - RL pipeline docs
- **[API Usage Guide](docs/API_USAGE_GUIDE.md)** - API reference
- **[How-To Guide](docs/HOW_TO.md)** - Quick commands

## Dependencies

Install lightweight requirements:
```bash
pip install -r adapter_service/requirements-lite.txt
```

## Usage (Coming Soon)

```bash
# Train lightweight adapter
python adapter_service/train_adapt.py --config adapter_config.yaml

# Start inference API
uvicorn adapter_service.api:app --host 0.0.0.0 --port 8100

# Run smoke tests
python test_prompts/run_smoke_tests.py
```