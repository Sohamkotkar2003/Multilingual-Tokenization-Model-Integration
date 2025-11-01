# 🚀 How to Run BHIV Sovereign AI Platform

## Quick Start Guide

### Prerequisites
- Python 3.11+
- NVIDIA GPU (RTX 4050 or better)
- Virtual environment activated

---

## 📦 Setup (First Time Only)

```bash
# 1. Navigate to project
cd C:\pc\Project

# 2. Activate virtual environment
venv\Scripts\activate

# 3. Install dependencies (if not already installed)
pip install -r requirements.txt

# 4. Set environment encoding
set PYTHONIOENCODING=utf-8
set PYTHONPATH=.
```

---

## 🏃 Running the Servers

### Start Both Servers (2 Terminals Required)

#### Terminal 1: LM Core (Groq + RAG + Multilingual)
```bash
cd C:\pc\Project
venv\Scripts\activate
set PYTHONIOENCODING=utf-8
set PYTHONPATH=.
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8117
```

**Wait for**: `✅ Model loaded on GPU: NVIDIA GeForce RTX 4050`

#### Terminal 2: Sovereign Core (KSML + RL + MCP + Vaani)
```bash
cd C:\pc\Project
venv\Scripts\activate
set PYTHONIOENCODING=utf-8
set PYTHONPATH=.
set MCP_STREAM_ENABLED=1
python -m uvicorn sovereign_core.api:app --host 127.0.0.1 --port 8116
```

**Wait for**: `✅ All Sovereign Core components initialized successfully`

---

## 🧪 Running Tests

### Quick Health Check
```bash
# In a third terminal
cd C:\pc\Project
venv\Scripts\activate
curl http://localhost:8117/health
curl http://localhost:8116/health
```

### Comprehensive Test Suite (Recommended)
```bash
cd C:\pc\Project
venv\Scripts\activate
set PYTHONIOENCODING=utf-8
python comprehensive_system_test.py
```

**Output**: 
- ✅ Runs 45 tests across all components
- ✅ Generates detailed logs in `test_results/`
- ✅ Takes ~3 minutes to complete
- ✅ Shows PASS/FAIL for each test

### Specific Component Tests
```bash
# Test RL + KSML integration
set PYTHONIOENCODING=utf-8
python scripts/smoke_rl_ksml.py

# Send RL feedback (10 samples)
python scripts/send_rl_feedback.py http://127.0.0.1:8116/rl.feedback

# Test all endpoints individually
python scripts/test_endpoints.py
```

---

## 🌐 Available Endpoints

### LM Core (Port 8117)
- `POST /generate` - Text generation (21 languages)
- `POST /qa` - Q&A with RAG
- `POST /multilingual-conversation` - Multi-turn chat
- `POST /language-detect` - Language detection
- `GET /health` - Health status
- `GET /stats` - System statistics

### Sovereign Core (Port 8116)
- `POST /align.ksml` - KSML semantic alignment
- `POST /rl.feedback` - RL policy updates
- `POST /compose.speech_ready` - Vaani TTS-ready output
- `POST /bridge.reason` - Complete pipeline (LM → KSML → RL → Vaani)
- `GET /health` - Health status
- `GET /stats` - Component statistics

---

## 📊 Verify Everything is Working

After starting both servers, check:

1. **LM Core**: http://localhost:8117/health
   - Should show: `"status": "healthy"`
   - Model on CUDA: ✅

2. **Sovereign Core**: http://localhost:8116/health
   - All 5 components: ✅
   - KSML, MCP, RL, Vaani, Bridge

3. **Run Full Test Suite**:
   ```bash
   python comprehensive_system_test.py
   ```
   - Should show: **45/45 tests passed (100%)**

---

## 🛑 Stopping the Servers

Press `CTRL+C` in each terminal window.

---

## 📝 Logs & Results

All logs and test results are saved in:
```
logs/
├── api.log                  # LM Core logs
├── ksml_bridge.jsonl        # Sovereign Core processing logs

test_results/
├── comprehensive_test_*.log # Detailed test logs
├── test_results_*.json      # Machine-readable results
└── COMPREHENSIVE_TEST_REPORT_*.md  # Test reports
```

---

## ⚡ Quick Test Commands

```bash
# Test language detection
curl -X POST http://localhost:8117/language-detect -H "Content-Type: application/json" -d "{\"text\": \"नमस्ते\"}"

# Test KSML alignment
curl -X POST http://localhost:8116/align.ksml -H "Content-Type: application/json" -d "{\"text\": \"Create a todo app\", \"target_lang\": \"en\"}"

# Test RL feedback
curl -X POST http://localhost:8116/rl.feedback -H "Content-Type: application/json" -d "{\"prompt\": \"test\", \"output\": \"response\", \"reward\": 0.8}"

# Test complete bridge
curl -X POST http://localhost:8116/bridge.reason -H "Content-Type: application/json" -d "{\"text\": \"What is AI?\", \"include_audio\": false}"
```

---

## 🎯 Success Indicators

When everything is running correctly, you should see:

✅ Both servers start without errors  
✅ Model loaded on GPU (RTX 4050)  
✅ All 5 Sovereign components initialized  
✅ MCP stream running in background  
✅ Health endpoints return "healthy"  
✅ Test suite shows 100% pass rate  

---

**That's it! Your platform is ready to use.** 🎉

For detailed documentation, see `README.md` and `sovereign_core/README.md`.

