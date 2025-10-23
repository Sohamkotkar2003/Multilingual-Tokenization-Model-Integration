# 🚀 Quick Start Guide

**Get up and running with the Multilingual AI System in 10 minutes!**

---

## 📋 Prerequisites

- **Python 3.11** or higher
- **NVIDIA GPU** with CUDA 12.6 support (RTX 4050 or better)
- **16GB RAM** minimum
- **10GB free disk space**
- **Windows 10/11** (or Linux/Mac with minor adjustments)

---

## ⚡ Quick Setup (5 Steps)

### Step 1: Clone Repository
```bash
git clone <your-repo-url>
cd Project
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
```

### Step 3: Install Dependencies
```bash
# Install PyTorch with CUDA first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install remaining packages
pip install -r requirements.txt
```

### Step 4: Download Trained Adapters

**Option A: Use Pre-trained NLLB Adapter (Recommended)**
1. Download `nllb_18languages_adapter.zip` from [Google Drive/Releases]
2. Extract to `adapters/nllb_18languages_adapter/`

**Option B: Train Your Own**
- See [Training Guide](#training-your-own-adapter) below

### Step 5: Verify Installation
```bash
python check_gpu.py
```

Expected output:
```
✅ CUDA available: True
✅ GPU: NVIDIA GeForce RTX 4050
✅ Python version: 3.11.x
✅ All dependencies installed
```

---

## 🎯 Running the Systems

### System 1: NLLB-200 Translation API

**Start the API:**
```bash
uvicorn adapter_service.standalone_api:app --host 127.0.0.1 --port 8115
```

**Test it:**
```bash
curl -X POST http://localhost:8115/generate-lite \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Hello, how are you?\",\"base_model\":\"facebook/nllb-200-distilled-600M\",\"adapter_path\":\"adapters/nllb_18languages_adapter\",\"max_new_tokens\":50}"
```

**Expected Response:**
```json
{
  "generated_text": "नमस्ते, आप कैसे हैं?",
  "time_taken": 0.51
}
```

**Supported Languages:** Assamese, Bengali, Bodo, English, Gujarati, Hindi, Kannada, Kashmiri, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu

---

### System 2: BLOOMZ Text Generation API

**Start the API:**
```bash
python main.py
```

Server will start on `http://localhost:8000`

**Test it:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"नमस्ते, मेरा नाम राज है\",\"language\":\"hindi\"}"
```

**Available Endpoints:**
- `POST /generate` - Generate text
- `POST /tokenize` - Tokenize text
- `POST /language-detect` - Detect language
- `POST /kb/query` - Query knowledge base
- `GET /health` - Health check

---

## 🧪 Testing

### Run Smoke Tests

**Test NLLB Translation:**
```bash
# Simple test
python scripts/test_simple_api.py

# Comprehensive test
python scripts/test_complete_api.py
```

**Test BLOOMZ API:**
```bash
python scripts/test_complete_api.py
```

**Test RL Pipeline:**
```bash
python scripts/test_rl_pipeline.py
```

---

## 📚 Common Use Cases

### 1. Translate English to Hindi
```bash
curl -X POST http://localhost:8115/generate-lite \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"Good morning, have a nice day!\",\"base_model\":\"facebook/nllb-200-distilled-600M\",\"adapter_path\":\"adapters/nllb_18languages_adapter\"}"
```

### 2. Translate to Multiple Languages
```python
import requests

def translate(text, target_lang_code):
    response = requests.post('http://localhost:8115/generate-lite', json={
        "prompt": text,
        "base_model": "facebook/nllb-200-distilled-600M",
        "adapter_path": "adapters/nllb_18languages_adapter",
        "max_new_tokens": 50
    })
    return response.json()['generated_text']

# Translate to multiple languages
text = "Hello, how are you?"
languages = {
    "Hindi": "hin_Deva",
    "Tamil": "tam_Taml",
    "Telugu": "tel_Telu"
}

for lang_name, lang_code in languages.items():
    translation = translate(text, lang_code)
    print(f"{lang_name}: {translation}")
```

### 3. Use Knowledge Base
```bash
curl -X POST http://localhost:8000/kb/query \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"भारत की राजधानी क्या है?\",\"language\":\"hindi\"}"
```

---

## 🎓 Training Your Own Adapter

### Using Google Colab (Recommended)

**Step 1:** Upload to Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload `notebooks/colab_train_nllb200.ipynb`
3. Set Runtime to **GPU (T4)**

**Step 2:** Upload Data
- Upload `data/flores200_dataset.tar.gz` when prompted

**Step 3:** Run All Cells
- Training takes ~2.5 hours

**Step 4:** Download Adapter
- `nllb_18languages_adapter.zip` will auto-download
- Extract to `adapters/nllb_18languages_adapter/`

---

## 🔧 Troubleshooting

### CUDA Not Available
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### API Not Starting
```bash
# Check if port is in use
netstat -ano | findstr :8115  # Windows
# lsof -i :8115               # Linux/Mac

# Kill process if needed
taskkill /PID <PID> /F        # Windows
# kill -9 <PID>               # Linux/Mac
```

### Out of Memory Errors
```bash
# Reduce batch size or use CPU fallback
# In standalone_api.py, model loads in FP16 by default
# For CPU testing, set device_map="cpu"
```

### Slow Performance
- Ensure GPU is being used (check `nvidia-smi`)
- Close other GPU-heavy applications
- Reduce `max_new_tokens` in requests

---

## 📖 Additional Resources

- **Full Documentation:** See `PROJECT_OVERVIEW.md` for complete technical details
- **File Reference:** See `FILE_DOCUMENTATION.md` for all files explained
- **API Testing:** Use Postman collections in `postman_collection/`
- **Example Code:** Check `examples/complete_integration_example.py`

---

## 🆘 Getting Help

**Common Issues:**
- Check `docs/` folder for detailed guides
- Review error logs in `logs/api.log`
- Test GPU with `python check_gpu.py`

**For Support:**
- Open an issue on GitHub
- Check existing documentation in `docs/`
- Review smoke test results in `results/`

---

## ✅ Verification Checklist

Before running in production, verify:

- [ ] Python 3.11+ installed
- [ ] CUDA 12.6 working (`nvidia-smi`)
- [ ] Virtual environment activated
- [ ] All dependencies installed (`pip list`)
- [ ] GPU detected (`python check_gpu.py`)
- [ ] NLLB adapter downloaded and extracted
- [ ] API starts without errors
- [ ] Test translation works
- [ ] Health endpoint returns 200

---

## 🚀 You're Ready!

Your multilingual AI system is now running. Start with simple translations and explore advanced features like Knowledge Base queries, multi-turn conversations, and RL episode collection.

**Default Ports:**
- NLLB Translation API: `http://localhost:8115`
- BLOOMZ Generation API: `http://localhost:8000`

**Quick Health Check:**
```bash
curl http://localhost:8115/health
curl http://localhost:8000/health
```

---

*Last Updated: October 23, 2025*  
*Questions? Check PROJECT_OVERVIEW.md for detailed explanations!*

