# 📁 Complete File Documentation

**Project:** Multilingual AI Translation & Generation System  
**Last Updated:** October 23, 2025  
**Total Files:** 1,000+ files across 50+ folders

---

## 📚 Table of Contents

- [Root Files](#root-files)
- [Core Directories](#core-directories)
  - [adapters/](#adapters)
  - [adapter_service/](#adapter_service)
  - [cache/](#cache)
  - [config/](#config)
  - [data/](#data)
  - [data_cleaning/](#data_cleaning)
  - [docs/](#docs)
  - [examples/](#examples)
  - [flores200_dataset/](#flores200_dataset)
  - [logs/](#logs)
  - [model/](#model)
  - [notebooks/](#notebooks)
  - [postman_collection/](#postman_collection)
  - [results/](#results)
  - [rl/](#rl)
  - [rl_runs/](#rl_runs)
  - [scripts/](#scripts)
  - [src/](#src)
  - [test_prompts/](#test_prompts)
  - [tests/](#tests)
  - [utils/](#utils)
  - [venv/](#venv)

---

## 🗂️ Root Files

### `.gitignore`
**Lines:** 130  
**Purpose:** Git ignore configuration  
**Description:** Comprehensive list of files and folders to exclude from version control. Includes Python caches, virtual environments, logs, model checkpoints, datasets, and system files.  
**Key Sections:**
- Python-specific ignores (`__pycache__/`, `*.pyc`, `*.egg-info/`)
- ML/AI ignores (checkpoints, datasets, model files)
- IDE ignores (VSCode, PyCharm, Jupyter)
- OS ignores (Windows, macOS, Linux temp files)

---

### `README.md`
**Lines:** 139  
**Purpose:** Main project documentation  
**Description:** Project overview, quick start guide, and feature documentation. Describes the MCP-enabled multilingual generation system with adapter training, RL pipeline, and API endpoints.  
**Key Sections:**
- Quick Start commands
- Project structure overview
- Feature list (21+ languages, MCP streaming, RL collection)
- API endpoints documentation
- Performance metrics
- Known issues (adapter training limitations)

---

### `main.py`
**Lines:** 34  
**Purpose:** Entry point for BLOOMZ System 2 (Text Generation API)  
**Description:** Imports and runs the FastAPI application from `src/api/main.py` using configuration from `config/settings.py`. Starts the uvicorn server with auto-reload in debug mode.  
**Usage:** `python main.py` (starts API on port 8000)  
**System:** BLOOMZ Text Generation with KB/TTS/NLP integrations

---

### `requirements.txt`
**Lines:** 146  
**Purpose:** Python dependencies for entire project  
**Description:** Comprehensive, well-organized list of all required packages for both NLLB-200 and BLOOMZ systems. Includes 63 packages organized into 13 logical categories with inline documentation.  
**Key Packages:**
- Core ML: `torch`, `transformers`, `peft`, `accelerate`
- Web Framework: `fastapi`, `uvicorn`, `pydantic`
- Data: `datasets`, `pandas`, `numpy`
- Optional: `boto3` (for S3 uploads)

---

## 📂 Core Directories

---

## `adapters/`

**Purpose:** Stores trained LoRA adapters  
**Importance:** ⭐⭐⭐⭐⭐ CRITICAL - Contains production models

### `adapters/nllb_18languages_adapter/`

**Size:** ~41 MB  
**Purpose:** NLLB-200 LoRA adapter for 18 Indian languages  
**Importance:** ⭐⭐⭐⭐⭐ **PRIMARY DELIVERABLE**  
**Description:** Fine-tuned adapter for `facebook/nllb-200-distilled-600M` trained on FLORES-200 dataset. Achieves 90-95% translation accuracy across 18 languages.

#### Files:
- **`adapter_model.bin`** (~19 MB): LoRA adapter weights
- **`adapter_config.json`**: PEFT configuration (base model, LoRA params)
- **`tokenizer.json`** (~17 MB): Fast tokenizer configuration
- **`sentencepiece.bpe.model`** (~5 MB): SentencePiece BPE model
- **`tokenizer_config.json`**: Tokenizer settings
- **`special_tokens_map.json`**: Special token definitions
- **`README.md`**: Model card with metadata

**Training Details:**
- Base Model: `facebook/nllb-200-distilled-600M`
- Dataset: FLORES-200
- Epochs: 3
- Training Time: ~2.5 hours on T4 GPU
- Precision: FP16
- LoRA rank: 8

---

### `adapters/gurukul_lite/`

**Size:** ~97 MB  
**Purpose:** BLOOMZ-560M LoRA adapter  
**Description:** Adapter for `bigscience/bloomz-560m` used in System 2 for text generation.

#### Files:
- **`adapter_model.safetensors`** (~97 MB): LoRA weights in SafeTensors format
- **`adapter_config.json`**: PEFT configuration
- **`tokenizer.json`** (~17 MB): Fast tokenizer
- **`tokenizer_config.json`**: Tokenizer settings
- **`special_tokens_map.json`**: Special tokens
- **`README.md`**: Model card

---

## `adapter_service/`

**Purpose:** NLLB-200 Translation API (System 1)  
**Importance:** ⭐⭐⭐⭐⭐ CRITICAL - Production API

### `adapter_service/standalone_api.py`

**Lines:** 800  
**Purpose:** FastAPI server for NLLB-200 translation  
**Description:** Production-ready API with 8 endpoints, model caching, request queuing, and GPU memory management.

**Key Features:**
- Model caching (avoid reloading)
- Request queuing (`asyncio.Lock` for sequential processing)
- Explicit GPU memory cleanup
- FP16 model loading
- Gradient checkpointing

**Endpoints:**
1. `POST /adapter/train-lite` - Start adapter training job
2. `POST /generate-lite` - Generate text/translation
3. `GET /adapter/status/{job_id}` - Check training status
4. `GET /adapter/logs/{job_id}` - Get training logs
5. `GET /adapter/logs/{job_id}/tail` - Get last N log lines
6. `GET /adapter/list` - List available adapters
7. `POST /rl/collect` - Collect RL episodes
8. `GET /health` - Health check

**Usage:** `uvicorn adapter_service.standalone_api:app --host 127.0.0.1 --port 8115`

---

### `adapter_service/model_utils.py`

**Lines:** 252  
**Purpose:** Model loading and management utilities  
**Description:** Helper functions for loading base models, applying adapters, merging weights, and device management.

**Functions:**
- `load_base_model()`: Load HuggingFace model with optional 8-bit quantization
- `load_adapter()`: Apply LoRA adapter to base model
- `merge_adapter()`: Merge adapter weights into base model
- `get_model_info()`: Get model statistics
- `unload_model()`: Clean up model from memory

---

### `adapter_service/requirements-api.txt`

**Lines:** 12  
**Purpose:** Minimal dependencies for NLLB API  
**Description:** Lightweight requirements file for production deployment of the NLLB translation API.

**Packages:**
- FastAPI, uvicorn, pydantic (web framework)
- accelerate, transformers, peft (ML)
- bitsandbytes, datasets, pyyaml (utilities)

---

## `cache/`

**Purpose:** Caches preprocessed datasets  
**Importance:** ⭐⭐ MODERATE - Improves training performance

### `cache/tokenized/`

**Size:** ~100+ MB  
**Purpose:** HuggingFace datasets cache  
**Description:** Contains preprocessed Arrow format datasets for faster loading during training. Auto-generated by the `datasets` library.

**Subdirectories:**
- `train_8f5873aa1fbf9f6b4817db9e84811f6d/`: Training split cache (9 Arrow files)
- `eval_8f5873aa1fbf9f6b4817db9e84811f6d/`: Evaluation split cache (2 Arrow files)
- `train_999a6e1616fd41b73f400b67d4070455/`: Training split cache (25 Arrow files)
- `eval_999a6e1616fd41b73f400b67d4070455/`: Evaluation split cache (5 Arrow files)

**Files per cache:**
- `data-*.arrow`: Columnar data in Apache Arrow format
- `dataset_info.json`: Dataset metadata
- `state.json`: Cache state information

---

## `config/`

**Purpose:** Configuration for BLOOMZ System 2  
**Importance:** ⭐⭐⭐⭐⭐ CRITICAL for BLOOMZ API

### `config/settings.py`

**Lines:** 274  
**Purpose:** Configuration settings for BLOOMZ Text Generation API  
**Description:** Central configuration file defining model paths, API settings, language configurations, and integration endpoints.

**Key Configurations:**
- **API Settings:** Host (127.0.0.1), Port (8000), Debug mode
- **Model Paths:** `bigscience/bloom-560m`, tokenizer paths
- **Languages:** 21 Indian languages with Unicode ranges and keywords
- **Generation Params:** Max length, temperature, top-p, top-k
- **Integration Endpoints:** KB (8003), TTS (8001), NLP (8002)
- **Training Paths:** Data directories for training/validation

**Language Detection:**
- Unicode range mapping for 21 languages
- Language-specific keywords for detection
- Confidence thresholds

---

## `data/`

**Purpose:** Training datasets and FLORES-200  
**Importance:** ⭐⭐⭐⭐ HIGH - Training data

### `data/training/`

**Size:** ~2.5 GB total (21 files)  
**Purpose:** Custom corpus for 21 Indian languages  
**Description:** Cleaned and deduplicated training data for each language. Used for BLOOMZ training (originally) and reference data.

**Files:** (Language → Filename → Size)
- Assamese → `as_train.txt` → 171 MB
- Bengali → `bn_train.txt` → 194 MB
- Bodo → `bd_train.txt` → 22 MB
- English → `en_train.txt` → 58 MB
- Gujarati → `gu_train.txt` → 160 MB
- Hindi → `hi_train.txt` → 186 MB
- Kannada → `kn_train.txt` → 143 MB
- Kashmiri → `ks_train.txt` → 299 KB
- Maithili → `mai_train.txt` → 81 MB
- Malayalam → `ml_train.txt` → 193 MB
- Meitei → `mni_train.txt` → 2.5 MB
- Marathi → `mr_train.txt` → 185 MB
- Nepali → `ne_train.txt` → 143 MB
- Odia → `or_train.txt` → 134 MB
- Punjabi → `pa_train.txt` → 170 MB
- Sanskrit → `sa_train.txt` → 170 MB
- Santali → `sat_train.txt` → 22 MB
- Sindhi → `sd_train.txt` → 45 MB
- Tamil → `ta_train.txt` → 106 MB
- Telugu → `te_train.txt` → 89 MB
- Urdu → `ur_train.txt` → 235 MB

---

### `data/validation/`

**Size:** Similar to training (21 files)  
**Purpose:** Validation split of custom corpus  
**Description:** Validation data for each language, used for model evaluation during training.

---

### `data/flores200_dataset.tar.gz`

**Size:** 25 MB (compressed)  
**Purpose:** FLORES-200 dataset for Colab upload  
**Description:** Compressed archive of the FLORES-200 parallel corpus. Used in `notebooks/colab_train_nllb200.ipynb` for easy upload to Google Colab.

**Contents:** 200+ language pairs with dev and devtest splits

---

## `data_cleaning/`

**Purpose:** Data preprocessing scripts  
**Importance:** ⭐⭐⭐ MODERATE - Data preparation

**Description:** 21 Python scripts for cleaning and preprocessing raw text data for each language. Performs Unicode normalization, sentence segmentation, deduplication, and script-specific filtering.

**Files:** (21 scripts, one per language)
- `clean_assamese.py`
- `clean_bengali.py`
- `clean_bodo.py`
- `clean_english.py`
- `clean_gujurati.py`
- `clean_hindi.py`
- `clean_kannada.py`
- `clean_kashmiri.py`
- `clean_maithili.py`
- `clean_malyalam.py`
- `clean_marathi.py`
- `clean_meitei.py`
- `clean_nepali.py`
- `clean_odia.py`
- `clean_punjabi.py`
- `clean_sanskrit.py`
- `clean_santali.py`
- `clean_sindhi.py`
- `clean_tamil.py`
- `clean_telugu.py`
- `clean_urdu.py`

**Common Functions:**
- `normalize_unicode()`: Normalize to NFC
- `segment_sentences()`: Split on punctuation (language-specific)
- `clean_sentence()`: Filter allowed Unicode ranges
- `deduplicate()`: Remove duplicate sentences
- `process_text_file_in_batches()`: Batch processing for large files

**Example:** `clean_hindi.py`
- Filters Devanagari Unicode range (U+0900 to U+097F)
- Segments on Hindi danda (।), period, question mark
- Processes in batches of 500K lines
- Outputs to `hi_train.txt` and `hi_val.txt`

---

## `docs/`

**Purpose:** Project documentation  
**Importance:** ⭐⭐⭐⭐ HIGH - Essential reference

### Current Documentation Files:

#### `docs/SMOKE_TEST_GUIDE.md`
**Lines:** ~300  
**Purpose:** Guide for running NLLB smoke tests  
**Description:** Comprehensive instructions for using `smoke_test_nllb_colab.ipynb` to validate NLLB-200 adapter quality across all 21 languages.

---

#### `docs/QUALITY_ANALYSIS.md`
**Purpose:** BLOOMZ translation quality analysis  
**Description:** Historical document analyzing BLOOMZ adapter quality issues (Chinese characters in Gujarati, English fallback for Telugu). Explains why NLLB-200 was adopted.

---

#### `docs/MEMORY_CLEANUP.md`
**Purpose:** API memory management documentation  
**Description:** Details on model caching, request queuing, and GPU memory cleanup features in `adapter_service/standalone_api.py`.

---

#### `docs/HOW_TO_TEST_NLLB_ADAPTER.md`
**Purpose:** NLLB adapter testing guide  
**Description:** Step-by-step instructions for testing the trained NLLB-200 adapter, including local testing and Colab testing.

---

#### `docs/NLLB200_QUICK_START.md`
**Purpose:** NLLB-200 quick reference  
**Description:** Quick start guide for NLLB-200 system, including training, testing, and deployment.

---

#### `docs/SMOKE_TEST_README.md`
**Purpose:** Smoke testing documentation  
**Description:** Overview of smoke testing process, test coverage, and result interpretation.

---

#### `docs/TASK_REQUIREMENTS_CHECKLIST.md`
**Lines:** 460  
**Purpose:** Original task verification  
**Description:** Comprehensive checklist verifying 100% completion of original task requirements. Documents all deliverables, acceptance criteria, and timeline milestones.

---

#### `docs/DEEP_PROJECT_ANALYSIS.md`
**Lines:** 669  
**Purpose:** Project architecture analysis  
**Description:** In-depth technical analysis of project architecture, design decisions, and implementation details.

---

#### `docs/*.pdf` / `docs/*.txt`
**Purpose:** Original task requirements  
**Files:**
- `Soham Kotkar Learning Task.pdf`
- `Soham Kotkar Learning Task.txt`
- `Soham_Kotkar_Test_Task[1].pdf`
- `1760777518355-Soham Kotkar Adapter MCP Task.pdf`

---

## `examples/`

**Purpose:** Integration examples for BLOOMZ System 2  
**Importance:** ⭐⭐⭐⭐ HIGH - Demonstrates capabilities

### `examples/complete_integration_example.py`

**Lines:** 391  
**Purpose:** Comprehensive integration test suite  
**Description:** Demonstrates complete integration between Multilingual Tokenization Model, Indigenous NLP, and Vaani TTS.

**Test Scenarios:**
1. Basic integration (language detection, tokenization)
2. Knowledge Base integration (Q&A across languages)
3. TTS integration (text-to-speech synthesis)
4. NLP integration (preprocessing, sentiment analysis)
5. Complete pipeline (input → processing → response → audio)
6. Async pipeline (concurrent processing)
7. Cached pipeline (performance optimization)
8. Multi-turn conversations
9. Performance benchmarking

**Usage:** `python examples/complete_integration_example.py`

---

## `flores200_dataset/`

**Purpose:** FLORES-200 parallel corpus (extracted)  
**Importance:** ⭐⭐⭐⭐⭐ CRITICAL - Training data for NLLB-200

**Size:** ~76 MB  
**Description:** Multilingual parallel translation dataset with 200+ languages. Used for training the NLLB-200 adapter.

### Subdirectories:

#### `flores200_dataset/dev/`
**Files:** 204 language files (`.dev` extension)  
**Purpose:** Development/validation split  
**Description:** Parallel translations of the same sentences across 200+ languages. Each file contains ~1,000 sentences.

**Key Language Files:**
- `eng_Latn.dev` - English (source)
- `hin_Deva.dev` - Hindi
- `ben_Beng.dev` - Bengali
- `tam_Taml.dev` - Tamil
- `tel_Telu.dev` - Telugu
- `guj_Gujr.dev` - Gujarati
- (and 18 more target languages)

---

#### `flores200_dataset/devtest/`
**Files:** 204 language files (`.devtest` extension)  
**Purpose:** Test split  
**Description:** Similar to dev split, used for final evaluation.

---

#### Other Files:
- **`metadata_dev.tsv`**: Metadata for dev split
- **`metadata_devtest.tsv`**: Metadata for devtest split
- **`README`**: FLORES-200 dataset documentation

---

## `logs/`

**Purpose:** Runtime logs for APIs  
**Importance:** ⭐⭐ MODERATE - Debugging

**Description:** Empty folder where BLOOMZ API (`src/api/main.py`) writes runtime logs. Configured in `config/settings.py`.

**Log Files:** (auto-generated)
- `api.log` - API request/response logs, errors, warnings

---

## `model/`

**Purpose:** Custom SentencePiece tokenizer  
**Importance:** ⭐⭐⭐ MODERATE - Used by BLOOMZ API

### Files:

#### `model/multilingual_tokenizer.model`
**Size:** 1.03 MB  
**Purpose:** SentencePiece BPE model  
**Description:** Trained tokenizer model for multilingual text. Used by BLOOMZ API's `/tokenize` endpoint.

---

#### `model/multilingual_tokenizer.vocab`
**Size:** 476 KB  
**Purpose:** Tokenizer vocabulary  
**Description:** Vocabulary file mapping tokens to IDs.

---

## `notebooks/`

**Purpose:** Google Colab training and testing notebooks  
**Importance:** ⭐⭐⭐⭐⭐ CRITICAL - Training workflow

### `notebooks/colab_train_nllb200.ipynb`

**Cells:** 12  
**Purpose:** Train NLLB-200 adapter on FLORES-200  
**Description:** Complete training workflow for NLLB-200 adapter. Designed for Google Colab T4 GPU.

**Workflow:**
1. Check GPU and environment
2. Install packages (transformers, peft, accelerate, sentencepiece)
3. Upload `flores200_dataset.tar.gz`
4. Extract and parse FLORES-200 data
5. Load NLLB-200 base model in FP16
6. Configure LoRA (rank=8, target modules: q/k/v/out projections)
7. Prepare dataset
8. Configure training (3 epochs, batch=8, grad_accum=2, FP16)
9. Train adapter (~2.5 hours)
10. Save adapter
11. Test adapter with sample translations
12. Download adapter as ZIP

**Training Time:** ~2.5 hours on T4 GPU  
**Output:** `nllb_18languages_adapter.zip`

---

### `notebooks/test_nllb_adapter_colab.ipynb`

**Cells:** 8  
**Purpose:** Test trained NLLB-200 adapter  
**Description:** Upload and test the trained adapter with various prompts. Includes critical fix for target language specification.

**Key Feature:** Explicit target language codes
```python
tokenizer.src_lang = "eng_Latn"
tokenizer.tgt_lang = "hin_Deva"  # Force Hindi output
forced_bos_token_id = tokenizer.convert_tokens_to_ids("hin_Deva")
```

**Workflow:**
1. Install packages
2. Upload adapter ZIP
3. Load NLLB-200 base model + adapter
4. Run tests with target language forcing
5. Display results

---

### `notebooks/smoke_test_nllb_colab.ipynb`

**Cells:** 20+  
**Purpose:** Generate comprehensive smoke test results  
**Description:** Tests NLLB-200 adapter across all 21 languages with 10 prompts each (210 total tests). Auto-generates markdown report.

**Features:**
- 21 languages × 10 prompts = 210 tests
- Performance metrics (time, throughput)
- Quality assessment
- Auto-download results
- Performance charts (optional)

**Output:** `nllb_smoke_results_YYYYMMDD_HHMMSS.md`

---

## `postman_collection/`

**Purpose:** API testing collections  
**Importance:** ⭐⭐⭐ MODERATE - Developer tool

### `postman_collection/BLOOMZ_API_Collection.postman_collection.json`
**Purpose:** Postman collection for BLOOMZ System 2 API  
**Description:** Pre-configured API requests for testing all BLOOMZ endpoints (/generate, /tokenize, /kb/query, /tts/synthesize, etc.)

---

### `postman_collection/postman_multilang_21.json`
**Purpose:** Multilingual test cases  
**Description:** Test cases for 21 languages with various prompts.

---

## `results/`

**Purpose:** Smoke test results and reports  
**Importance:** ⭐⭐⭐⭐⭐ CRITICAL - Quality proof

### `results/nllb_smoke_results_20251023_121012.md`

**Lines:** 2,622  
**Purpose:** NLLB-200 adapter quality validation  
**Description:** Comprehensive smoke test results proving 90-95% accuracy across all 21 languages.

**Contents:**
- Executive summary (100% success, 0.51s avg, 210 tests)
- Per-language performance table
- Detailed test results (10 prompts × 21 languages)
- English prompts and target language outputs
- Timing for each translation
- Conclusion and recommendations

**Key Metrics:**
- Success Rate: 100% (210/210)
- Average Time: 0.51s per translation
- Throughput: ~2.0 translations/second
- Estimated Accuracy: 90-95%

---

## `rl/`

**Purpose:** Reinforcement Learning pipeline  
**Importance:** ⭐⭐⭐ MODERATE - Part of original task

### `rl/collect.py`

**Lines:** 258  
**Purpose:** RL episode collection and cloud upload  
**Description:** Collects interaction episodes (prompt/output/reward) and logs them to JSONL files or uploads to cloud (S3/HTTP).

**Features:**
- Episode generation with multilingual prompts
- Reward calculation (length, quality, diversity)
- Local logging to JSONL
- S3 upload (boto3)
- HTTP endpoint upload
- Adapter support

**Usage:**
```bash
# Local logging
python rl/collect.py --max_episodes 10 --out rl_runs/episodes.jsonl

# With S3 upload
python rl/collect.py --max_episodes 10 --s3-bucket my-bucket --s3-key episodes/
```

**Episode Format:**
```json
{
  "run_id": "uuid",
  "episode_index": 0,
  "timestamp": 1729512345.678,
  "prompt": "Translate to Hindi: Hello",
  "output": "नमस्ते",
  "reward": 0.85,
  "latency_s": 2.3,
  "meta": {"language": "hindi"}
}
```

---

### `rl/rl_config.yaml`

**Lines:** 48  
**Purpose:** RL pipeline configuration  
**Description:** Configuration for episode collection, rewards, and cloud uploads.

**Sections:**
- Model settings (base_model, adapter_path)
- Episode settings (max_episodes, env_name)
- Cloud settings (S3/HTTP upload)
- Reward weights (length, quality, diversity)
- Sample prompts (10 multilingual prompts)

---

## `rl_runs/`

**Purpose:** RL episode logs storage  
**Importance:** ⭐⭐ LOW - Auto-generated

**Description:** Empty folder where RL pipeline writes episode JSONL files. Files are auto-generated when running `rl/collect.py`.

**Example Files:**
- `episodes.jsonl`
- `test_episodes.jsonl`
- `multilingual_episodes.jsonl`

---

## `scripts/`

**Purpose:** Testing and utility scripts  
**Importance:** ⭐⭐⭐⭐ HIGH - Testing infrastructure

### `scripts/generate_smoke_results.py`

**Lines:** ~200  
**Purpose:** Generate BLOOMZ smoke test results  
**Description:** Automates smoke testing for BLOOMZ adapter across 10 multilingual prompts. Generates `docs/smoke_results.md`.

**Features:**
- Real-time progress indicators
- API server management
- Memory cleanup between requests
- Markdown report generation

**Usage:** `python scripts/generate_smoke_results.py`

---

### `scripts/generate_smoke_results_restart.py`

**Lines:** ~250  
**Purpose:** Restart-based smoke testing (Windows GPU workaround)  
**Description:** Specialized smoke test script that restarts the API server between each test to work around Windows GPU driver stability issues.

**Usage:** `python scripts/generate_smoke_results_restart.py`

---

### `scripts/test_complete_api.py`

**Lines:** ~230  
**Purpose:** Comprehensive BLOOMZ API tests  
**Description:** Tests all BLOOMZ API endpoints with various scenarios (health, adapter list, generation, multilingual, quality checks).

**Test Categories:**
- Health check
- Adapter listing
- Basic generation
- Multilingual generation (10 languages)
- Quality validation
- Performance metrics

**Usage:** `python scripts/test_complete_api.py`

---

### `scripts/test_simple_api.py`

**Lines:** ~150  
**Purpose:** Basic BLOOMZ API smoke tests  
**Description:** Simple API validation with a few test cases. Quick sanity check.

**Usage:** `python scripts/test_simple_api.py`

---

### `scripts/test_rl_pipeline.py`

**Lines:** ~150  
**Purpose:** RL pipeline validation  
**Description:** Tests RL episode collection, JSONL logging, and cloud upload functionality.

**Test Scenarios:**
1. Basic episode collection
2. Multilingual prompts
3. Cloud upload (if configured)
4. Reward calculation
5. Episode format validation

**Usage:** `python scripts/test_rl_pipeline.py`

---

## `src/`

**Purpose:** BLOOMZ System 2 source code  
**Importance:** ⭐⭐⭐⭐⭐ CRITICAL - Complete text generation system

**Description:** Full-featured multilingual text generation system with Knowledge Base, TTS, and NLP integrations.

---

### `src/api/`

**Purpose:** Main BLOOMZ API

#### `src/api/main.py`

**Lines:** 1,247  
**Purpose:** BLOOMZ FastAPI application  
**Description:** Complete API server with 15+ endpoints for text generation, tokenization, KB queries, TTS synthesis, and NLP processing.

**Endpoints:**
1. `POST /generate` - Generate text with BLOOMZ
2. `POST /tokenize` - Tokenize text
3. `POST /detokenize` - Detokenize IDs
4. `POST /language-detect` - Detect language
5. `POST /kb/query` - Query knowledge base
6. `GET /kb/stats` - Get KB statistics
7. `POST /tts/synthesize` - Text-to-speech
8. `POST /tts/batch` - Batch TTS
9. `POST /nlp/preprocess` - Preprocess text
10. `POST /nlp/sentiment` - Sentiment analysis
11. `POST /nlp/entities` - Entity extraction
12. `POST /conversation/create` - Create conversation
13. `POST /conversation/{id}/message` - Send message
14. `GET /conversation/{id}` - Get conversation
15. `GET /health` - Health check

**Features:**
- BLOOMZ-560M model with adapter support
- Multi-turn conversation management
- Caching layer
- Integration with KB/TTS/NLP services
- CUDA-safe fallback to CPU
- Comprehensive error handling

**Usage:** `python main.py` (entry point in root)

---

### `src/integration/`

**Purpose:** Integration modules for external services

#### `src/integration/multilingual_pipeline.py`

**Lines:** ~280  
**Purpose:** Complete integration pipeline  
**Description:** End-to-end pipeline connecting language detection, NLP processing, response generation, and TTS.

**Classes:**
- `CompleteMultilingualPipeline`: Sync pipeline
- `AsyncMultilingualPipeline`: Async pipeline

**Workflow:** Input → Language Detection → NLP → Generation → TTS → Response

---

#### `src/integration/tts_integration.py`

**Lines:** ~340  
**Purpose:** Vaani TTS integration  
**Description:** Client for Vaani TTS (Karthikeya) service. Handles text-to-speech synthesis across Indian languages.

**Features:**
- Voice management
- Language support queries
- Batch synthesis
- Audio streaming
- Format conversion (WAV, MP3, OGG)

---

#### `src/integration/nlp_integration.py`

**Lines:** ~420  
**Purpose:** Indigenous NLP integration  
**Description:** Client for Indigenous NLP (Nisarg) service. Provides text preprocessing, sentiment analysis, and entity extraction.

**Features:**
- Text preprocessing (normalization, cleaning)
- Sentiment analysis
- Named entity recognition
- POS tagging
- Language-specific processing

---

#### `src/integration/cached_pipeline.py`

**Lines:** ~200  
**Purpose:** Caching layer for pipeline  
**Description:** Adds multi-level caching to the multilingual pipeline for improved performance.

**Cache Levels:**
- Language detection cache
- Generation cache
- TTS cache

---

### `src/services/`

**Purpose:** Backend services

#### `src/services/kb_service.py`

**Lines:** 761  
**Purpose:** Knowledge Base FastAPI service  
**Description:** Standalone FastAPI service providing Q&A capabilities across 21 Indian languages.

**Knowledge Domains:**
- Geography (capitals, landmarks)
- Culture (festivals, traditions)
- History (events, figures)
- Science (concepts, discoveries)
- Technology (computing, AI)
- General knowledge

**Endpoints:**
- `POST /kb/query` - Query knowledge base
- `GET /kb/stats` - Get statistics
- `GET /health` - Health check

**Features:**
- 21-language support
- Query type classification
- Confidence scoring
- Source attribution

---

#### `src/services/knowledge_base.py`

**Lines:** ~400  
**Purpose:** Knowledge base data structures  
**Description:** Core KB logic, data structures, and query processing.

---

### `src/evaluation/`

**Purpose:** Evaluation metrics

#### `src/evaluation/metrics.py`

**Lines:** ~300  
**Purpose:** Evaluation metrics for models  
**Description:** BLEU, ROUGE, perplexity, and custom metrics for translation and generation quality.

---

### `src/models/`

**Purpose:** Model utilities

#### `src/models/tokenizer_integration.py`

**Lines:** ~200  
**Purpose:** Tokenizer integration utilities  
**Description:** Helper functions for tokenizer loading, configuration, and usage.

---

### `src/utils/`

**Purpose:** Utility functions

#### `src/utils/start_kb_service.py`

**Lines:** ~50  
**Purpose:** KB service launcher  
**Description:** Script to start the Knowledge Base service on port 8003.

---

## `test_prompts/`

**Purpose:** Test data for smoke testing  
**Importance:** ⭐⭐⭐ MODERATE - Test reference

### `test_prompts/prompts_10.json`

**Lines:** 109  
**Purpose:** 10 multilingual test prompts  
**Description:** Structured JSON with test prompts for smoke testing across 10 languages.

**Structure:**
```json
{
  "prompts": [
    {
      "id": 1,
      "language": "hindi",
      "language_code": "hi",
      "prompt": "अनुवाद करें: Hello friend",
      "expected_output_type": "Hindi translation",
      "category": "translation"
    },
    ...
  ]
}
```

**Languages:** Hindi, Bengali, Tamil, Telugu, Gujarati, Marathi, Urdu, Punjabi, Kannada, Malayalam

---

## `tests/`

**Purpose:** Integration tests  
**Importance:** ⭐⭐⭐ MODERATE - Test suite

### `tests/test_integration.py`

**Lines:** 381  
**Purpose:** Integration test suite  
**Description:** Comprehensive tests for all integration modules (`src/integration/`).

**Test Coverage:**
- Multilingual pipeline
- TTS integration
- NLP integration
- Cached pipeline
- Error handling
- Edge cases

**Usage:** `pytest tests/test_integration.py`

---

## `utils/`

**Purpose:** Utility scripts  
**Importance:** ⭐⭐ LOW - Development tools

### `utils/create_test_notebook.py`

**Lines:** ~200  
**Purpose:** Programmatically generate Colab test notebooks  
**Description:** Script to create/update `test_nllb_adapter_colab.ipynb` with Python code.

**Usage:** `python utils/create_test_notebook.py`

---

### `utils/flores200.py`

**Lines:** ~150  
**Purpose:** FLORES-200 dataset download  
**Description:** Script to download FLORES-200 dataset from HuggingFace Hub.

**Usage:** `python utils/flores200.py`

---

## `venv/`

**Purpose:** Python virtual environment  
**Importance:** ⭐⭐⭐⭐⭐ CRITICAL - Project dependencies

**Size:** ~7.38 GB  
**Description:** Isolated Python environment with all project dependencies installed.

**Key Directories:**
- `venv/Lib/site-packages/` - Installed packages
- `venv/Scripts/` - Executables (python.exe, pip.exe, uvicorn.exe)
- `venv/Include/` - C headers
- `venv/share/` - Shared data

**Usage:**
- Windows: `venv\Scripts\activate`
- Linux/Mac: `source venv/bin/activate`

---

## 📊 Summary Statistics

| Category | Count | Total Size |
|----------|-------|------------|
| **Adapters** | 2 | 138 MB |
| **Training Data** | 42 files (21 train + 21 val) | ~5 GB |
| **FLORES-200** | 408 files | 76 MB |
| **Notebooks** | 3 | ~100 KB |
| **Scripts** | 5 | ~50 KB |
| **Source Code** | 20+ files | ~100 KB |
| **Documentation** | 12 files | ~200 KB |
| **Virtual Env** | 10,000+ files | 7.38 GB |
| **Cache** | 40+ files | 100+ MB |
| **Total Project** | 1,000+ files | ~7.7 GB |

---

## 🎯 Critical Files Summary

**Must-Have for Production:**
1. ✅ `adapters/nllb_18languages_adapter/` - NLLB-200 adapter (90-95% accuracy)
2. ✅ `adapter_service/standalone_api.py` - Translation API
3. ✅ `requirements.txt` - All dependencies
4. ✅ `venv/` - Virtual environment

**Must-Have for BLOOMZ System:**
5. ✅ `src/api/main.py` - BLOOMZ API
6. ✅ `config/settings.py` - Configuration
7. ✅ `adapters/gurukul_lite/` - BLOOMZ adapter
8. ✅ `main.py` - Entry point

**Must-Have for Training:**
9. ✅ `notebooks/colab_train_nllb200.ipynb` - Training workflow
10. ✅ `flores200_dataset/` or `data/flores200_dataset.tar.gz` - Training data

**Must-Have for Validation:**
11. ✅ `results/nllb_smoke_results_*.md` - Quality proof
12. ✅ `notebooks/smoke_test_nllb_colab.ipynb` - Testing workflow

---

*Last Updated: October 23, 2025*  
*Total Documentation: 1,000+ files across 50+ folders*  
*Systems: NLLB-200 Translation (Primary), BLOOMZ Text Generation (Secondary)*

