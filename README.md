# Multilingual Tokenization Model Integration v2.0

A comprehensive system for multilingual tokenization and language model integration supporting **20+ Indian languages** with advanced MCP (Multi-Corpus Preprocessing) capabilities and full integration with Indigenous NLP + Vaani TTS.

## 🚀 Features

### Core Capabilities
- **20+ Indian Languages**: Hindi, Sanskrit, Marathi, English, Tamil, Telugu, Kannada, Bengali, Gujarati, Punjabi, Odia, Malayalam, Assamese, Urdu, Nepali, Kashmiri, Konkani, Manipuri, Sindhi, Bodo, Dogri, Maithili, Santali
- **MCP Pipeline**: Advanced Multi-Corpus Preprocessing for robust data handling across scripts
- **Enhanced Language Detection**: Unicode range + keyword-based detection for all supported languages
- **FastAPI Integration**: RESTful API with comprehensive endpoints for 20+ languages
- **Knowledge Base Integration**: Custom KB service for Q&A functionality
- **Model Fine-tuning**: Support for fine-tuning language models on multilingual data

### Advanced Features
- **Docker Deployment**: Complete containerization with GPU support
- **Evaluation Metrics**: BLEU/ROUGE, perplexity, tokenization accuracy
- **Language Switching**: Mid-conversation language switching validation
- **TTS Integration**: Ready for Vaani TTS integration
- **NLP Integration**: Compatible with Indigenous NLP composer
- **Caching**: Redis-based caching for improved performance
- **Load Balancing**: Nginx configuration for horizontal scaling

## 📁 Project Structure

```
├── src/                          # Source code
│   ├── api/                      # FastAPI application
│   │   └── main.py              # Enhanced API with 20+ language support
│   ├── models/                   # Model integration
│   │   └── tokenizer_integration.py
│   ├── services/                 # Backend services
│   │   ├── knowledge_base.py    # KB integration service
│   │   └── kb_service.py        # Custom KB service
│   ├── training/                 # Training scripts
│   │   ├── fine_tune.py         # Model fine-tuning
│   │   ├── train_tokenizer.py   # Original tokenizer training
│   │   ├── train_multilingual_tokenizer.py  # NEW: 20+ language training
│   │   ├── colab_training.py    # Colab training script
│   │   └── colab_training.ipynb # Colab notebook
│   ├── data_processing/          # Data processing utilities
│   │   ├── create_sample_data.py
│   │   ├── data_splitter.py
│   │   ├── mcp_pipeline.py      # NEW: Multi-Corpus Preprocessing
│   │   ├── corpus_collector.py  # NEW: Corpus collection for 20+ languages
│   │   └── extraction_scripts/
│   ├── evaluation/               # NEW: Evaluation metrics
│   │   └── metrics.py           # BLEU/ROUGE, perplexity, fluency evaluation
│   └── utils/                    # Utility functions
│       └── start_kb_service.py
├── config/                       # Configuration files
│   └── settings.py              # Enhanced settings for 20+ languages
├── data/                        # Data directory
│   ├── training/                # Training data for 20+ languages
│   └── validation/              # Validation data for 20+ languages
├── model/                       # Model files
├── tests/                       # Test files
├── docs/                        # Documentation
│   └── INTEGRATION_GUIDE.md    # NEW: Complete integration guide
├── logs/                        # Log files
├── cache/                       # NEW: Caching directory
├── main.py                      # Main entry point
├── requirements.txt             # Dependencies
├── Dockerfile                   # NEW: Docker configuration
├── docker-compose.yml           # NEW: Docker Compose setup
└── nginx.conf                   # NEW: Load balancer configuration
```

## 🛠️ Installation

### Option 1: Docker Deployment (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Multilingual-Tokenization-Model-Integration
   ```

2. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

   This will start:
   - Multilingual API on `http://localhost:8000`
   - Redis cache on `http://localhost:6379`
   - Nginx load balancer on `http://localhost:80`

### Option 2: Local Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Multilingual-Tokenization-Model-Integration
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional dependencies for 20+ languages**:
   ```bash
   pip install indic-nlp-library sacremoses nltk scikit-learn
   ```

## 🚀 Quick Start

### 1. Data Collection and Preprocessing

```bash
# Collect sample data for all 20+ languages
python src/data_processing/corpus_collector.py --sample-only

# Or collect from external sources (optional)
python src/data_processing/corpus_collector.py --external
```

### 2. Train Multilingual Tokenizer

```bash
# Train tokenizer for 20+ languages using MCP pipeline
python src/training/train_multilingual_tokenizer.py

# Or use existing processed data
python src/training/train_multilingual_tokenizer.py --skip-preprocessing --processed-file data/processed.txt
```

### 3. Start the API

```bash
python main.py
```

The API will be available at `http://127.0.0.1:8000`

### 4. Test Language Detection

```bash
# Test with different languages
curl -X POST "http://localhost:8000/language-detect" \
     -H "Content-Type: application/json" \
     -d '{"text": "नमस्ते, आप कैसे हैं?"}'

curl -X POST "http://localhost:8000/language-detect" \
     -H "Content-Type: application/json" \
     -d '{"text": "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?"}'
```

## 📚 API Endpoints

### Core Endpoints (Enhanced for 20+ Languages)

- `GET /` - Health check
- `POST /tokenize` - Tokenize text (supports all 20+ languages)
- `POST /generate` - Generate text (supports all 20+ languages)
- `POST /language-detect` - Detect language (enhanced for 20+ languages)

### Knowledge Base Endpoints

- `POST /qa` - Q&A with knowledge base (multilingual)
- `POST /multilingual-conversation` - Multilingual conversation with KB integration
- `GET /conversation/{session_id}/history` - Get conversation history
- `DELETE /conversation/{session_id}` - Clear conversation history

### Advanced Endpoints

- `POST /test-language-switching` - Test mid-conversation language switching
- `GET /health` - Detailed health check with component status
- `GET /stats` - Comprehensive API statistics including KB integration
- `GET /config` - Configuration information for all languages

### Example API Calls

```bash
# Language Detection
curl -X POST "http://localhost:8000/language-detect" \
     -H "Content-Type: application/json" \
     -d '{"text": "नमस्ते, आप कैसे हैं?"}'

# Text Generation
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, how are you?", "language": "english"}'

# Multilingual Conversation
curl -X POST "http://localhost:8000/multilingual-conversation" \
     -H "Content-Type: application/json" \
     -d '{"text": "भारत के बारे में बताइए", "language": "hindi"}'
```

## 🔧 Configuration

Edit `config/settings.py` to customize:

- Model paths and names
- API host and port
- Language detection thresholds
- Training parameters
- Logging configuration

## 🎯 Training

### Fine-tune a Model

```bash
python src/training/fine_tune.py
```

### Train a Tokenizer

```bash
python src/training/train_tokenizer.py
```

### Google Colab Training

Use the provided notebook: `src/training/colab_training.ipynb`

## 📊 Data Processing

### Split Data by Language

```bash
python src/data_processing/data_splitter.py
```

### Extract Data from Corpora

```bash
python src/data_processing/extraction_scripts/extract_english.py
```

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/
```

## 📖 Documentation

- **API Documentation**: Available at `http://127.0.0.1:8000/docs` when running
- **Colab Guide**: See `docs/colab_guide.md`
- **Postman Collection**: Available in `docs/postman_collection/`

## 🌐 Supported Languages (20+)

### Devanagari Script
- **Hindi** (हिंदी)
- **Sanskrit** (संस्कृत)
- **Marathi** (मराठी)
- **Nepali** (नेपाली)
- **Konkani** (कोंकणी)
- **Bodo** (बड़ो)
- **Dogri** (डोगरी)
- **Maithili** (मैथिली)

### South Indian Scripts
- **Tamil** (தமிழ்)
- **Telugu** (తెలుగు)
- **Kannada** (ಕನ್ನಡ)
- **Malayalam** (മലയാളം)

### Bengali Script
- **Bengali** (বাংলা)
- **Assamese** (অসমীয়া)

### Other Scripts
- **Gujarati** (ગુજરાતી)
- **Punjabi** (ਪੰਜਾਬੀ)
- **Odia** (ଓଡ଼ିଆ)
- **Urdu** (اردو)
- **Kashmiri** (کٲشُر)
- **Manipuri** (মৈতৈলোন)
- **Sindhi** (سنڌي)
- **Santali** (ᱥᱟᱱᱛᱟᱲᱤ)
- **English**

## 🆕 New Features in v2.0

### MCP (Multi-Corpus Preprocessing) Pipeline
- Robust data preprocessing for 20+ languages
- Unicode normalization for different scripts
- Language-specific cleaning and tokenization preparation
- Deduplication and noise removal
- Sentence segmentation for Indian languages

### Enhanced Language Detection
- Unicode range-based detection for all scripts
- Keyword-based language identification
- Support for mixed-language text
- Mid-conversation language switching validation

### Evaluation Metrics
- BLEU/ROUGE scores for text quality
- Perplexity for model performance
- Tokenization accuracy measurement
- Language-specific fluency evaluation
- Latency and performance metrics

### Docker Deployment
- Complete containerization with GPU support
- Docker Compose for easy deployment
- Nginx load balancer configuration
- Redis caching for improved performance

## 🔗 Integration with Indigenous NLP + Vaani TTS

### Complete Integration Pipeline
```
User Input → Language Detection → KB Query → Response Generation → TTS → Audio Output
```

### Integration Components
- **Indigenous NLP Composer (Nisarg)**: Text preprocessing and analysis
- **Vaani TTS (Karthikeya)**: Text-to-speech synthesis
- **Multilingual API**: Language detection, tokenization, and generation
- **Knowledge Base**: Q&A and conversation management

### Quick Integration Setup
```python
# Example integration code
from src.integration.multilingual_pipeline import CompleteMultilingualPipeline

pipeline = CompleteMultilingualPipeline()

# Process user input
result = pipeline.process_user_input(
    text="नमस्ते, भारत के बारे में बताइए",
    user_id="user123",
    session_id="session456"
)

print(f"Response: {result['text_response']}")
print(f"Language: {result['language']}")
print(f"Audio URL: {result['audio_url']}")
```

For detailed integration instructions, see [docs/INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md).

## 🧪 Testing and Evaluation

### Run Evaluation Metrics
```bash
# Evaluate all languages
python src/evaluation/metrics.py --languages hindi tamil telugu kannada

# Evaluate specific language
python src/evaluation/metrics.py --languages hindi --output hindi_evaluation.json
```

### Test Language Switching
```bash
# Test mid-conversation language switching
curl -X POST "http://localhost:8000/test-language-switching" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, how are you?", "session_id": "test123"}'
```

## 🚀 Deployment

### Production Deployment
```bash
# Build and deploy with Docker Compose
docker-compose -f docker-compose.prod.yml up -d

# Scale horizontally
docker-compose up --scale multilingual-api=3
```

### Cloud Deployment
- **AWS**: Use ECS with Fargate and Application Load Balancer
- **GCP**: Use Cloud Run with Cloud Load Balancing
- **Azure**: Use Container Instances with Application Gateway

## 🔧 Development

### Project Structure Guidelines

- `src/api/` - FastAPI application and endpoints
- `src/models/` - Model integration and tokenization
- `src/services/` - Backend services and integrations
- `src/training/` - Training scripts and utilities
- `src/data_processing/` - Data extraction and preprocessing
- `src/evaluation/` - Evaluation metrics and testing
- `src/utils/` - Utility functions and helpers
- `config/` - Configuration files
- `tests/` - Test files
- `docs/` - Documentation

### Adding New Languages

1. Add language to `SUPPORTED_LANGUAGES` in `config/settings.py`
2. Add Unicode ranges and keywords for detection
3. Add sample data in `corpus_collector.py`
4. Update MCP pipeline for language-specific processing
5. Test with evaluation metrics

## 📊 Performance Metrics

### Expected Performance
- **Language Detection**: < 100ms per request
- **Text Generation**: < 2s per request (256 tokens)
- **Tokenization**: < 50ms per request
- **Concurrent Requests**: 100+ requests/second
- **Memory Usage**: < 8GB with 4-bit quantization

### Monitoring
- Health checks: `GET /health`
- Statistics: `GET /stats`
- Metrics: `GET /metrics` (Prometheus format)

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run evaluation metrics
6. Submit a pull request

## 📞 Support

For support and questions:
- Open an issue in the repository
- Check the [Integration Guide](docs/INTEGRATION_GUIDE.md)
- Review the API documentation at `http://localhost:8000/docs`

## 🙏 Acknowledgments

- AI4Bharat for Indic NLP preprocessing best practices
- Hugging Face for transformer models and tokenizers
- SentencePiece for multilingual tokenization
- FastAPI for the web framework
- The open-source community for various libraries and tools
