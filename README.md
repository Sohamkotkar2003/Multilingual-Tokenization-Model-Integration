# Multilingual Tokenization Model Integration

A comprehensive system for multilingual tokenization and language model integration supporting Hindi, Sanskrit, Marathi, and English languages.

## 🚀 Features

- **Multilingual Support**: Hindi, Sanskrit, Marathi, and English
- **FastAPI Integration**: RESTful API with comprehensive endpoints
- **Knowledge Base Integration**: Custom KB service for Q&A functionality
- **Language Detection**: Automatic language detection and classification
- **Model Fine-tuning**: Support for fine-tuning language models
- **Training Scripts**: Complete training pipeline with Colab support
- **Data Processing**: Tools for data extraction and preprocessing

## 📁 Project Structure

```
├── src/                          # Source code
│   ├── api/                      # FastAPI application
│   │   └── main.py              # Main API application
│   ├── models/                   # Model integration
│   │   └── tokenizer_integration.py
│   ├── services/                 # Backend services
│   │   ├── knowledge_base.py    # KB integration service
│   │   └── kb_service.py        # Custom KB service
│   ├── training/                 # Training scripts
│   │   ├── fine_tune.py         # Model fine-tuning
│   │   ├── train_tokenizer.py   # Tokenizer training
│   │   ├── colab_training.py    # Colab training script
│   │   └── colab_training.ipynb # Colab notebook
│   ├── data_processing/          # Data processing utilities
│   │   ├── create_sample_data.py
│   │   ├── data_splitter.py
│   │   └── extraction_scripts/
│   └── utils/                    # Utility functions
│       └── start_kb_service.py
├── config/                       # Configuration files
│   └── settings.py              # Application settings
├── data/                        # Data directory
│   ├── training/                # Training data
│   └── validation/              # Validation data
├── model/                       # Model files
├── tests/                       # Test files
├── docs/                        # Documentation
├── logs/                        # Log files
├── main.py                      # Main entry point
└── requirements.txt             # Dependencies
```

## 🛠️ Installation

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

## 🚀 Quick Start

### 1. Start the Main API

```bash
python main.py
```

The API will be available at `http://127.0.0.1:8000`

### 2. Start the Knowledge Base Service (Optional)

```bash
python src/utils/start_kb_service.py
```

The KB service will be available at `http://127.0.0.1:8001`

### 3. Create Sample Data (if needed)

```bash
python src/data_processing/create_sample_data.py
```

## 📚 API Endpoints

### Core Endpoints

- `GET /` - Health check
- `POST /tokenize` - Tokenize text
- `POST /generate` - Generate text
- `POST /language-detect` - Detect language

### Knowledge Base Endpoints

- `POST /qa` - Q&A with knowledge base
- `POST /multilingual-conversation` - Multilingual conversation
- `GET /conversation/{session_id}/history` - Get conversation history
- `DELETE /conversation/{session_id}` - Clear conversation history

### Utility Endpoints

- `GET /health` - Detailed health check
- `GET /stats` - API statistics
- `GET /config` - Configuration information

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

## 🌐 Supported Languages

- **Hindi** (हिंदी)
- **Sanskrit** (संस्कृत)
- **Marathi** (मराठी)
- **English**

## 🔧 Development

### Project Structure Guidelines

- `src/api/` - FastAPI application and endpoints
- `src/models/` - Model integration and tokenization
- `src/services/` - Backend services and integrations
- `src/training/` - Training scripts and utilities
- `src/data_processing/` - Data extraction and preprocessing
- `src/utils/` - Utility functions and helpers
- `config/` - Configuration files
- `tests/` - Test files
- `docs/` - Documentation

### Adding New Features

1. Create appropriate modules in the `src/` directory
2. Update imports in `main.py` if needed
3. Add tests in the `tests/` directory
4. Update documentation

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For support and questions, please open an issue in the repository.
