# Lightweight Adapter + MCP Pipeline

## Overview
This project implements a lightweight multilingual adapter training pipeline with MCP (Model Context Protocol) streaming support, optimized for RTX 4050.

## Project Structure

```
├── adapter_service/           # Core adapter training and inference
│   ├── requirements-lite.txt # Lightweight dependencies
│   ├── train_adapt.py        # Streaming LoRA trainer (to be created)
│   ├── api.py                # FastAPI endpoints (to be created)
│   └── model_utils.py        # Model utilities (to be created)
├── rl/                       # Reinforcement Learning pipeline
│   ├── collect.py            # Episode collection (to be created)
│   └── upload_helper.py      # Cloud upload utilities (to be created)
├── test_prompts/             # Test prompts for smoke testing
│   └── prompts_10.json      # 10 multilingual test prompts (to be created)
├── mcp_connectors.yml        # MCP data source configuration
├── adapter_config.yaml       # Adapter training configuration
└── smoke_results.md         # Smoke test results (to be created)
```

## Configuration Files

### mcp_connectors.yml
Defines remote data sources for streaming multilingual corpora:
- Hugging Face datasets
- S3/Cloud storage
- Qdrant vector database
- HTTP API sources

### adapter_config.yaml
Configuration for lightweight fine-tuning:
- LoRA parameters optimized for RTX 4050
- 8-bit quantization settings
- Training parameters for streaming data

## Next Steps

1. **Implement streaming data loader** in `adapter_service/train_adapt.py`
2. **Create FastAPI endpoints** in `adapter_service/api.py`
3. **Add RL pipeline** in `rl/collect.py`
4. **Create smoke tests** with 10 multilingual prompts

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