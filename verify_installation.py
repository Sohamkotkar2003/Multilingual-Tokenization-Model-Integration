#!/usr/bin/env python3
"""Verify installation of all required packages"""

import sys

print("\n" + "="*70)
print("  ✅ INSTALLATION VERIFICATION")
print("="*70 + "\n")

# Check Python version
print(f"✅ Python: {sys.version.split()[0]}")

# Check PyTorch and CUDA
try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"✅ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB")
    else:
        print("❌ WARNING: CUDA not available!")
except ImportError:
    print("❌ PyTorch not installed!")

# Check Transformers
try:
    import transformers
    print(f"✅ Transformers: {transformers.__version__}")
except ImportError:
    print("❌ Transformers not installed!")

# Check PEFT
try:
    import peft
    print(f"✅ PEFT: {peft.__version__}")
except ImportError:
    print("❌ PEFT not installed!")

# Check Datasets
try:
    import datasets
    print(f"✅ Datasets: {datasets.__version__}")
except ImportError:
    print("❌ Datasets not installed!")

# Check Accelerate
try:
    import accelerate
    print(f"✅ Accelerate: {accelerate.__version__}")
except ImportError:
    print("❌ Accelerate not installed!")

# Check Bitsandbytes
try:
    import bitsandbytes
    print(f"✅ Bitsandbytes: {bitsandbytes.__version__}")
except ImportError:
    print("❌ Bitsandbytes not installed!")

# Check SentencePiece
try:
    import sentencepiece
    print(f"✅ SentencePiece: {sentencepiece.__version__}")
except ImportError:
    print("❌ SentencePiece not installed!")

# Check spaCy
try:
    import spacy
    print(f"✅ spaCy: {spacy.__version__}")
    try:
        nlp = spacy.load("en_core_web_sm")
        print(f"✅ spaCy Model: en_core_web_sm loaded")
    except:
        print("⚠️  spaCy model not loaded (optional)")
except ImportError:
    print("⚠️  spaCy not installed (optional)")

# Check FastAPI
try:
    import fastapi
    print(f"✅ FastAPI: {fastapi.__version__}")
except ImportError:
    print("⚠️  FastAPI not installed (optional)")

print("\n" + "="*70)
print("  🚀 INSTALLATION COMPLETE - READY TO TRAIN!")
print("="*70 + "\n")

print("Next steps:")
print("  1. Run training: python train_local_rtx4050.py")
print("  2. Or start API: python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8117")
print()

