#!/usr/bin/env python3
"""
Local Training Script for Single Language Adapters

Purpose:
- Train LoRA adapters for new languages on your laptop (RTX 4050)
- Optimized for 8GB VRAM with 8-bit quantization
- Run overnight (6-8 hours per language)

Usage:
    python train_single_language_local.py --language sinhala
    python train_single_language_local.py --language tibetan --epochs 3
    
Supported languages:
    sinhala, tibetan, dzongkha, pashto, dari, vietnamese, thai, burmese
"""

import sys
import io
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Force UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# Language configuration
LANGUAGE_CONFIG = {
    "sinhala": {
        "data_files": ["data/training_new/sinhala_wiki.jsonl"],
        "code": "si",
        "script": "Sinhala",
        "samples": 500
    },
    "tibetan": {
        "data_files": ["data/training_new/tibetan_wiki.jsonl"],
        "code": "bo",
        "script": "Tibetan",
        "samples": 500
    },
    "dzongkha": {
        "data_files": ["data/training_new/dzongkha_wiki.jsonl"],
        "code": "dz",
        "script": "Tibetan",
        "samples": 500
    },
    "pashto": {
        "data_files": ["data/training_new/pashto_wiki.jsonl", "data/training_new/pashto_news.jsonl"],
        "code": "ps",
        "script": "Arabic",
        "samples": 509
    },
    "dari": {
        "data_files": ["data/training_new/dari_news.jsonl"],
        "code": "prs",
        "script": "Arabic",
        "samples": 29
    },
    "vietnamese": {
        "data_files": ["data/training_new/vietnamese_wiki.jsonl", "data/training_new/vietnamese_news.jsonl"],
        "code": "vi",
        "script": "Latin",
        "samples": 340
    },
    "thai": {
        "data_files": ["data/training_new/thai_wiki.jsonl", "data/training_new/thai_news.jsonl"],
        "code": "th",
        "script": "Thai",
        "samples": 530
    },
    "burmese": {
        "data_files": ["data/training_new/burmese_wiki.jsonl"],
        "code": "my",
        "script": "Myanmar",
        "samples": 366
    },
}

# Model configuration
BASE_MODEL = "bigscience/bloomz-560m"
MAX_LENGTH = 256


def load_training_data(language: str, max_samples: int = None):
    """Load and prepare training data from JSONL files"""
    
    config = LANGUAGE_CONFIG[language]
    print(f"\n📂 Loading data for {language.title()}...")
    
    all_texts = []
    
    for data_file in config['data_files']:
        file_path = Path(data_file)
        if not file_path.exists():
            print(f"   ⚠️  File not found: {data_file}")
            continue
        
        print(f"   Reading: {data_file}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    text = item.get('text', item.get('content', ''))
                    if text and len(text) > 100:  # Skip very short texts
                        all_texts.append(text)
                except json.JSONDecodeError:
                    continue
    
    if max_samples:
        all_texts = all_texts[:max_samples]
    
    print(f"   ✅ Loaded {len(all_texts)} samples")
    print(f"   Average length: {sum(len(t) for t in all_texts) / len(all_texts):.0f} chars")
    
    return all_texts


def create_training_prompts(texts: list, language: str):
    """Create instruction-based training prompts"""
    
    prompts = []
    
    for text in texts:
        # Create language-locked prompt
        prompt = f"{text}\n\nकृपया केवल {language} में उपरोक्त विषय पर विस्तार से लिखें:"
        prompts.append(prompt)
    
    return prompts


def prepare_dataset(texts: list, tokenizer, language: str):
    """Tokenize and prepare dataset"""
    
    print(f"\n🔧 Preparing dataset...")
    
    # Create prompts
    prompts = create_training_prompts(texts, language)
    
    # Tokenize
    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding=False,
        max_length=MAX_LENGTH,
        return_tensors=None
    )
    
    # Create dataset
    dataset = Dataset.from_dict({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask']
    })
    
    # Split train/validation (90/10)
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    print(f"   ✅ Train samples: {len(split_dataset['train'])}")
    print(f"   ✅ Validation samples: {len(split_dataset['test'])}")
    
    return split_dataset['train'], split_dataset['test']


def setup_model_and_tokenizer():
    """Load model with 8-bit quantization for RTX 4050"""
    
    print(f"\n🤖 Loading BLOOMZ-560m with 8-bit quantization...")
    
    # 8-bit quantization config (for 8GB VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
        bnb_8bit_use_double_quant=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    print(f"   ✅ Model loaded with 8-bit quantization")
    
    return model, tokenizer


def train_adapter(language: str, epochs: int = 2, max_samples: int = None):
    """Train adapter for a single language"""
    
    print("="*80)
    print(f"🚀 TRAINING ADAPTER: {language.upper()}")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")
    print("="*80)
    
    # Load data
    texts = load_training_data(language, max_samples)
    
    # Setup model
    model, tokenizer = setup_model_and_tokenizer()
    
    # Prepare dataset
    train_dataset, val_dataset = prepare_dataset(texts, tokenizer, language)
    
    # Output directory
    output_dir = Path(f"adapters/{language}_lite")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training arguments - OPTIMIZED FOR RTX 4050 (8GB VRAM)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,  # Small batch for 8GB VRAM
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Effective batch = 1*16 = 16
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=50,
        logging_steps=25,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True,  # Mixed precision for speed
        optim="adamw_torch",
        report_to="none",
        dataloader_pin_memory=False,  # Reduce memory usage
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    # Calculate estimated time
    total_steps = len(train_dataset) // (1 * 16) * epochs
    estimated_hours = total_steps * 2.5 / 3600  # ~2.5 seconds per step on RTX 4050
    
    print(f"\n⏱️  Training Configuration:")
    print(f"   Total steps: ~{total_steps}")
    print(f"   Estimated time: {estimated_hours:.1f} - {estimated_hours * 1.3:.1f} hours")
    print(f"   Should finish by: {datetime.now().strftime('%H:%M')} + {estimated_hours:.0f}h")
    print(f"\n🚀 Starting training...\n")
    
    # Train!
    trainer.train()
    
    # Save final model
    print(f"\n💾 Saving adapter...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save metadata
    metadata = {
        "language": language,
        "language_code": LANGUAGE_CONFIG[language]["code"],
        "script": LANGUAGE_CONFIG[language]["script"],
        "samples": len(texts),
        "epochs": epochs,
        "base_model": BASE_MODEL,
        "training_date": datetime.now().isoformat(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Adapter saved to: {output_dir}/")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Train adapter for a single language")
    parser.add_argument(
        '--language', '-l',
        required=True,
        choices=list(LANGUAGE_CONFIG.keys()),
        help='Language to train'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=2,
        help='Number of training epochs (default: 2)'
    )
    parser.add_argument(
        '--max_samples', '-n',
        type=int,
        default=None,
        help='Maximum number of samples to use (default: all)'
    )
    
    args = parser.parse_args()
    
    # Verify data files exist
    config = LANGUAGE_CONFIG[args.language]
    missing_files = [f for f in config['data_files'] if not Path(f).exists()]
    if missing_files:
        print(f"❌ ERROR: Missing data files:")
        for f in missing_files:
            print(f"   - {f}")
        print(f"\nRun 'python scrape_all_languages.py' first!")
        sys.exit(1)
    
    # Train
    train_adapter(args.language, args.epochs, args.max_samples)


if __name__ == "__main__":
    main()

