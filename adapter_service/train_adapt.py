#!/usr/bin/env python3
"""
Streaming LoRA Adapter Training Script

This script trains lightweight LoRA adapters on BLOOMZ-560M using streaming datasets
from MCP sources (or local fallback). Optimized for RTX 4050 (6GB VRAM).

Key Features:
- MCP streaming support (no large downloads)
- 8-bit quantization for memory efficiency
- LoRA adapters (only ~1% parameters trained)
- Proven configuration from successful fine_tune.py

Based on working training script with verified settings that completed successfully.

Usage:
    python adapter_service/train_adapt.py --config adapter_config.yaml --max-samples 1000

Author: Based on src/training/fine_tune.py (which successfully completed training)
"""

import os
import sys
import logging
import argparse
import gc
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType

# Add parent directory to path for MCP imports
sys.path.insert(0, str(Path(__file__).parent))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Platform detection
import platform
IS_WINDOWS = platform.system() == "Windows"


def clear_gpu_memory():
    """Clear GPU memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    logger.info("GPU memory cleared")


def check_gpu_usage():
    """Check and log GPU usage"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"Memory - Allocated: {memory_allocated:.2f} GB | Reserved: {memory_reserved:.2f} GB | Total: {memory_total:.2f} GB")
        return True
    else:
        logger.info("CUDA not available - using CPU")
        return False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    logger.info(f"Loading configuration from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("Configuration loaded successfully")
    return config


def load_streaming_data(max_samples: int = 1000):
    """
    Load training data using MCP streaming or local fallback
    
    Args:
        max_samples: Maximum samples to load
        
    Returns:
        train_dataset, eval_dataset
    """
    logger.info(f"Loading training data (max {max_samples} samples)...")
    
    # Try MCP streaming first
    try:
        from mcp_streaming import MCPDataLoader
        
        loader = MCPDataLoader("mcp_connectors.yml")
        
        train_texts = []
        logger.info("Attempting to stream from MCP sources...")
        
        for i, sample in enumerate(loader.stream("multilingual_corpus", max_samples=max_samples)):
            if i >= max_samples:
                break
            
            text = sample.get("text", "").strip()
            if len(text) > 10:  # Filter out very short texts
                train_texts.append(text)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Loaded {i + 1} samples...")
        
        logger.info(f"✅ Loaded {len(train_texts)} training samples via MCP streaming")
        
    except Exception as e:
        logger.warning(f"MCP streaming failed: {e}")
        logger.info("Falling back to local data loading...")
        
        # Fallback to local files
        train_texts = load_local_data(max_samples)
    
    # Create validation split (10% of training data)
    split_idx = int(len(train_texts) * 0.9)
    eval_texts = train_texts[split_idx:]
    train_texts = train_texts[:split_idx]
    
    logger.info(f"Training samples: {len(train_texts)}")
    logger.info(f"Validation samples: {len(eval_texts)}")
    
    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": eval_texts}) if eval_texts else None
    
    return train_dataset, eval_dataset


def load_local_data(max_samples: int = 1000):
    """Load data from local training files"""
    logger.info("Loading from local data/training/ directory...")
    
    train_texts = []
    data_dir = Path("data/training")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Training data directory not found: {data_dir}")
    
    # Load from all .txt files
    txt_files = list(data_dir.glob("*.txt"))
    logger.info(f"Found {len(txt_files)} training files")
    
    samples_per_file = max(1, max_samples // len(txt_files))
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            # Filter and sample
            filtered = [line.strip() for line in lines if len(line.strip()) > 10]
            
            if len(filtered) > samples_per_file:
                import random
                random.seed(42)
                filtered = random.sample(filtered, samples_per_file)
            
            train_texts.extend(filtered)
            logger.info(f"Loaded {len(filtered)} samples from {txt_file.name}")
            
        except Exception as e:
            logger.warning(f"Failed to load {txt_file}: {e}")
            continue
    
    logger.info(f"✅ Loaded {len(train_texts)} total samples from local files")
    return train_texts


def create_model_and_tokenizer(config: Dict[str, Any]):
    """
    Create model and tokenizer with quantization and LoRA
    
    This uses PROVEN settings from fine_tune.py that successfully completed training
    """
    model_config = config.get("model", {})
    lora_config_dict = config.get("lora", {})
    
    base_model = model_config.get("base_model", "bigscience/bloomz-560m")
    use_8bit = model_config.get("use_8bit", True)
    
    logger.info(f"Loading model: {base_model}")
    logger.info(f"8-bit quantization: {'Enabled' if use_8bit else 'Disabled'}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Added EOS token as padding token")
    
    # Configure quantization (PROVEN SETTINGS from fine_tune.py)
    quantization_config = None
    if use_8bit and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,  # Proven setting
            llm_int8_has_fp16_weight=False,  # Proven setting
        )
        logger.info("🔧 Using 8-bit quantization (proven configuration)")
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if quantization_config else torch.float32,
    )
    
    logger.info("Model loaded successfully")
    check_gpu_usage()
    
    # Apply LoRA configuration (PROVEN SETTINGS for BLOOM from fine_tune.py)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_config_dict.get("r", 8),  # Proven: 8 works well
        lora_alpha=lora_config_dict.get("lora_alpha", 16),  # Proven: 16
        lora_dropout=lora_config_dict.get("lora_dropout", 0.05),  # Proven: 0.05
        # CRITICAL: Use proven BLOOM target modules from fine_tune.py
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    )
    
    logger.info("Applying LoRA adapters with PROVEN configuration...")
    logger.info(f"  Rank: {lora_config.r}")
    logger.info(f"  Alpha: {lora_config.lora_alpha}")
    logger.info(f"  Dropout: {lora_config.lora_dropout}")
    logger.info(f"  Target modules: {lora_config.target_modules}")
    
    model = get_peft_model(model, lora_config)
    logger.info("✅ LoRA adapters applied successfully")
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    # Verify trainable parameters exist
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    if trainable_params == 0:
        raise ValueError("❌ No trainable parameters found! LoRA not applied correctly.")
    
    logger.info(f"✅ Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer


def train_adapter(config_path: str, max_samples: int = 1000):
    """
    Main training function
    
    Uses PROVEN settings from fine_tune.py that successfully completed training
    """
    logger.info("=" * 80)
    logger.info("🚀 Lightweight LoRA Adapter Training")
    logger.info("=" * 80)
    logger.info(f"Config: {config_path}")
    logger.info(f"Max samples: {max_samples}")
    logger.info(f"Platform: {'Windows' if IS_WINDOWS else 'Linux/Mac'}")
    logger.info("=" * 80)
    
    # Load configuration
    config = load_config(config_path)
    
    # Extract training configuration
    training_config = config.get("training", {})
    output_dir = training_config.get("output_dir", "adapters/gurukul_lite")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Clear GPU memory
    clear_gpu_memory()
    
    # Load model and tokenizer
    model, tokenizer = create_model_and_tokenizer(config)
    
    # Load training data
    train_dataset, eval_dataset = load_streaming_data(max_samples)
    
    # Tokenization function
    max_length = training_config.get("block_size", 512)
    
    logger.info(f"Tokenizing datasets (max_length={max_length})...")
    logger.info("Using manual tokenization to avoid Windows multiprocessing issues...")
    
    # Manual tokenization (avoids .map() issues on Windows)
    def tokenize_dataset_manual(dataset):
        tokenized_data = {
            "input_ids": [],
            "attention_mask": []
        }
        
        for i, example in enumerate(dataset):
            tokens = tokenizer(
                example["text"],
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None
            )
            tokenized_data["input_ids"].append(tokens["input_ids"])
            tokenized_data["attention_mask"].append(tokens["attention_mask"])
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Tokenized {i + 1} samples...")
        
        return Dataset.from_dict(tokenized_data)
    
    tokenized_train = tokenize_dataset_manual(train_dataset)
    logger.info(f"✅ Training dataset tokenized: {len(tokenized_train)} samples")
    
    tokenized_eval = None
    if eval_dataset:
        tokenized_eval = tokenize_dataset_manual(eval_dataset)
        logger.info(f"✅ Validation dataset tokenized: {len(tokenized_eval)} samples")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM
        pad_to_multiple_of=8,  # GPU optimization
        return_tensors="pt",
    )
    
    # Training arguments (PROVEN SETTINGS from fine_tune.py)
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=training_config.get("num_train_epochs", 3),
        max_steps=training_config.get("max_steps", -1),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
        warmup_steps=training_config.get("warmup_steps", 100),
        learning_rate=training_config.get("learning_rate", 2e-4),
        logging_steps=training_config.get("logging_steps", 10),
        save_steps=training_config.get("save_steps", 100),
        eval_steps=training_config.get("eval_steps", 50) if eval_dataset else None,
        # CRITICAL PROVEN SETTINGS:
        fp16=False,  # Disable FP16 to avoid issues (PROVEN)
        dataloader_num_workers=0 if IS_WINDOWS else 2,  # CRITICAL: 0 on Windows! (PROVEN)
        dataloader_drop_last=True,
        dataloader_pin_memory=False,  # Save GPU memory
        report_to=None,  # No wandb/tensorboard
        gradient_checkpointing=False,  # Disabled with PEFT (PROVEN)
        save_total_limit=2,  # Save disk space
        max_grad_norm=1.0,  # Gradient clipping
        save_strategy="steps",
        eval_strategy="steps" if eval_dataset else "no",
        load_best_model_at_end=True if eval_dataset else False,
        remove_unused_columns=True,
    )
    
    logger.info("Training configuration:")
    logger.info(f"  Epochs: {training_args.num_train_epochs}")
    logger.info(f"  Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  Learning rate: {training_args.learning_rate}")
    logger.info(f"  Dataloader workers: {training_args.dataloader_num_workers} (CRITICAL for Windows!)")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )
    
    # Add progress callback
    class ProgressCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 10 == 0:
                current_loss = state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'
                logger.info(f"Step {state.global_step}/{state.max_steps} - Loss: {current_loss}")
    
    trainer.add_callback(ProgressCallback())
    
    # Clear memory before training
    clear_gpu_memory()
    check_gpu_usage()
    
    # Train!
    logger.info("=" * 80)
    logger.info("🎯 STARTING TRAINING...")
    logger.info("=" * 80)
    
    try:
        trainer.train()
        logger.info("✅ Training completed successfully!")
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"❌ CUDA out of memory: {e}")
        logger.info("Try reducing batch_size or max_length in config")
        clear_gpu_memory()
        raise
        
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        clear_gpu_memory()
        raise
    
    # Save adapter
    logger.info(f"Saving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("=" * 80)
    logger.info("🎉 TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Adapter saved to: {output_dir}")
    logger.info("You can now use this adapter with the API!")
    
    return output_dir


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train LoRA adapter with MCP streaming")
    parser.add_argument(
        "--config",
        type=str,
        default="adapter_config.yaml",
        help="Path to adapter configuration file"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum training samples to use"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    try:
        output_path = train_adapter(args.config, args.max_samples)
        print(f"\n✅ SUCCESS! Adapter saved to: {output_path}")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ TRAINING FAILED: {e}")
        sys.exit(1)

