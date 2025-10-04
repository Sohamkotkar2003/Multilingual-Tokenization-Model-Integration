"""
Multilingual Fine-tuning Training Script

This script fine-tunes a language model on multilingual data (Hindi, Sanskrit, Marathi, English)
without requiring any command-line arguments. All configuration is handled through variables
and settings.py.

Usage:
    python train.py                    # Normal training with caching
    python train.py --clear-cache      # Clear cached tokenized data and retrain

Features:
    - Automatic caching of tokenized datasets to avoid re-tokenization
    - Memory-optimized training for 6GB GPUs
    - Gradient checkpointing and accumulation
    - Automatic cache management

The fine-tuned model will be saved to 'mbart_finetuned/' directory and can be used
immediately with app.py for text generation.

Requirements:
    - Training data in data/training/ directory
    - Validation data in data/validation/ directory
    - Sufficient GPU memory for training
"""

import logging
import os
import gc
import pickle
import hashlib
import sys
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import settings

# Set environment variable for CUDA memory management (only if supported)
import platform
IS_WINDOWS = platform.system() == "Windows"
if not IS_WINDOWS:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Training Configuration Variables (No Command Line Args)
# =============================================================================

# Model Configuration
MODEL_NAME = settings.MODEL_NAME  # Use the same model as app.py
OUTPUT_DIR = "finetuned_generation_model"  # Output directory for fine-tuned model

# Training Parameters
EPOCHS = 3
BATCH_SIZE = 2  # Use larger batch with quantization
GRADIENT_ACCUMULATION_STEPS = 2  # Adjust based on batch size
WARMUP_STEPS = 500
LEARNING_RATE = 5e-5
FP16 = False  # Disable FP16 to avoid gradient scaling issues
GRADIENT_CHECKPOINTING = True  # Enable gradient checkpointing to save memory
USE_QUANTIZATION = True  # Enable 8-bit quantization for faster training
USE_PEFT = True  # Enable PEFT with LoRA for efficient fine-tuning

# Logging and Saving
LOGGING_STEPS = 100
SAVE_STEPS = 500  # Save every 500 steps
EVAL_STEPS = 500  # Evaluate every 500 steps

# Data Configuration
MAX_LENGTH = 512  # Reduced from 1024 to 512 to save memory per sample

# Cache Configuration
CACHE_DIR = "cache"
TOKENIZED_CACHE_DIR = os.path.join(CACHE_DIR, "tokenized")

def get_cache_key(tokenizer_name, max_length, data_files):
    """Generate a unique cache key based on tokenizer, max_length, and data files"""
    # Create a hash of the relevant parameters
    key_string = f"{tokenizer_name}_{max_length}_{sorted(data_files.items())}"
    return hashlib.md5(key_string.encode()).hexdigest()

def clear_gpu_memory():
    """Clear GPU memory and run garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def check_gpu_usage():
    """Check and log GPU usage"""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device)
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
        memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB, Total: {memory_total:.2f} GB")
        return True
    else:
        logger.info("CUDA not available - using CPU")
        return False

def save_tokenized_cache(tokenized_train, tokenized_eval, cache_key):
    """Save tokenized datasets to cache"""
    os.makedirs(TOKENIZED_CACHE_DIR, exist_ok=True)
    
    train_cache_path = os.path.join(TOKENIZED_CACHE_DIR, f"train_{cache_key}.pkl")
    eval_cache_path = os.path.join(TOKENIZED_CACHE_DIR, f"eval_{cache_key}.pkl")
    
    logger.info(f"Saving tokenized training data to cache: {train_cache_path}")
    with open(train_cache_path, 'wb') as f:
        pickle.dump(tokenized_train, f)
    
    if tokenized_eval:
        logger.info(f"Saving tokenized validation data to cache: {eval_cache_path}")
        with open(eval_cache_path, 'wb') as f:
            pickle.dump(tokenized_eval, f)
    
    logger.info("✅ Tokenized datasets cached successfully!")

def load_tokenized_cache(cache_key):
    """Load tokenized datasets from cache"""
    train_cache_path = os.path.join(TOKENIZED_CACHE_DIR, f"train_{cache_key}.pkl")
    eval_cache_path = os.path.join(TOKENIZED_CACHE_DIR, f"eval_{cache_key}.pkl")
    
    if not os.path.exists(train_cache_path):
        return None, None
    
    logger.info(f"Loading tokenized training data from cache: {train_cache_path}")
    with open(train_cache_path, 'rb') as f:
        tokenized_train = pickle.load(f)
    
    tokenized_eval = None
    if os.path.exists(eval_cache_path):
        logger.info(f"Loading tokenized validation data from cache: {eval_cache_path}")
        with open(eval_cache_path, 'rb') as f:
            tokenized_eval = pickle.load(f)
    
    logger.info("✅ Tokenized datasets loaded from cache successfully!")
    return tokenized_train, tokenized_eval

def clear_old_caches():
    """Clear old cache files to save disk space"""
    if not os.path.exists(TOKENIZED_CACHE_DIR):
        return
    
    cache_files = [f for f in os.listdir(TOKENIZED_CACHE_DIR) if f.endswith('.pkl')]
    if len(cache_files) > 5:  # Keep only 5 most recent caches
        cache_files.sort(key=lambda x: os.path.getmtime(os.path.join(TOKENIZED_CACHE_DIR, x)))
        files_to_remove = cache_files[:-5]
        
        for file in files_to_remove:
            file_path = os.path.join(TOKENIZED_CACHE_DIR, file)
            os.remove(file_path)
            logger.info(f"Removed old cache file: {file}")

def clear_all_caches():
    """Clear all cached tokenized data (useful for debugging or when data changes)"""
    if not os.path.exists(TOKENIZED_CACHE_DIR):
        logger.info("No cache directory found.")
        return
    
    cache_files = [f for f in os.listdir(TOKENIZED_CACHE_DIR) if f.endswith('.pkl')]
    for file in cache_files:
        file_path = os.path.join(TOKENIZED_CACHE_DIR, file)
        os.remove(file_path)
        logger.info(f"Removed cache file: {file}")
    
    logger.info(f"✅ Cleared {len(cache_files)} cache files")

def get_safe_max_length(tokenizer, default_max_length=512, task="training"):
    """
    Get a safe max_length value from tokenizer, avoiding overflow issues
    
    Args:
        tokenizer: The tokenizer object
        default_max_length: Default length to use (1024 for BLOOM)
        task: Either "training" or "inference" - inference can use longer contexts
    """
    try:
        model_max_length = tokenizer.model_max_length
        
        # BLOOM models have very large model_max_length values that cause overflow
        if model_max_length > 100000:
            if task == "inference":
                return settings.INFERENCE_MAX_LENGTH
            return default_max_length
            
        # Use the model's actual max length if reasonable
        return min(model_max_length, default_max_length)
        
    except (AttributeError, OverflowError):
        return default_max_length

def load_local_training_data():
    """
    Load training and validation data from local files
    Returns: train_dataset, eval_dataset
    """
    logger.info("Loading local training data...")
    
    # Load training data from all language files
    train_texts = []
    for lang, filename in settings.CORPUS_FILES.items():
        filepath = os.path.join(settings.TRAINING_DATA_PATH, filename)
        if os.path.exists(filepath):
            logger.info(f"Loading {lang} training data from {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Filter out empty lines and very short lines
                filtered_lines = [line.strip() for line in lines if len(line.strip()) > 10]
                train_texts.extend(filtered_lines)
                logger.info(f"Loaded {len(filtered_lines)} {lang} training samples")
        else:
            logger.warning(f"Training file not found: {filepath}")
    
    # Load validation data
    eval_texts = []
    val_files = {
        "hindi": "hi_val.txt",
        "sanskrit": "sa_val.txt", 
        "marathi": "mr_val.txt",
        "english": "en_val.txt"
    }
    
    for lang, filename in val_files.items():
        filepath = os.path.join(settings.VALIDATION_DATA_PATH, filename)
        if os.path.exists(filepath):
            logger.info(f"Loading {lang} validation data from {filepath}")
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Filter out empty lines and very short lines
                filtered_lines = [line.strip() for line in lines if len(line.strip()) > 10]
                eval_texts.extend(filtered_lines)
                logger.info(f"Loaded {len(filtered_lines)} {lang} validation samples")
        else:
            logger.warning(f"Validation file not found: {filepath}")
    
    logger.info(f"Total training samples: {len(train_texts)}")
    logger.info(f"Total validation samples: {len(eval_texts)}")
    
    # Create datasets
    train_dataset = Dataset.from_dict({"text": train_texts})
    eval_dataset = Dataset.from_dict({"text": eval_texts}) if eval_texts else None
    
    return train_dataset, eval_dataset

def main():
    """Main training function - no command line arguments needed"""
    
    # Check for no-quantization flag
    global USE_QUANTIZATION, USE_PEFT, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS
    if len(sys.argv) > 1 and sys.argv[1] == "--no-quantization":
        USE_QUANTIZATION = False
        USE_PEFT = False
        BATCH_SIZE = 1
        GRADIENT_ACCUMULATION_STEPS = 4
    
    # Print startup info
    print("🔧 Development environment detected")
    print("=" * 80)
    print(f"🚀 Multilingual Fine-tuning Training v{settings.API_VERSION}")
    print("=" * 80)
    print(f"🤖 Model: {MODEL_NAME}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    print(f"🌐 Languages: {', '.join(settings.SUPPORTED_LANGUAGES)}")
    print(f"📊 Training Epochs: {EPOCHS}")
    print(f"📦 Batch Size: {BATCH_SIZE}")
    print(f"🎯 Learning Rate: {LEARNING_RATE}")
    print(f"💾 Cache Directory: {TOKENIZED_CACHE_DIR}")
    print(f"🔧 Quantization: {'Enabled (8-bit)' if USE_QUANTIZATION else 'Disabled'}")
    print(f"🔧 PEFT/LoRA: {'Enabled' if USE_PEFT else 'Disabled'}")
    print("=" * 80)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Clear GPU memory before loading model
    clear_gpu_memory()
    
    # Load model and tokenizer
    logger.info(f"Loading tokenizer and model from {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Configure quantization for faster training
    quantization_config = None
    if USE_QUANTIZATION and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        logger.info("🔧 Using 8-bit quantization for faster training")
    
    # Load model with memory optimizations
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if quantization_config else torch.float32,
    )

    for name, module in model.named_modules():
        print(name)
    
    # Check GPU usage and ensure model is on GPU
    check_gpu_usage()
    
    # Ensure model is on GPU if available (skip if using quantization with device_map)
    if torch.cuda.is_available() and not quantization_config:
        model = model.cuda()
        logger.info(f"Model moved to GPU: {torch.cuda.get_device_name()}")
        check_gpu_usage()
    elif quantization_config:
        logger.info(f"Model loaded with quantization on GPU: {torch.cuda.get_device_name()}")
        check_gpu_usage()
    else:
        logger.info("Model loaded on CPU")
    
    # Enable gradient checkpointing to save memory (but not with PEFT)
    if GRADIENT_CHECKPOINTING and not USE_PEFT:
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing to save memory")
    elif USE_PEFT:
        logger.info("Skipping gradient checkpointing for PEFT training")
    
    # Apply PEFT/LoRA configuration if enabled
    if USE_PEFT:
        # Configure LoRA
        lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query_key_value", "dense"],  # Correct modules for this model architecture
    )
        
        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
        logger.info("🔧 Applied LoRA adapters for efficient fine-tuning")
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        # Ensure model is in training mode and parameters require gradients
        model.train()
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad = True
        
        # Verify trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        # Ensure we have trainable parameters
        if trainable_params == 0:
            logger.error("❌ No trainable parameters found! LoRA adapters may not be applied correctly.")
            raise ValueError("No trainable parameters found in the model")
    
    # Test model on GPU with a simple forward pass
    if torch.cuda.is_available():
        try:
            test_input = torch.tensor([[1, 2, 3]], device='cuda')
            with torch.no_grad():
                _ = model(test_input)
            logger.info("✅ Model GPU test successful")
        except Exception as e:
            logger.warning(f"Model GPU test failed: {e}")
            logger.info("Model may not be properly on GPU")
    
    # Add padding token if it doesn't exist (important for training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Added EOS token as padding token")
    
    # Get safe max length
    max_length = get_safe_max_length(tokenizer, default_max_length=MAX_LENGTH)
    logger.info(f"Using max_length: {max_length}")
    
    # Generate cache key for this configuration
    cache_key = get_cache_key(MODEL_NAME, max_length, settings.CORPUS_FILES)
    logger.info(f"Cache key: {cache_key}")
    
    # Try to load from cache first
    tokenized_train, tokenized_eval = load_tokenized_cache(cache_key)
    
    if tokenized_train is None:
        logger.info("No cached tokenized data found. Loading and tokenizing data...")
        
        # Load local training data
        train_dataset, eval_dataset = load_local_training_data()
        
        if len(train_dataset) == 0:
            logger.error("No training data found! Please check your data files.")
            return
        
        # Tokenization function with safe max_length
        def tokenize_fn(examples):
            return tokenizer(
                examples["text"], 
                truncation=True, 
                max_length=max_length,
                padding=True,
                return_tensors=None  # Don't return tensors here, let the data collator handle it
            )
        
        # Tokenize datasets
        logger.info("Tokenizing training dataset...")
        tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        
        tokenized_eval = None
        if eval_dataset:
            logger.info("Tokenizing validation dataset...")
            tokenized_eval = eval_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        
        # Save to cache for future use
        save_tokenized_cache(tokenized_train, tokenized_eval, cache_key)
        
        # Clear old caches to save disk space
        clear_old_caches()
    else:
        logger.info("🚀 Using cached tokenized data - skipping tokenization step!")
    
    # Data collator with optimizations
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # This is for causal LM, not masked LM
        pad_to_multiple_of=8,  # Pad to multiple of 8 for better GPU utilization
        return_tensors="pt",  # Return PyTorch tensors directly
    )
    
    # Training arguments - optimized for memory usage
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=False,  # Disable FP16 to avoid gradient scaling issues
        dataloader_drop_last=True,
        dataloader_pin_memory=False,  # Disable pin memory to save GPU memory
        report_to=None,  # Disable wandb/tensorboard logging
        gradient_checkpointing=GRADIENT_CHECKPOINTING and not USE_PEFT,  # Disable with PEFT
        # Memory optimization settings
        dataloader_num_workers=0 if IS_WINDOWS else 2,  # Use 0 workers on Windows to avoid multiprocessing issues
        save_total_limit=2,  # Keep only 2 checkpoints to save disk space
        # Gradient clipping for stability
        max_grad_norm=1.0,  # Enable gradient clipping with reasonable value
        # PEFT-specific settings
        save_strategy="steps" if USE_PEFT else "epoch",
        eval_strategy="steps" if USE_PEFT else "no",
        load_best_model_at_end=True if USE_PEFT else False,
        # Performance optimizations
        remove_unused_columns=True,  # Remove unused columns to save memory
        # Additional settings to prevent hanging
        # dataloader_persistent_workers=False,  # May not be supported in all versions
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )
    
    # Add progress callback to show training progress
    from transformers import TrainerCallback
    class ProgressCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 10 == 0:  # Log every 10 steps
                logger.info(f"Training step {state.global_step}/{state.max_steps} - Loss: {state.log_history[-1].get('train_loss', 'N/A') if state.log_history else 'N/A'}")
    
    trainer.add_callback(ProgressCallback())
    
    # Clear memory before training
    clear_gpu_memory()
    
    # Log GPU memory status
    if torch.cuda.is_available():
        logger.info(f"GPU memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Train
    logger.info("Starting training...")
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA out of memory error: {e}")
        logger.info("Try reducing BATCH_SIZE further or MAX_LENGTH")
        clear_gpu_memory()
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        clear_gpu_memory()
        raise
    
    # Save model and tokenizer
    logger.info(f"Saving model to {OUTPUT_DIR}")
    if USE_PEFT:
        # For PEFT models, save the adapters
        model.save_pretrained(OUTPUT_DIR)
        logger.info("✅ Saved LoRA adapters")
    else:
        # For full fine-tuning, save the entire model
        trainer.save_model()
    
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    logger.info("Training completed successfully!")
    logger.info(f"Fine-tuned model saved to: {OUTPUT_DIR}")
    logger.info("You can now use this model with app.py by setting MODEL_PATH in settings.py")
    
    # Optional: Update settings.py automatically
    update_settings_for_finetuned_model()

def update_settings_for_finetuned_model():
    """Update settings.py to use the fine-tuned model"""
    settings_file = "core/settings.py"
    if os.path.exists(settings_file):
        try:
            with open(settings_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace the MODEL_PATH line
            old_line = 'MODEL_PATH = os.getenv("MODEL_PATH", "")  # local checkpoint folder (if used). Empty -> use MODEL_NAME'
            new_line = 'MODEL_PATH = os.getenv("MODEL_PATH", "model")  # local checkpoint folder (if used). Empty -> use MODEL_NAME'
            
            if old_line in content:
                content = content.replace(old_line, new_line)
                
                with open(settings_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info("✅ Updated settings.py to use fine-tuned model")
                logger.info("🔄 Restart app.py to use the fine-tuned model")
            else:
                logger.info("ℹ️  Please manually set MODEL_PATH = 'mbart_finetuned' in core/settings.py")
                
        except Exception as e:
            logger.warning(f"Could not auto-update settings.py: {e}")
            logger.info("ℹ️  Please manually set MODEL_PATH = 'mbart_finetuned' in core/settings.py")
    else:
        logger.info("ℹ️  Please manually set MODEL_PATH = 'mbart_finetuned' in core/settings.py")

if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--clear-cache":
            print("🗑️  Clearing all cached tokenized data...")
            clear_all_caches()
            print("✅ Cache cleared! Starting training with fresh tokenization...")
        elif sys.argv[1] == "--no-quantization":
            print("🔧 Disabling quantization for this run...")
            # These will be set in main() function
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python train.py                    # Training with quantization + LoRA (recommended)")
            print("  python train.py --clear-cache      # Clear cache and retrain")
            print("  python train.py --no-quantization  # Full fine-tuning without quantization")
            print("  python train.py --help             # Show this help")
            print("\nNote: Quantization + LoRA is much faster and uses less memory!")
            sys.exit(0)
    
    main()