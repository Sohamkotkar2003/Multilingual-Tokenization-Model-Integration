"""
Multilingual Fine-tuning Training Script

This script fine-tunes a language model on multilingual data (Hindi, Sanskrit, Marathi, English)
without requiring any command-line arguments. All configuration is handled through variables
and settings.py.

Usage:
    python train.py

The fine-tuned model will be saved to 'mbart_finetuned/' directory and can be used
immediately with app.py for text generation.

Requirements:
    - Training data in data/training/ directory
    - Validation data in data/validation/ directory
    - Sufficient GPU memory for training
"""

import logging
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
from core import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Training Configuration Variables (No Command Line Args)
# =============================================================================

# Model Configuration
MODEL_NAME = settings.MODEL_NAME  # Use the same model as app.py
OUTPUT_DIR = "mbart_finetuned"  # Output directory for fine-tuned model

# Training Parameters
EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1
WARMUP_STEPS = 500
LEARNING_RATE = 5e-5
FP16 = True  # Use FP16 if GPU available

# Logging and Saving
LOGGING_STEPS = 100
SAVE_STEPS = 500  # Save every 500 steps
EVAL_STEPS = 500  # Evaluate every 500 steps

# Data Configuration
MAX_LENGTH = settings.TRAINING_MAX_LENGTH  # Use training max length from settings

def get_safe_max_length(tokenizer, default_max_length=1024, task="training"):
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
    print("=" * 80)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model and tokenizer
    logger.info(f"Loading tokenizer and model from {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # Add padding token if it doesn't exist (important for training)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Added EOS token as padding token")
    
    # Get safe max length
    max_length = get_safe_max_length(tokenizer, default_max_length=MAX_LENGTH)
    logger.info(f"Using max_length: {max_length}")
    
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
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # This is for causal LM, not masked LM
    )
    
    # Training arguments - using basic parameters for compatibility
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
        fp16=FP16 and torch.cuda.is_available(),
        dataloader_drop_last=True,
        report_to=None,  # Disable wandb/tensorboard logging
        # Removed problematic parameters for compatibility
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model and tokenizer
    logger.info(f"Saving model to {OUTPUT_DIR}")
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
            new_line = 'MODEL_PATH = os.getenv("MODEL_PATH", "mbart_finetuned")  # local checkpoint folder (if used). Empty -> use MODEL_NAME'
            
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
    main()