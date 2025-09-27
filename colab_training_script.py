"""
🚀 Multilingual Fine-tuning Training Script for Google Colab
================================================================

This is a standalone script that can be run in Google Colab to fine-tune a language model
on multilingual data (Hindi, Sanskrit, Marathi, English).

Features:
- All dependencies included
- Sample data generation for demo
- Memory-optimized for Colab's GPU constraints
- Automatic GPU detection and configuration
- Progress tracking and logging
- LoRA/PEFT support for efficient training

Usage in Colab:
1. Upload this script to a Colab notebook
2. Run each cell in sequence
3. The fine-tuned model will be saved and can be downloaded

Requirements:
- Google Colab with GPU enabled (Runtime > Change runtime type > GPU)
- Internet connection for downloading models
"""

# =============================================================================
# CELL 1: Install Dependencies
# =============================================================================

# Install required packages
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q transformers datasets accelerate peft bitsandbytes
!pip install -q sentencepiece langdetect

# Verify installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# =============================================================================
# CELL 2: Import Libraries and Setup Configuration
# =============================================================================

import logging
import os
import gc
import pickle
import hashlib
import sys
import platform
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CELL 3: Configuration Settings
# =============================================================================

# Model Configuration
MODEL_NAME = "AhinsaAI/ahinsa0.5-llama3.2-3B"  # Change this to your preferred model
OUTPUT_DIR = "fine_tuned_model"

# Training Parameters (optimized for Colab)
EPOCHS = 2  # Reduced for Colab demo
BATCH_SIZE = 1  # Conservative for Colab GPU
GRADIENT_ACCUMULATION_STEPS = 4  # Compensate for small batch size
WARMUP_STEPS = 100
LEARNING_RATE = 5e-5
FP16 = False
GRADIENT_CHECKPOINTING = True
USE_QUANTIZATION = True  # Enable 8-bit quantization
USE_PEFT = True  # Enable LoRA for efficient training

# Logging and Saving
LOGGING_STEPS = 50
SAVE_STEPS = 200
EVAL_STEPS = 200

# Data Configuration
MAX_LENGTH = 512  # Conservative for Colab memory

# Supported Languages
SUPPORTED_LANGUAGES = ["hindi", "sanskrit", "marathi", "english"]

# Cache Configuration
CACHE_DIR = "cache"
TOKENIZED_CACHE_DIR = os.path.join(CACHE_DIR, "tokenized")

# Print configuration
print("🔧 Configuration Settings:")
print(f"🤖 Model: {MODEL_NAME}")
print(f"📁 Output Directory: {OUTPUT_DIR}")
print(f"🌐 Languages: {', '.join(SUPPORTED_LANGUAGES)}")
print(f"📊 Training Epochs: {EPOCHS}")
print(f"📦 Batch Size: {BATCH_SIZE}")
print(f"🎯 Learning Rate: {LEARNING_RATE}")
print(f"🔧 Quantization: {'Enabled (8-bit)' if USE_QUANTIZATION else 'Disabled'}")
print(f"🔧 PEFT/LoRA: {'Enabled' if USE_PEFT else 'Disabled'}")

# =============================================================================
# CELL 4: Utility Functions
# =============================================================================

def get_cache_key(tokenizer_name, max_length, data_files):
    """Generate a unique cache key based on tokenizer, max_length, and data files"""
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

def get_safe_max_length(tokenizer, default_max_length=512, task="training"):
    """Get a safe max_length value from tokenizer, avoiding overflow issues"""
    try:
        model_max_length = tokenizer.model_max_length
        
        # Handle very large model_max_length values
        if model_max_length > 100000:
            return default_max_length
            
        # Use the model's actual max length if reasonable
        return min(model_max_length, default_max_length)
        
    except (AttributeError, OverflowError):
        return default_max_length

# =============================================================================
# CELL 5: Sample Data Generation
# =============================================================================

def create_sample_data():
    """Create sample multilingual training data for demonstration"""
    
    # Sample data for each language
    sample_data = {
        "hindi": [
            "मैं एक भारतीय हूं और मुझे हिंदी भाषा सीखना अच्छा लगता है।",
            "यह दुनिया बहुत सुंदर है और हमें इसे संरक्षित करना चाहिए।",
            "शिक्षा सबसे महत्वपूर्ण चीज है जो एक व्यक्ति को सफल बना सकती है।",
            "प्रेम और करुणा हमारे जीवन को सुंदर बनाते हैं।",
            "भारत विविधता में एकता का देश है।"
        ],
        "sanskrit": [
            "सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः।",
            "विद्या ददाति विनयं विनयाद्याति पात्रताम्।",
            "सत्यमेव जयते नानृतं सत्येन पन्था विततो देवयानः।",
            "अहिंसा परमो धर्मः धर्मस्य प्रतिष्ठा।",
            "वसुधैव कुटुम्बकम् इति सुभाषितम्।"
        ],
        "marathi": [
            "मी एक महाराष्ट्रीय आहे आणि मला मराठी भाषा आवडते।",
            "शिक्षण हे माणसाच्या जीवनातील सर्वात महत्वाची गोष्ट आहे।",
            "प्रेम आणि करुणा यांनी जग सुंदर बनतो।",
            "महाराष्ट्र हा संस्कृतीचा आणि परंपरेचा गौरवशाली राज्य आहे।",
            "एकता ही शक्तीचा स्रोत आहे।"
        ],
        "english": [
            "I am learning multiple languages to understand different cultures better.",
            "Education is the key to success and personal development.",
            "Love and compassion make the world a better place to live.",
            "India is a diverse country with unity in diversity.",
            "Technology has revolutionized the way we communicate and learn."
        ]
    }
    
    # Create training data directory
    os.makedirs("data/training", exist_ok=True)
    os.makedirs("data/validation", exist_ok=True)
    
    # Write sample data to files
    for lang, texts in sample_data.items():
        # Training data
        train_file = f"data/training/{lang}_train.txt"
        with open(train_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + "\n")
        
        # Validation data (smaller subset)
        val_file = f"data/validation/{lang}_val.txt"
        with open(val_file, 'w', encoding='utf-8') as f:
            for text in texts[:2]:  # Use first 2 samples for validation
                f.write(text + "\n")
    
    print("✅ Sample data created successfully!")
    print("📁 Training data files:")
    for lang in SUPPORTED_LANGUAGES:
        train_file = f"data/training/{lang}_train.txt"
        val_file = f"data/validation/{lang}_val.txt"
        print(f"  - {train_file}")
        print(f"  - {val_file}")

# Create sample data
create_sample_data()

# =============================================================================
# CELL 6: Data Loading Functions
# =============================================================================

def load_local_training_data():
    """Load training and validation data from local files"""
    logger.info("Loading local training data...")
    
    # Corpus files mapping
    corpus_files = {
        "hindi": "hindi_train.txt",
        "sanskrit": "sanskrit_train.txt", 
        "marathi": "marathi_train.txt",
        "english": "english_train.txt"
    }
    
    # Load training data from all language files
    train_texts = []
    for lang, filename in corpus_files.items():
        filepath = os.path.join("data/training", filename)
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
        "hindi": "hindi_val.txt",
        "sanskrit": "sanskrit_val.txt", 
        "marathi": "marathi_val.txt",
        "english": "english_val.txt"
    }
    
    for lang, filename in val_files.items():
        filepath = os.path.join("data/validation", filename)
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

# =============================================================================
# CELL 7: Model Loading and Configuration
# =============================================================================

def load_model_and_tokenizer():
    """Load model and tokenizer with optimizations"""
    
    # Clear GPU memory before loading model
    clear_gpu_memory()
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {MODEL_NAME}")
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
    logger.info(f"Loading model from {MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16 if quantization_config else torch.float32,
    )
    
    # Check GPU usage
    check_gpu_usage()
    
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
            r=16,  # Rank of adaptation
            lora_alpha=32,  # LoRA scaling parameter
            lora_dropout=0.1,  # LoRA dropout
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Llama-specific modules
        )
        
        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
        logger.info("🔧 Applied LoRA adapters for efficient fine-tuning")
        
        # Print trainable parameters
        model.print_trainable_parameters()
        
        # Ensure model is in training mode
        model.train()
        
        # Verify trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"✅ Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        # Ensure we have trainable parameters
        if trainable_params == 0:
            logger.error("❌ No trainable parameters found! LoRA adapters may not be applied correctly.")
            raise ValueError("No trainable parameters found in the model")
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Added EOS token as padding token")
    
    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# =============================================================================
# CELL 8: Data Preparation and Tokenization
# =============================================================================

def prepare_data():
    """Prepare and tokenize the training data"""
    
    # Load local training data
    train_dataset, eval_dataset = load_local_training_data()
    
    if len(train_dataset) == 0:
        logger.error("No training data found! Please check your data files.")
        return None, None
    
    # Get safe max length
    max_length = get_safe_max_length(tokenizer, default_max_length=MAX_LENGTH)
    logger.info(f"Using max_length: {max_length}")
    
    # Tokenization function
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            max_length=max_length,
            padding=True,
            return_tensors=None  # Don't return tensors here
        )
    
    # Tokenize datasets
    logger.info("Tokenizing training dataset...")
    tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    tokenized_eval = None
    if eval_dataset:
        logger.info("Tokenizing validation dataset...")
        tokenized_eval = eval_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    logger.info("✅ Data tokenization completed!")
    return tokenized_train, tokenized_eval

# Prepare data
tokenized_train, tokenized_eval = prepare_data()

# =============================================================================
# CELL 9: Training Setup
# =============================================================================

def setup_training():
    """Setup training arguments and trainer"""
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # This is for causal LM, not masked LM
        pad_to_multiple_of=8,  # Pad to multiple of 8 for better GPU utilization
        return_tensors="pt",  # Return PyTorch tensors directly
    )
    
    # Training arguments - optimized for Colab
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
        gradient_checkpointing=GRADIENT_CHECKPOINTING and not USE_PEFT,
        dataloader_num_workers=0,  # Use 0 workers in Colab
        save_total_limit=2,  # Keep only 2 checkpoints
        max_grad_norm=1.0,  # Gradient clipping
        save_strategy="steps" if USE_PEFT else "epoch",
        eval_strategy="steps" if USE_PEFT else "no",
        load_best_model_at_end=True if USE_PEFT else False,
        remove_unused_columns=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
    )
    
    # Add progress callback
    from transformers import TrainerCallback
    class ProgressCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % 10 == 0:
                logger.info(f"Training step {state.global_step}/{state.max_steps} - Loss: {state.log_history[-1].get('train_loss', 'N/A') if state.log_history else 'N/A'}")
    
    trainer.add_callback(ProgressCallback())
    
    return trainer

# Setup training
trainer = setup_training()

# =============================================================================
# CELL 10: Training Execution
# =============================================================================

def train_model():
    """Execute the training process"""
    
    # Clear memory before training
    clear_gpu_memory()
    
    # Log GPU memory status
    if torch.cuda.is_available():
        logger.info(f"GPU memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # Train
    logger.info("🚀 Starting training...")
    try:
        trainer.train()
        logger.info("✅ Training completed successfully!")
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA out of memory error: {e}")
        logger.info("Try reducing BATCH_SIZE further or MAX_LENGTH")
        clear_gpu_memory()
        raise
    except Exception as e:
        logger.error(f"Training error: {e}")
        clear_gpu_memory()
        raise
    
    return trainer

# Execute training
trained_trainer = train_model()

# =============================================================================
# CELL 11: Save Model and Generate Download Link
# =============================================================================

def save_and_package_model():
    """Save the trained model and create download package"""
    
    # Save model and tokenizer
    logger.info(f"Saving model to {OUTPUT_DIR}")
    if USE_PEFT:
        # For PEFT models, save the adapters
        model.save_pretrained(OUTPUT_DIR)
        logger.info("✅ Saved LoRA adapters")
    else:
        # For full fine-tuning, save the entire model
        trained_trainer.save_model()
    
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Create a zip file for easy download
    import zipfile
    zip_filename = f"{OUTPUT_DIR}.zip"
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUTPUT_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, OUTPUT_DIR)
                zipf.write(file_path, arcname)
    
    logger.info(f"✅ Model saved and packaged as {zip_filename}")
    logger.info(f"📁 Fine-tuned model saved to: {OUTPUT_DIR}")
    logger.info("🎉 Training completed! You can now download the model.")
    
    # Display download link in Colab
    from google.colab import files
    print(f"\n📥 Download your trained model:")
    files.download(zip_filename)
    
    return OUTPUT_DIR, zip_filename

# Save model
output_dir, zip_file = save_and_package_model()

# =============================================================================
# CELL 12: Test the Fine-tuned Model (Optional)
# =============================================================================

def test_fine_tuned_model():
    """Test the fine-tuned model with sample prompts"""
    
    # Load the fine-tuned model for testing
    if USE_PEFT:
        # For PEFT models, we need to load the base model and then the adapters
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model = get_peft_model(base_model, LoraConfig.from_pretrained(output_dir))
    else:
        model = AutoModelForCausalLM.from_pretrained(output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    
    # Test prompts in different languages
    test_prompts = [
        "मैं एक भारतीय हूं",  # Hindi
        "सर्वे भवन्तु सुखिनः",  # Sanskrit
        "मी एक महाराष्ट्रीय आहे",  # Marathi
        "I am learning multiple languages"  # English
    ]
    
    print("🧪 Testing fine-tuned model with sample prompts:")
    print("=" * 60)
    
    for prompt in test_prompts:
        print(f"\n📝 Prompt: {prompt}")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 Response: {response}")
        print("-" * 40)

# Test the model
test_fine_tuned_model()

# =============================================================================
# CELL 13: Summary and Next Steps
# =============================================================================

print("\n" + "=" * 80)
print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"✅ Model saved to: {output_dir}")
print(f"📦 Download package: {zip_file}")
print(f"🤖 Base model: {MODEL_NAME}")
print(f"🔧 Training method: {'LoRA/PEFT' if USE_PEFT else 'Full fine-tuning'}")
print(f"🌐 Languages trained on: {', '.join(SUPPORTED_LANGUAGES)}")
print("\n📋 Next Steps:")
print("1. Download the model zip file")
print("2. Extract it to your local machine")
print("3. Use the model with your inference script")
print("4. The model can generate text in all trained languages")
print("\n💡 Tips:")
print("- The model uses LoRA adapters for efficient fine-tuning")
print("- You can adjust training parameters for better results")
print("- Add more diverse training data for improved performance")
print("- Experiment with different prompts for better generation")
print("=" * 80)

# =============================================================================
# END OF NOTEBOOK
# =============================================================================
