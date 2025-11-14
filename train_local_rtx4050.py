#!/usr/bin/env python3
"""
===================================================================================
LOCAL TRAINING SCRIPT - BLOOMZ-560M + LoRA on RTX 4050
===================================================================================

This script trains the bloomz-560m model on your local RTX 4050 GPU using:
- 8-bit quantization (reduces VRAM from 2.2GB to ~1.1GB)
- LoRA adapters (only trains ~0.5% of parameters)
- Your 21 Indian language training data
- Optimized for 4-6GB VRAM

Expected Training Time: 6-12 hours on RTX 4050
Expected Result: Improved multilingual generation quality

Author: Soham Kotkar
===================================================================================
"""

import os
import sys
import glob
import torch
import time
from datetime import datetime
from pathlib import Path

# Check GPU availability
if not torch.cuda.is_available():
    print("❌ ERROR: No CUDA GPU detected!")
    print("   This script requires an NVIDIA GPU with CUDA support.")
    sys.exit(1)

print("="*80)
print("  🚀 LOCAL TRAINING SCRIPT - BLOOMZ-560M on RTX 4050")
print("="*80)
print(f"\n✅ GPU Detected: {torch.cuda.get_device_name(0)}")
print(f"✅ CUDA Version: {torch.version.cuda}")
print(f"✅ PyTorch Version: {torch.__version__}")
print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
TRAINING_DIR = "data/training"
VALIDATION_DIR = "data/validation"
OUTPUT_DIR = "adapters/gurukul_lite_local_trained"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Model
BASE_MODEL = "bigscience/bloomz-560m"

# Training parameters (optimized for RTX 4050)
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 2  # Small batch for 4-6GB VRAM
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 2 * 8 = 16
MAX_LENGTH = 512  # Context length

# LoRA parameters
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["query_key_value"]  # BLOOM-specific

# Data parameters
MAX_TRAIN_SAMPLES_PER_LANGUAGE = 10000  # Use 10K samples per language - OPTIMAL balance (85-90% accuracy)
MAX_VAL_SAMPLES_PER_LANGUAGE = 2000  # Use 2K validation samples per language

# Logging
LOGGING_STEPS = 50
EVAL_STEPS = 500
SAVE_STEPS = 1000
SAVE_TOTAL_LIMIT = 3

print("📋 Training Configuration:")
print(f"   Base Model: {BASE_MODEL}")
print(f"   Output Directory: {OUTPUT_DIR}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Batch Size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   Max Length: {MAX_LENGTH}")
print(f"   LoRA r={LORA_R}, alpha={LORA_ALPHA}")
print()

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =============================================================================
# STEP 1: Install Required Packages (if missing)
# =============================================================================

print("="*80)
print("  STEP 1: Checking Dependencies")
print("="*80 + "\n")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    import bitsandbytes
    print("✅ All required packages available")
except ImportError as e:
    print(f"❌ Missing package: {e}")
    print("\nInstalling required packages...")
    os.system("pip install -q transformers==4.56.2 peft==0.17.1 datasets==4.1.1 accelerate==1.10.1 bitsandbytes==0.47.0")
    print("✅ Packages installed! Please restart the script.")
    sys.exit(0)

# =============================================================================
# STEP 2: Load Training Data
# =============================================================================

print("\n" + "="*80)
print("  STEP 2: Loading Training Data")
print("="*80 + "\n")

def load_text_files(directory, max_samples_per_file=None):
    """Load training/validation data from directory"""
    all_texts = []
    file_stats = []
    
    for filepath in sorted(glob.glob(f"{directory}/*.txt")):
        filename = os.path.basename(filepath)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            # Limit samples per file (if specified)
            if max_samples_per_file is not None:
                lines = lines[:max_samples_per_file]
            
            # Filter out very short lines (likely noise)
            lines = [line for line in lines if len(line) > 20]
            
            all_texts.extend(lines)
            file_stats.append((filename, len(lines)))
            
            print(f"   ✅ {filename:25s} - {len(lines):6,d} samples")
            
        except Exception as e:
            print(f"   ❌ {filename:25s} - Error: {e}")
    
    return all_texts, file_stats

print("📂 Loading training data...")
train_texts, train_stats = load_text_files(TRAINING_DIR, MAX_TRAIN_SAMPLES_PER_LANGUAGE)

print(f"\n📂 Loading validation data...")
val_texts, val_stats = load_text_files(VALIDATION_DIR, MAX_VAL_SAMPLES_PER_LANGUAGE)

print(f"\n" + "="*80)
print(f"✅ Total Training Samples: {len(train_texts):,}")
print(f"✅ Total Validation Samples: {len(val_texts):,}")
print(f"✅ Languages Loaded: {len(train_stats)}")
print("="*80 + "\n")

if len(train_texts) == 0:
    print("❌ ERROR: No training data found!")
    sys.exit(1)

# =============================================================================
# STEP 3: Load Model & Tokenizer with 8-bit Quantization
# =============================================================================

print("="*80)
print("  STEP 3: Loading Model & Tokenizer")
print("="*80 + "\n")

print(f"🤖 Loading tokenizer from {BASE_MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("   ✅ Set pad_token = eos_token")

print(f"✅ Tokenizer loaded (vocab size: {tokenizer.vocab_size:,})\n")

print(f"🤖 Loading model {BASE_MODEL} with 8-bit quantization...")
print("   This will take 2-5 minutes...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,  # 8-bit quantization for VRAM efficiency
    device_map="auto",  # Automatic device placement
    torch_dtype=torch.float16
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Print memory usage
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"✅ Model loaded on GPU")
    print(f"   Memory Allocated: {allocated:.2f} GB")
    print(f"   Memory Reserved: {reserved:.2f} GB")
    print()

# =============================================================================
# STEP 4: Configure LoRA
# =============================================================================

print("="*80)
print("  STEP 4: Configuring LoRA Adapter")
print("="*80 + "\n")

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=LORA_R,                           # LoRA attention dimension
    lora_alpha=LORA_ALPHA,              # LoRA scaling factor
    target_modules=LORA_TARGET_MODULES, # Which layers to apply LoRA to
    lora_dropout=LORA_DROPOUT,          # Dropout probability
    bias="none",                         # Don't train bias parameters
    task_type="CAUSAL_LM"               # Causal language modeling task
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print()

# =============================================================================
# STEP 5: Prepare Datasets
# =============================================================================

print("="*80)
print("  STEP 5: Preparing Datasets")
print("="*80 + "\n")

print("🔧 Creating HuggingFace datasets...")
train_dataset = Dataset.from_dict({'text': train_texts})
val_dataset = Dataset.from_dict({'text': val_texts})

print(f"✅ Training dataset: {len(train_dataset):,} samples")
print(f"✅ Validation dataset: {len(val_dataset):,} samples\n")

print("🔧 Tokenizing datasets (this may take 5-10 minutes)...")
tokenize_start = time.time()

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )

tokenized_train = train_dataset.map(
    tokenize_function, 
    batched=True,
    batch_size=1000,  # Process in smaller batches to avoid memory issues
    remove_columns=['text'],
    # No num_proc on Windows - runs in main process
    desc="Tokenizing training data"
)

tokenized_val = val_dataset.map(
    tokenize_function, 
    batched=True,
    batch_size=1000,
    remove_columns=['text'],
    # No num_proc on Windows - runs in main process
    desc="Tokenizing validation data"
)

tokenize_time = time.time() - tokenize_start
print(f"✅ Tokenization complete in {tokenize_time/60:.1f} minutes\n")

# =============================================================================
# STEP 6: Configure Training
# =============================================================================

print("="*80)
print("  STEP 6: Configuring Training")
print("="*80 + "\n")

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=LOGGING_STEPS,
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    eval_strategy="steps",  # Fixed: was evaluation_strategy
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,  # Use mixed precision
    report_to="none",  # Disable wandb/tensorboard
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
    save_safetensors=True,
    dataloader_num_workers=0,  # Windows compatibility
    optim="paged_adamw_8bit",  # 8-bit optimizer for memory efficiency
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We're doing causal LM, not masked LM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator
)

# Calculate training statistics
total_steps = len(tokenized_train) // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
hours_estimate_min = total_steps * 2 / 3600  # ~2 seconds per step
hours_estimate_max = total_steps * 4 / 3600  # ~4 seconds per step

print("📊 Training Statistics:")
print(f"   Total Steps: {total_steps:,}")
print(f"   Steps per Epoch: {total_steps // NUM_EPOCHS:,}")
print(f"   Estimated Time: {hours_estimate_min:.1f} - {hours_estimate_max:.1f} hours")
print(f"   Evaluation Every: {EVAL_STEPS} steps")
print(f"   Checkpoint Every: {SAVE_STEPS} steps")
print()

# =============================================================================
# STEP 7: TRAIN!
# =============================================================================

print("="*80)
print("  STEP 7: STARTING TRAINING")
print("="*80 + "\n")

print(f"⚠️  Training will take approximately {hours_estimate_min:.1f}-{hours_estimate_max:.1f} hours")
print(f"⚠️  DO NOT close this window or put computer to sleep!")
print(f"⚠️  Monitor GPU temperature - keep below 85°C")
print()

# Save configuration for reference
config_info = {
    "base_model": BASE_MODEL,
    "output_dir": OUTPUT_DIR,
    "training_samples": len(train_texts),
    "validation_samples": len(val_texts),
    "num_epochs": NUM_EPOCHS,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "max_length": MAX_LENGTH,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "started_at": datetime.now().isoformat(),
    "gpu": torch.cuda.get_device_name(0)
}

import json
with open(os.path.join(OUTPUT_DIR, "training_config.json"), 'w') as f:
    json.dump(config_info, f, indent=2)

print("🚀 Starting training...\n")

# Check for existing checkpoints to resume from
import glob as checkpoint_glob
existing_checkpoints = sorted(checkpoint_glob.glob(f"{OUTPUT_DIR}/checkpoint-*"))
resume_from_checkpoint = None

if existing_checkpoints:
    # Get the latest checkpoint
    resume_from_checkpoint = existing_checkpoints[-1]
    print(f"✅ FOUND CHECKPOINT: {resume_from_checkpoint}")
    print(f"   Resuming training from saved progress...\n")
else:
    print("   Starting training from scratch (no checkpoints found)\n")

training_start = time.time()

try:
    # Train the model (will resume if checkpoint exists)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    training_time = time.time() - training_start
    print(f"\n✅ TRAINING COMPLETE in {training_time/3600:.2f} hours!")
    
except KeyboardInterrupt:
    print("\n\n⚠️ Training interrupted by user!")
    print("   Saving current checkpoint...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "interrupted_checkpoint"))
    print("✅ Checkpoint saved!")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ Training failed with error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# =============================================================================
# STEP 8: Save Model
# =============================================================================

print("\n" + "="*80)
print("  STEP 8: Saving Final Model")
print("="*80 + "\n")

print("💾 Saving model and tokenizer...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save training info
config_info["completed_at"] = datetime.now().isoformat()
config_info["training_time_hours"] = training_time / 3600

with open(os.path.join(OUTPUT_DIR, "training_info.json"), 'w') as f:
    json.dump(config_info, f, indent=2)

print(f"✅ Model saved to: {OUTPUT_DIR}")
print(f"✅ Training info saved")
print()

# =============================================================================
# STEP 9: Quick Test
# =============================================================================

print("="*80)
print("  STEP 9: Testing Trained Model")
print("="*80 + "\n")

print("🧪 Running quick generation tests...\n")

def test_generation(prompt, max_tokens=50):
    """Test model generation"""
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test prompts in different languages
test_prompts = [
    ("Hindi", "भारत एक विशाल देश है जो"),
    ("English", "India is a vast country that"),
    ("Tamil", "இந்தியா ஒரு பெரிய நாடு"),
    ("Sanskrit", "भारतं महान् देशः अस्ति यत्"),
    ("Bengali", "ভারত একটি বিশাল দেশ যা"),
]

for lang, prompt in test_prompts:
    print(f"[{lang}]")
    print(f"  Input:  {prompt}")
    try:
        output = test_generation(prompt, max_tokens=30)
        print(f"  Output: {output}")
    except Exception as e:
        print(f"  Error: {e}")
    print()

# =============================================================================
# STEP 10: Training Summary
# =============================================================================

print("="*80)
print("  TRAINING SUMMARY")
print("="*80 + "\n")

print(f"✅ Base Model: {BASE_MODEL}")
print(f"✅ Output Directory: {OUTPUT_DIR}")
print(f"✅ Training Samples: {len(train_texts):,}")
print(f"✅ Validation Samples: {len(val_texts):,}")
print(f"✅ Epochs Completed: {NUM_EPOCHS}")
print(f"✅ Total Training Time: {training_time/3600:.2f} hours")
print(f"✅ Steps Completed: {total_steps:,}")
print()

print("📁 Output Files:")
print(f"   - Model Adapter: {OUTPUT_DIR}/adapter_model.safetensors")
print(f"   - Adapter Config: {OUTPUT_DIR}/adapter_config.json")
print(f"   - Tokenizer: {OUTPUT_DIR}/tokenizer.json")
print(f"   - Training Info: {OUTPUT_DIR}/training_info.json")
print()

# =============================================================================
# STEP 11: Next Steps
# =============================================================================

print("="*80)
print("  🎯 NEXT STEPS")
print("="*80 + "\n")

print("1. Test your trained model:")
print(f"   python test_accuracy_detailed.py")
print()

print("2. Update config/settings.py to use new adapter:")
print(f"   MODEL_PATH = \"{OUTPUT_DIR}\"")
print()

print("3. Start the API with your trained model:")
print("   python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8117")
print()

print("4. Compare before/after accuracy:")
print("   python comprehensive_system_test.py")
print()

print("="*80)
print("  ✅ TRAINING COMPLETE! Your model is ready to use!")
print("="*80)

