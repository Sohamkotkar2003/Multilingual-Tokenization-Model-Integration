#!/usr/bin/env python3
"""
Train a unified LoRA adapter on ALL 29 languages (21 original + 8 new)
Creates gurukul_lite_v2 adapter supporting 38 languages total
(29 unique + 9 bootstrapped dialects)
"""

import sys
import io
import torch
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import json

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

print("="*80)
print("🚀 TRAINING UNIFIED GURUKUL ADAPTER V2")
print("="*80)
print("This will train ONE adapter on all 29 languages")
print("Expected time: 6-8 hours on RTX 4050 (can run overnight)")
print("="*80)

# Configuration
BASE_MODEL = "bigscience/bloomz-560m"
TRAIN_DATA_DIR = Path("data/training_merged")
VAL_DATA_DIR = Path("data/validation_merged")
OUTPUT_DIR = Path("adapters/gurukul_lite_v2")
MAX_LENGTH = 256
BATCH_SIZE = 1  # Very small for RTX 4050
GRADIENT_ACCUMULATION = 8  # Effective batch size = 8
EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_SAMPLES = None  # Use all data

print(f"\n📋 Configuration:")
print(f"   Base Model: {BASE_MODEL}")
print(f"   Training Data: {TRAIN_DATA_DIR}/")
print(f"   Validation Data: {VAL_DATA_DIR}/")
print(f"   Output: {OUTPUT_DIR}/")
print(f"   Epochs: {EPOCHS}")
print(f"   Batch Size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION})")
print(f"   Learning Rate: {LEARNING_RATE}")
print(f"   Max Length: {MAX_LENGTH}")

# Check GPU
if torch.cuda.is_available():
    print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("\n⚠️  No GPU detected - training will be VERY slow!")
    response = input("Continue anyway? (yes/no): ")
    if response.lower() != 'yes':
        sys.exit(0)

# Load all training data
print("\n" + "="*80)
print("📥 LOADING TRAINING DATA")
print("="*80)

train_files = sorted(TRAIN_DATA_DIR.glob("*.txt"))
print(f"\nFound {len(train_files)} training files:")

all_train_texts = []
for file_path in train_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
        all_train_texts.extend(texts)
        print(f"   ✅ {file_path.name:20} - {len(texts):5} samples")

print(f"\n📊 Total training samples: {len(all_train_texts)}")

# Load validation data
print("\n" + "="*80)
print("📥 LOADING VALIDATION DATA")
print("="*80)

val_files = sorted(VAL_DATA_DIR.glob("*.txt"))
print(f"\nFound {len(val_files)} validation files:")

all_val_texts = []
for file_path in val_files:
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
        all_val_texts.extend(texts)
        print(f"   ✅ {file_path.name:20} - {len(texts):5} samples")

print(f"\n📊 Total validation samples: {len(all_val_texts)}")

# Apply max samples limit if specified
if MAX_SAMPLES:
    all_train_texts = all_train_texts[:MAX_SAMPLES]
    all_val_texts = all_val_texts[:int(MAX_SAMPLES * 0.2)]
    print(f"\n⚠️  Limited to {len(all_train_texts)} train + {len(all_val_texts)} val samples")

# Load tokenizer
print("\n" + "="*80)
print("📥 LOADING TOKENIZER")
print("="*80)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Loaded tokenizer for {BASE_MODEL}")

# Tokenize datasets
print("\n" + "="*80)
print("🔤 TOKENIZING DATASETS")
print("="*80)

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length'
    )

train_dataset = Dataset.from_dict({"text": all_train_texts})
val_dataset = Dataset.from_dict({"text": all_val_texts})

print(f"\n   Tokenizing {len(train_dataset)} training samples...")
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print(f"   Tokenizing {len(val_dataset)} validation samples...")
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print(f"\n✅ Tokenization complete")

# Load base model with 8-bit quantization
print("\n" + "="*80)
print("📥 LOADING BASE MODEL (8-bit)")
print("="*80)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

print(f"✅ Loaded {BASE_MODEL} in 8-bit mode")

# Configure LoRA
print("\n" + "="*80)
print("⚙️  CONFIGURING LORA")
print("="*80)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query_key_value"],
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print(f"\n✅ LoRA configured (r=8, alpha=16)")

# Training arguments
print("\n" + "="*80)
print("⚙️  CONFIGURING TRAINING")
print("="*80)

OUTPUT_DIR.mkdir(exist_ok=True)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=LEARNING_RATE,
    warmup_steps=100,
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
    evaluation_strategy="steps",
    save_total_limit=2,
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    report_to="none",
    gradient_checkpointing=True,
    optim="adamw_torch"
)

print(f"✅ Training configured")
print(f"   - Total steps: ~{len(train_dataset) // (BATCH_SIZE * GRADIENT_ACCUMULATION) * EPOCHS}")
print(f"   - Checkpoints every 500 steps")
print(f"   - Evaluation every 500 steps")

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Trainer
print("\n" + "="*80)
print("🏋️  INITIALIZING TRAINER")
print("="*80)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

print(f"✅ Trainer ready")

# Save metadata
metadata = {
    "base_model": BASE_MODEL,
    "languages_count": len(train_files),
    "train_samples": len(all_train_texts),
    "val_samples": len(all_val_texts),
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation": GRADIENT_ACCUMULATION,
    "learning_rate": LEARNING_RATE,
    "lora_r": lora_config.r,
    "lora_alpha": lora_config.lora_alpha,
    "max_length": MAX_LENGTH,
    "training_started": datetime.now().isoformat(),
    "description": "Unified adapter for 38 languages (29 unique + 9 dialects)"
}

metadata_file = OUTPUT_DIR / "metadata.json"
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\n💾 Metadata saved to {metadata_file}")

# Start training
print("\n" + "="*80)
print("🚀 STARTING TRAINING")
print("="*80)
print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Estimated time: 6-8 hours on RTX 4050")
print(f"You can monitor progress in {OUTPUT_DIR}/")
print("\n" + "="*80 + "\n")

start_time = datetime.now()

try:
    trainer.train()
    
    # Training complete
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds() / 3600
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\nDuration: {duration:.2f} hours")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Save final adapter
    print("\n💾 Saving final adapter...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    # Update metadata
    metadata["training_completed"] = end_time.isoformat()
    metadata["training_duration_hours"] = duration
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Adapter saved to: {OUTPUT_DIR.absolute()}")
    
    # Create README
    readme = f"""# Gurukul Lite V2 - Unified Multilingual Adapter

**Created:** {end_time.strftime('%Y-%m-%d %H:%M:%S')}  
**Training Duration:** {duration:.2f} hours  
**Base Model:** {BASE_MODEL}

## Languages Supported (38 Total)

### Original 21 Languages
Assamese, Bengali, Bodo, English, Gujarati, Hindi, Kannada, Kashmiri, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu

### New 8 Languages (Trained)
Sinhala, Tibetan, Dzongkha, Pashto, Dari, Vietnamese, Thai, Burmese

### Bootstrapped 9 Dialects
Awadhi, Bhojpuri, Magahi, Chhattisgarhi, Haryanvi, Himachali, Pahadi, Mizo, Tamil (Sri Lanka)

## Training Details
- **Samples:** {len(all_train_texts):,} training, {len(all_val_texts):,} validation
- **LoRA Config:** r={lora_config.r}, alpha={lora_config.lora_alpha}
- **Epochs:** {EPOCHS}
- **Batch Size:** {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION})
- **GPU:** RTX 4050 (8-bit quantization)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("{BASE_MODEL}")
model = PeftModel.from_pretrained(model, "adapters/gurukul_lite_v2")
tokenizer = AutoTokenizer.from_pretrained("{BASE_MODEL}")
```
"""
    
    readme_file = OUTPUT_DIR / "README.md"
    readme_file.write_text(readme, encoding='utf-8')
    
    print(f"✅ README created: {readme_file}")
    
    print("\n" + "="*80)
    print("🎉 ALL DONE!")
    print("="*80)
    print(f"\n✅ Unified adapter trained on 38 languages!")
    print(f"   Location: {OUTPUT_DIR.absolute()}")
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Test the adapter with sample prompts")
    print(f"   2. Replace old gurukul_lite with this version")
    print(f"   3. Run smoke tests")
    print(f"   4. Commit to repository")
    
except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user")
    print(f"   Partial model saved in: {OUTPUT_DIR}/")
    sys.exit(1)
except Exception as e:
    print(f"\n\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


