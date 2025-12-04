#!/usr/bin/env python3
"""
Unsloth Fine-Tuning Script for WSL
Trains Llama-3.2-3B-Instruct on Hindi instruction dataset (all 200k samples)
"""

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import os
import json
from collections import Counter, defaultdict
import random

print("=" * 70)
print("Unsloth Fine-Tuning - WSL")
print("=" * 70)

# ============================================================================
# Step 1: Load Dataset
# ============================================================================
print("\n[Step 1] Loading Hindi dataset...")

# Dataset path (relative to project root)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
hindi_path = os.path.join(project_root, "data", "Hindi.jsonl")

# Check if dataset exists
if not os.path.exists(hindi_path):
    raise FileNotFoundError(f"Hindi dataset not found: {hindi_path}")

print(f"  Hindi dataset: {hindi_path}")

# Load dataset
dataset = load_dataset("json", data_files=hindi_path, split="train")

print(f"  ✓ Loaded {len(dataset):,} Hindi samples")
print(f"  ✓ Using all {len(dataset):,} samples for training (no sampling)")

# Show first example
if len(dataset) > 0:
    print("\n  First example:")
    print(json.dumps(dataset[0], ensure_ascii=False, indent=2)[:500])

# ============================================================================
# Step 3: Load Model
# ============================================================================
print("\n[Step 3] Loading model...")

max_seq_length = 2048
dtype = None  # Auto-detect
load_in_4bit = True

print("  Loading Llama-3.2-3B-Instruct model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
print("  ✓ Model loaded successfully")

# ============================================================================
# Step 4: Format Dataset
# ============================================================================
print("\n[Step 4] Formatting dataset...")

# Multilingual prompt template
multilingual_prompt = """You are a helpful multilingual assistant capable of understanding and responding in multiple Indian languages including Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Marathi, Punjabi, Odia, Assamese, Urdu, and more.

Please respond to the following in the same language as the input:

{}
"""
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = []
    if 'instruction' in examples and 'output' in examples:
        for instruction, output in zip(examples['instruction'], examples['output']):
            text = multilingual_prompt.format(f"{instruction}\n\n{output}") + EOS_TOKEN
            texts.append(text)
    elif 'text' in examples:
        for text in examples['text']:
            texts.append(text + EOS_TOKEN)
    else:
        raise ValueError(f"Unknown dataset format. Keys: {list(examples.keys())}")
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)
print("  ✓ Dataset formatted successfully")

# ============================================================================
# Step 5: Configure LoRA
# ============================================================================
print("\n[Step 5] Configuring LoRA...")

model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
print("  ✓ LoRA configuration applied")

# ============================================================================
# Step 6: Setup Trainer
# ============================================================================
print("\n[Step 6] Setting up trainer...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=25000,  # 1 full epoch for ~200k samples
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)
print("  ✓ Trainer configured and ready")

# ============================================================================
# Step 7: Train Model
# ============================================================================
print("\n" + "=" * 70)
print("[Step 7] Starting training...")
print("=" * 70)
print("This may take 6-8 hours depending on your GPU (RTX 3060).")
print("=" * 70)

trainer.train()

print("=" * 70)
print("  ✓ Training complete!")
print("=" * 70)

# ============================================================================
# Step 8: Save Model
# ============================================================================
print("\n[Step 8] Saving model...")

output_dir = os.path.join(project_root, "lora_model")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"  ✓ Model saved to: {output_dir}")

print("\n" + "=" * 70)
print("✅ All done! Model saved successfully!")
print("=" * 70)

