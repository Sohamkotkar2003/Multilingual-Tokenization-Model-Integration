#!/usr/bin/env python3
"""Test adapter directly without API to debug"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
import io

# Fix Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*80)
print("DIRECT ADAPTER TEST (No API)")
print("="*80)

print("\nLoading base model...")
base_model_name = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True
)

print("Loading adapter...")
model = PeftModel.from_pretrained(model, "adapters/gurukul_lite")
model.eval()

print("✅ Model loaded!\n")

# Test prompts - exactly like Colab
test_prompts = [
    "Translate to Hindi: Hello friend, how are you?",
    "Translate to Bengali: Good morning, have a nice day.",
    "Translate to Tamil: Thank you very much."
]

print("="*80)
print("TESTING")
print("="*80)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{i}. Prompt: {prompt}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate - using simple settings
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,  # Greedy - most reliable
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"   Output: {generated_text}")
    print(f"   Length: {len(generated_text)} chars")

print("\n" + "="*80)
print("Test complete!")

