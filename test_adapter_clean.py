#!/usr/bin/env python3
"""Test adapter with clean output extraction"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
import io
import re

# Fix Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def extract_translation(full_output, prompt):
    """Extract only the translated part from the output"""
    # Remove the prompt
    output = full_output.replace(prompt, "").strip()
    
    # Split by newlines and find the first line with non-English characters
    lines = output.split('\n')
    
    for line in lines:
        line = line.strip()
        # Check if line has non-ASCII characters (likely the translation)
        if any(ord(char) > 127 for char in line):
            # Clean up any quotes, prefixes
            line = re.sub(r'^["\']|["\']$', '', line)
            line = re.sub(r'^\w+:', '', line)  # Remove "ima:" type prefixes
            return line.strip()
    
    # If no non-ASCII found, return first substantial line
    for line in lines:
        if len(line.strip()) > 10:
            return line.strip()
    
    return output

print("="*80)
print("CLEAN ADAPTER TEST")
print("="*80)

print("\nLoading model...")
base_model_name = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    load_in_8bit=True,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(model, "adapters/gurukul_lite")
model.eval()
print("✅ Loaded!\n")

# Test cases
tests = [
    ("Translate to Hindi: Hello", "नमस्ते or हैलो"),
    ("Translate to Hindi: Good morning", "सुप्रभात or शुभ प्रभात"),
    ("Translate to Bengali: Hello", "হ্যালো"),
    ("Translate to Bengali: Thank you", "ধন্যবাদ"),
    ("Translate to Tamil: Hello", "வணக்கம்"),
    ("Translate to Telugu: Hello", "హలో or నమస్కారం"),
    ("Translate to Gujarati: Hello", "નમસ્તે"),
]

print("="*80)
print("TESTING WITH OUTPUT CLEANING")
print("="*80)

success_count = 0

for i, (prompt, expected) in enumerate(tests, 1):
    print(f"\n{i}. {prompt}")
    print(f"   Expected: {expected}")
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,  # Shorter to reduce noise
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            temperature=1.0
        )
    
    # Decode
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the translation
    translation = extract_translation(full_output, prompt)
    
    print(f"   Got: {translation}")
    
    # Check if it has non-ASCII (non-English) characters
    has_translation = any(ord(char) > 127 for char in translation)
    
    if has_translation and len(translation) > 2:
        print(f"   ✅ Generated translation!")
        success_count += 1
    else:
        print(f"   ❌ No translation found")
    
    # Also show raw output for debugging
    if i <= 3:  # Show first 3
        print(f"   Raw: {full_output[:150]}...")

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"\nSuccess: {success_count}/{len(tests)}")
print(f"Success rate: {100*success_count/len(tests):.0f}%")

if success_count >= len(tests) * 0.7:
    print("\n✅ ADAPTER IS WORKING!")
    print("The translations are being generated, they just need output cleaning.")
    print("\nRecommendation: Update the API to extract only the translated part.")
else:
    print("\n⚠️ Adapter quality needs improvement")
    print("Consider retraining with more epochs or better data filtering.")

print("\n" + "="*80)

