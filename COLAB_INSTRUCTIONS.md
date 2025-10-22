# 🚀 Google Colab Training Instructions

## **STEP-BY-STEP GUIDE**

### **STEP 1: Open Google Colab** (30 seconds)

1. Go to https://colab.research.google.com/
2. Click **"New Notebook"**

---

### **STEP 2: Enable GPU** (30 seconds)

1. Click **"Runtime"** → **"Change runtime type"**
2. Set **"Hardware accelerator"** to **"T4 GPU"**
3. Click **"Save"**

---

### **STEP 3: Copy & Paste This Code** (1 minute)

Copy ALL the code below and paste it into the first cell in Colab:

```python
# ============================================================================
# BLOOMZ-560M LoRA Adapter Training on Google Colab
# ============================================================================

print("📦 Installing dependencies...")
!pip install -q transformers datasets peft accelerate bitsandbytes scipy

# Check GPU
import torch
print(f"\n🎮 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU!'}")
print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB" if torch.cuda.is_available() else "")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'model_name': 'bigscience/bloomz-560m',
    'output_dir': 'trained_adapter',
    'max_samples': 500,  # Adjust this number
    'num_epochs': 3,
    'batch_size': 4,
    'learning_rate': 2e-4,
    'max_length': 512,
    'lora_r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.05,
}

print("\n⚙️ Configuration:")
for k, v in CONFIG.items():
    print(f"   {k}: {v}")

# ============================================================================
# TRAINING DATA
# ============================================================================

print("\n📊 Creating sample training data...")

# Sample multilingual texts (replace with your own data!)
train_texts = [
    "Hello, how are you today?",
    "नमस्ते, आप कैसे हैं?",  # Hindi
    "你好，你今天怎么样？",  # Chinese
    "Bonjour, comment allez-vous?",  # French
    "Hola, ¿cómo estás?",  # Spanish
    "こんにちは、お元気ですか？",  # Japanese
    "Привет, как дела?",  # Russian
    "Olá, como você está?",  # Portuguese
] * (CONFIG['max_samples'] // 8)

train_texts = train_texts[:CONFIG['max_samples']]

# Split train/val
split_idx = int(len(train_texts) * 0.9)
train_data = train_texts[:split_idx]
val_data = train_texts[split_idx:]

print(f"✅ Training: {len(train_data)} | Validation: {len(val_data)}")

# ============================================================================
# LOAD MODEL
# ============================================================================

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

print("\n🤖 Loading model...")

tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

model = AutoModelForCausalLM.from_pretrained(
    CONFIG['model_name'],
    quantization_config=quantization_config,
    device_map='auto',
    torch_dtype=torch.float16,
)

print("✅ Model loaded")

# ============================================================================
# APPLY LORA
# ============================================================================

print("\n🔧 Applying LoRA...")

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=CONFIG['lora_r'],
    lora_alpha=CONFIG['lora_alpha'],
    lora_dropout=CONFIG['lora_dropout'],
    target_modules=['query_key_value', 'dense', 'dense_h_to_4h', 'dense_4h_to_h'],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================================
# TOKENIZE DATA
# ============================================================================

print("\n📝 Tokenizing...")

def tokenize_texts(texts):
    tokenized = []
    for text in texts:
        tokens = tokenizer(text, truncation=True, max_length=CONFIG['max_length'], padding=False)
        tokenized.append(tokens)
    return Dataset.from_dict({
        'input_ids': [t['input_ids'] for t in tokenized],
        'attention_mask': [t['attention_mask'] for t in tokenized]
    })

train_dataset = tokenize_texts(train_data)
val_dataset = tokenize_texts(val_data)

print("✅ Tokenization complete")

# ============================================================================
# TRAINING
# ============================================================================

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=CONFIG['output_dir'],
    num_train_epochs=CONFIG['num_epochs'],
    per_device_train_batch_size=CONFIG['batch_size'],
    per_device_eval_batch_size=CONFIG['batch_size'],
    gradient_accumulation_steps=4,
    learning_rate=CONFIG['learning_rate'],
    logging_steps=10,
    save_steps=50,
    eval_steps=50,
    fp16=False,
    save_total_limit=2,
    eval_strategy='steps',
    load_best_model_at_end=True,
    report_to='none',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

print("\n" + "="*80)
print("🎯 TRAINING...")
print("="*80 + "\n")

trainer.train()

print("\n" + "="*80)
print("🎉 TRAINING COMPLETE!")
print("="*80)

# Save adapter
print(f"\n💾 Saving adapter...")
model.save_pretrained(CONFIG['output_dir'])
tokenizer.save_pretrained(CONFIG['output_dir'])

print("\n✅ SUCCESS!")
```

---

### **STEP 4: Run the Code** (5-10 minutes)

1. Click the **Play button** ▶️ on the left of the cell
2. Wait for it to complete (5-10 minutes)
3. You'll see progress updates

---

### **STEP 5: Download the Adapter** (1 minute)

After training completes, run this in a NEW cell:

```python
# Download trained adapter
import shutil
from google.colab import files

print("📦 Packaging adapter...")
shutil.make_archive('trained_adapter', 'zip', 'trained_adapter')

print("⬇️ Downloading...")
files.download('trained_adapter.zip')

print("\n🎉 DONE!")
print("\nUnzip on your computer and place in: adapters/gurukul_lite/")
```

---

### **STEP 6: Use on Your Computer** (2 minutes)

1. **Unzip** `trained_adapter.zip`
2. **Copy contents** to `C:\pc\Project\adapters\gurukul_lite\`
3. **Test** with your API:

```bash
# Start API
python -m uvicorn adapter_service.standalone_api:app --port 8110

# Test with adapter
curl -X POST http://localhost:8110/generate \
  -d '{"prompt":"Translate to Hindi: Hello friend", "adapter_path":"adapters/gurukul_lite"}'
```

---

## 🎯 **TO USE YOUR OWN DATA**

Replace the sample data in the code with your own:

```python
# Instead of sample texts, load your files:
train_texts = []

# Upload files to Colab first, then:
import os
for filename in os.listdir('/content/'):
    if filename.endswith('.txt'):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if len(line.strip()) > 10]
            train_texts.extend(lines)
```

Or click the 📁 folder icon in Colab and drag your `.txt` files!

---

## ✅ **EXPECTED OUTPUT**

You should see:
```
🎮 GPU: Tesla T4
   Memory: 15.0 GB

⚙️ Configuration:
   model_name: bigscience/bloomz-560m
   max_samples: 500
   ...

✅ Model loaded
✅ Tokenization complete

🎯 TRAINING...
[Training progress bars]

🎉 TRAINING COMPLETE!
✅ SUCCESS!
```

---

## 🎉 **THAT'S IT!**

**Total time**: ~15-20 minutes

**Cost**: $0 (Colab is free!)

**Result**: Trained LoRA adapter ready to use!

