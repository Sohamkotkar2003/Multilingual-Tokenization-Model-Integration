"""
===================================================================================
COMPLETE COLAB TRAINING SCRIPT - Gurukul Lite Enhanced (Phase C)
===================================================================================

Copy this entire file into a new Google Colab notebook and run it!

Expected Time: 2-6 hours (depending on GPU)
Expected Result: Accuracy 66.7% → 85%+

Instructions:
1. Open Google Colab: https://colab.research.google.com
2. Runtime → Change runtime type → GPU (T4 for free, V100/A100 for Pro)
3. Copy-paste this entire script
4. Upload your data files to Colab
5. Run all cells
6. Download the trained model
===================================================================================
"""

# ============================================================================
# STEP 1: Install Packages
# ============================================================================
print("📦 Installing required packages...")

!pip install -q transformers==4.35.0 peft==0.6.0 datasets==2.14.0 accelerate==0.24.0 bitsandbytes==0.41.0

print("✅ Packages installed!\n")

# Check GPU
!nvidia-smi

# ============================================================================
# STEP 2: Mount Google Drive (OPTION A) OR Upload Files (OPTION B)
# ============================================================================

print("\n" + "="*80)
print("  CHOOSE DATA SOURCE")
print("="*80)
print("\nOPTION A: Mount Google Drive (if you uploaded data there)")
print("OPTION B: Upload files manually (use Files panel on left)\n")

# OPTION A: Uncomment these lines if using Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# TRAINING_DIR = '/content/drive/MyDrive/YourProject/data/training'
# VALIDATION_DIR = '/content/drive/MyDrive/YourProject/data/validation'

# OPTION B: Manual upload (recommended for first time)
!mkdir -p data/training
!mkdir -p data/validation

TRAINING_DIR = 'data/training'
VALIDATION_DIR = 'data/validation'

print(f"\n📁 Training data directory: {TRAINING_DIR}")
print(f"📁 Validation data directory: {VALIDATION_DIR}")
print("\n⚠️ UPLOAD ALL .txt FILES NOW using the Files panel on the left!")
print("   Upload to data/training/ and data/validation/\n")

input("Press ENTER after uploading files...")

# ============================================================================
# STEP 3: Load Base Model
# ============================================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import glob

MODEL_NAME = "bigscience/bloomz-560m"
OUTPUT_DIR = "gurukul_lite_enhanced"

print(f"\n🤖 Loading {MODEL_NAME}...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Tokenizer loaded")

# Load model with 8-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)

model = prepare_model_for_kbit_training(model)
print(f"✅ Model loaded on GPU ({torch.cuda.get_device_name(0)})")
print(f"   Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ============================================================================
# STEP 4: Configure LoRA
# ============================================================================

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

print("✅ LoRA configured!")

# ============================================================================
# STEP 5: Load & Prepare Data
# ============================================================================

print("\n📊 Loading training data...")

def load_text_files(directory, max_lines=1000):
    """Load all txt files from directory"""
    texts = []
    for filepath in glob.glob(f"{directory}/*.txt"):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()[:max_lines] if line.strip()]
            texts.extend(lines)
        print(f"   Loaded {filepath.split('/')[-1]}: {len(lines)} lines")
    return texts

train_texts = load_text_files(TRAINING_DIR, max_lines=800)
val_texts = load_text_files(VALIDATION_DIR, max_lines=200)

print(f"\n✅ Total training samples: {len(train_texts)}")
print(f"✅ Total validation samples: {len(val_texts)}")

# ============================================================================
# STEP 6: Tokenize Data
# ============================================================================

print("\n🔧 Tokenizing datasets...")

train_dataset = Dataset.from_dict({'text': train_texts})
val_dataset = Dataset.from_dict({'text': val_texts})

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512
    )

tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

print("✅ Tokenization complete!")

# ============================================================================
# STEP 7: Configure Training
# ============================================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    logging_steps=50,
    eval_steps=200,
    save_steps=500,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    fp16=True,
    report_to="none",
    push_to_hub=False
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator
)

total_steps = len(tokenized_train) // (4 * 4) * 3
print(f"\n✅ Training configured!")
print(f"   Total steps: ~{total_steps}")
print(f"   Estimated time: {total_steps * 2 / 3600:.1f} - {total_steps * 4 / 3600:.1f} hours")

# ============================================================================
# STEP 8: TRAIN!
# ============================================================================

print("\n" + "="*80)
print("  🚀 STARTING TRAINING!")
print("="*80)
print("\n⚠️ This will take 2-6 hours. Don't close your browser!\n")

trainer.train()

print("\n✅ TRAINING COMPLETE!")

# ============================================================================
# STEP 9: Save Model
# ============================================================================

print("\n💾 Saving model...")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Model saved to {OUTPUT_DIR}")

# Zip for download
!zip -r gurukul_lite_enhanced.zip {OUTPUT_DIR}

print("\n✅ Model zipped! Download gurukul_lite_enhanced.zip")

# ============================================================================
# STEP 10: Quick Test
# ============================================================================

print("\n🧪 Testing new model...\n")

def test_gen(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.6)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

tests = [
    "केवल हिंदी में लिखें: भारत",
    "Write in English only: India",
    "தமிழில் மட்டும் எழுதுங்கள்: இந்தியா"
]

for prompt in tests:
    print(f"Input: {prompt}")
    print(f"Output: {test_gen(prompt)[:100]}...\n")

print("="*80)
print("  ✅ DONE! Download gurukul_lite_enhanced.zip and deploy to your PC!")
print("="*80)

