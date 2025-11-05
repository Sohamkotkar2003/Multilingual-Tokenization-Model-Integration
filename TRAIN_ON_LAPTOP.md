# 🖥️ Train New Language Adapters on Your Laptop

## ✅ Prerequisites:

You've already completed:
- ✅ Scraped 3,274 samples for 8 languages (`data/training_new/`)
- ✅ Bootstrapped 9 dialect languages (`adapters/awadhi_lite/`, etc.)

---

## 🚀 Training on Your Laptop (RTX 4050)

### **Step 1: Train One Language at a Time**

```bash
# Activate venv
venv\Scripts\activate

# Train Sinhala (6-8 hours)
python train_single_language_local.py --language sinhala

# Train Tibetan (6-8 hours)
python train_single_language_local.py --language tibetan

# Train Dzongkha (6-8 hours)
python train_single_language_local.py --language dzongkha

# Train Pashto (6-8 hours)
python train_single_language_local.py --language pashto

# Train Dari (2-3 hours - only 29 samples)
python train_single_language_local.py --language dari

# Train Vietnamese (5-7 hours)
python train_single_language_local.py --language vietnamese

# Train Thai (6-8 hours)
python train_single_language_local.py --language thai

# Train Burmese (5-7 hours)
python train_single_language_local.py --language burmese
```

---

## ⏱️ **Recommended Schedule:**

### **Night 1 (Tonight):**
```bash
python train_single_language_local.py --language sinhala
```
- Start: 11 PM
- Finish: 6-7 AM next morning
- Wake up with trained adapter!

### **Night 2:**
```bash
python train_single_language_local.py --language tibetan
```

### **Night 3:**
```bash
python train_single_language_local.py --language dzongkha
```

### **Night 4:**
```bash
python train_single_language_local.py --language pashto
```

### **Night 5:**
```bash
python train_single_language_local.py --language vietnamese
```

### **Night 6:**
```bash
python train_single_language_local.py --language thai
```

### **Night 7:**
```bash
python train_single_language_local.py --language burmese
```

### **Night 8 (Quick - 2-3 hours):**
```bash
python train_single_language_local.py --language dari
```

---

## 📊 **What Happens During Training:**

1. **Loads data** from `data/training_new/`
2. **Loads BLOOMZ-560m** with 8-bit quantization (fits in 8GB VRAM)
3. **Trains LoRA adapter** for 2 epochs
4. **Saves adapter** to `adapters/{language}_lite/`
5. **Saves metadata** with training stats

---

## 🎯 **Training Configuration:**

- **Batch size:** 1 (for 8GB VRAM)
- **Gradient accumulation:** 16 (effective batch = 16)
- **Learning rate:** 2e-4
- **Epochs:** 2
- **FP16:** Enabled (faster training)
- **Checkpoints:** Every 200 steps

---

## 💾 **Output:**

After training, you'll have:
```
adapters/sinhala_lite/
├── adapter_config.json
├── adapter_model.safetensors  ← The trained adapter
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── metadata.json              ← Training stats
```

---

## 🔍 **Monitor Training:**

You'll see:
```
🚀 TRAINING ADAPTER: SINHALA
Started: 2025-11-05 23:00:00
Device: NVIDIA GeForce RTX 4050
VRAM: 8.0 GB

📂 Loading data for Sinhala...
   ✅ Loaded 500 samples
   Average length: 2821 chars

🤖 Loading BLOOMZ-560m with 8-bit quantization...
   ✅ Model loaded with 8-bit quantization
   
⏱️  Training Configuration:
   Total steps: ~56
   Estimated time: 6.2 - 8.1 hours
   Should finish by: 05:00

🚀 Starting training...

[Training progress bars...]

✅ TRAINING COMPLETE!
Adapter saved to: adapters/sinhala_lite/
```

---

## ⚡ **Quick Commands:**

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Train with more epochs (better quality, longer time)
python train_single_language_local.py --language sinhala --epochs 3

# Train with fewer samples (faster testing)
python train_single_language_local.py --language sinhala --max_samples 100
```

---

## ✅ **After All Training:**

You'll have:
- ✅ 21 original languages (Gurukul Lite)
- ✅ 9 bootstrapped languages (Awadhi, Bhojpuri, etc.)
- ✅ 8 newly trained languages (Sinhala, Tibetan, etc.)

**Total: 38 languages!** 🎉

---

## 🚀 **START NOW:**

Just run:
```bash
python train_single_language_local.py --language sinhala
```

And go to bed! Wake up with your first adapter trained! 🌙

