# 📚 Two-Notebook Training Workflow

## 🎯 **Why Split Into Two Notebooks?**

### **Problem with One Notebook:**
- ❌ Tokenizing 10M samples takes 1-2 hours
- ❌ If training fails, you re-tokenize everything
- ❌ Can't experiment with different training configs
- ❌ Wastes Colab GPU time on CPU work

### **Solution: Split the Work!**
- ✅ Tokenize ONCE, reuse forever
- ✅ Experiment with training configs freely
- ✅ Save time and GPU quota
- ✅ Better debugging

---

## 📋 **The Two Notebooks**

### **Notebook 1: `tokenize_data.ipynb`**
**Purpose:** Tokenize raw text → Save processed data

- **Input:** `training_merged.rar` (raw text files)
- **Output:** `tokenized_data.zip` (processed dataset)
- **Time:** 1-2 hours
- **GPU:** Not needed (CPU is fine)
- **Run:** Once per dataset

### **Notebook 2: `train_from_tokenized.ipynb`**
**Purpose:** Load processed data → Train adapter

- **Input:** `tokenized_data.zip` (from Notebook 1)
- **Output:** `gurukul_lite_v2.zip` (trained adapter)
- **Time:** 4-6 hours (depends on data size)
- **GPU:** T4 required
- **Run:** As many times as needed!

---

## 🚀 **Complete Workflow**

### **Phase 1: Tokenization (Do Once)**

1. **Open Colab:** https://colab.research.google.com/
2. **Upload:** `notebooks/tokenize_data.ipynb`
3. **Runtime:** CPU is fine (or GPU if available)
4. **Upload data:** `training_merged.rar`
5. **Set sampling:**
   ```python
   SAMPLE_PERCENTAGE = 0.3  # 30% = good balance
   ```
6. **Run all cells**
7. **Wait:** 1-2 hours
8. **Download:** `tokenized_data.zip` (~200-500 MB)
9. **Save on PC:** Keep this file safe!

---

### **Phase 2: Training (Repeat as Needed)**

1. **Open NEW Colab session**
2. **Upload:** `notebooks/train_from_tokenized.ipynb`
3. **Enable T4 GPU:** Runtime → Change runtime type → T4
4. **Upload:** `tokenized_data.zip` (from Phase 1)
5. **Run all cells**
6. **Keep alive:** F12 → paste keep-alive code
7. **Wait:** 4-6 hours
8. **Download:** `gurukul_lite_v2.zip`

---

## 🎛️ **Experiment Freely!**

Since tokenization is done, you can now experiment with:

### **Different Training Configs:**

**Faster training:**
```python
num_train_epochs=1  # Instead of 3
```

**Better quality:**
```python
learning_rate=1e-4  # Lower learning rate
num_train_epochs=5  # More epochs
```

**Smaller batch (if OOM):**
```python
per_device_train_batch_size=2  # Instead of 4
```

**Just re-upload `tokenized_data.zip` and train again!**

---

## 📊 **Recommended Sampling Strategies**

### **For Fast Testing (8-12 hours):**
In `tokenize_data.ipynb`:
```python
SAMPLE_PERCENTAGE = 0.1  # 10% (~1M samples)
```

### **For Good Quality (20-30 hours):**
```python
SAMPLE_PERCENTAGE = 0.3  # 30% (~3M samples)
```

### **For Best Quality (100+ hours):**
```python
SAMPLE_PERCENTAGE = 1.0  # 100% (all 10M samples)
```

---

## 💡 **Pro Tips**

### **1. Create Multiple Tokenized Versions**

Tokenize different percentages:
- `tokenized_data_10pct.zip` - For quick tests
- `tokenized_data_30pct.zip` - For good quality
- `tokenized_data_100pct.zip` - For production

### **2. Reuse Across Experiments**

Upload same `tokenized_data.zip` to:
- Try different learning rates
- Try different epoch counts
- Try different LoRA configs

### **3. Save Tokenization Time**

1-2 hours tokenization → Reused 10+ times = **10-20 hours saved!**

---

## 🔄 **Complete File Flow**

```
PC: training_merged.rar
    ↓ (upload to Colab 1)
Colab 1: tokenize_data.ipynb
    ↓ (tokenize 1-2 hours)
Colab 1: tokenized_data.zip
    ↓ (download to PC)
PC: tokenized_data.zip
    ↓ (upload to Colab 2)
Colab 2: train_from_tokenized.ipynb
    ↓ (train 4-6 hours)
Colab 2: gurukul_lite_v2.zip
    ↓ (download to PC)
PC: adapters/gurukul_lite_v2/
    ✅ DONE!
```

---

## ⚠️ **Important Notes**

### **Tokenization Notebook:**
- ✅ Can stop/restart anytime (caches progress)
- ✅ CPU-only is fine (slower but works)
- ✅ Can close browser after starting
- ⚠️ Make sure to download `tokenized_data.zip` at the end!

### **Training Notebook:**
- ⚠️ NEEDS GPU (T4 minimum)
- ⚠️ Must keep browser open (use keep-alive code)
- ⚠️ Cannot easily resume if interrupted
- ✅ Much faster than combined notebook

---

## 📝 **Quick Start Commands**

### **Current Session (if already started):**

**Stop your current training:**
- Runtime → Interrupt execution

**Start fresh with new workflow:**

1. **Tokenize:**
   - Upload `tokenize_data.ipynb`
   - Set `SAMPLE_PERCENTAGE = 0.3`
   - Run all cells
   - Download `tokenized_data.zip`

2. **Train:**
   - Upload `train_from_tokenized.ipynb`
   - Upload `tokenized_data.zip`
   - Enable T4 GPU
   - Run all cells

---

## ✅ **Benefits Summary**

| Benefit | Time Saved | Flexibility |
|---------|------------|-------------|
| Tokenize once | 1-2 hours per experiment | ⭐⭐⭐⭐⭐ |
| Experiment freely | Multiple retries | ⭐⭐⭐⭐⭐ |
| Faster debugging | Immediate | ⭐⭐⭐⭐⭐ |
| Better organization | Cleaner workflow | ⭐⭐⭐⭐⭐ |

---

## 🎉 **Ready to Start!**

1. ✅ `notebooks/tokenize_data.ipynb` - Created
2. ✅ `notebooks/train_from_tokenized.ipynb` - Created
3. ✅ `training_merged.rar` - You have this

**Upload the first notebook and let's go!** 🚀

