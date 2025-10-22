# 🚀 FLORES Parallel Data Training Guide

## 📋 What You Have

- **Notebook**: `colab_train_flores.ipynb`
- **Data**: `flores_training_data.txt` (26,117 translation pairs)
- **Expected Training Time**: 40-60 minutes on Colab T4 GPU

---

## ✅ Step-by-Step Instructions

### **Step 1: Upload Data to Google Drive** (5 min)

1. Go to https://drive.google.com/
2. Upload `flores_training_data.txt` to your Drive
   - You can put it in root (My Drive) or in a folder
3. Note the exact path (e.g., `MyDrive/flores_training_data.txt`)

---

### **Step 2: Open Notebook in Colab** (2 min)

1. Go to https://colab.research.google.com/
2. Click **File → Upload notebook**
3. Select `colab_train_flores.ipynb`

---

### **Step 3: Enable GPU** (1 min)

1. Click **Runtime → Change runtime type**
2. Select **Hardware accelerator: GPU**
3. Select **GPU type: T4** (free tier)
4. Click **Save**

---

### **Step 4: Update File Path** (1 min)

In **Cell 2**, update this line to match where you uploaded the file:

```python
# UPDATE THIS PATH
data_file = "/content/drive/MyDrive/flores_training_data.txt"
```

If you put it in a folder like `training_data`, change to:
```python
data_file = "/content/drive/MyDrive/training_data/flores_training_data.txt"
```

---

### **Step 5: Run Training** (40-60 min)

1. Click **Runtime → Run all**
2. When prompted "Permit this notebook to access your Google Drive files?":
   - Click **Connect to Google Drive**
   - Sign in and click **Allow**
3. Wait for training to complete

**What you'll see:**
```
Loading BLOOMZ-560M...
Model loaded!
LoRA applied!
Trainable params: 2,359,296 (0.42%)

Starting training...

  0%|          | 0/8155 [00:00<?, ?it/s]
  1%|▏         | 100/8155 [02:15<3:01:15, 1.35s/it, loss=2.45]
  2%|▏         | 200/8155 [04:30<2:58:30, 1.35s/it, loss=1.89]
  ...
100%|██████████| 8155/8155 [1:02:15<00:00, 2.18it/s, loss=0.52]

TRAINING COMPLETE!
```

---

### **Step 6: Download Adapter** (2 min)

After training completes, Cell 9 will automatically:
1. Save the adapter
2. Create a ZIP file
3. Trigger download

**File**: `gurukul_flores_adapter.zip` (~12 MB)

---

### **Step 7: Test Locally** (5 min)

1. Extract `gurukul_flores_adapter.zip`
2. Copy contents to `adapters/gurukul_lite/` (overwrite old adapter)
3. Run test script:
   ```bash
   python test_colab_adapter.py
   ```

---

## 📊 Expected Results

### **Training Metrics:**

| Metric | Start | End | Improvement |
|--------|-------|-----|-------------|
| Loss | ~2.5 | ~0.5 | 80% reduction |
| Perplexity | ~12 | ~1.6 | 87% reduction |

### **Generation Quality:**

**Before (monolingual data):**
```
Prompt: Translate to Hindi: Hello friend
Output: - andhonelup...orateckugsotnsa ❌
```

**After (FLORES parallel data):**
```
Prompt: Translate to Hindi: Hello friend
Output: नमस्ते दोस्त ✅
```

---

## 🔧 Troubleshooting

### **"File not found" Error**
- Double-check the path in Cell 2
- Make sure file is actually uploaded to Drive
- Path is case-sensitive!

### **"No GPU detected"**
- Runtime → Change runtime type → GPU → Save
- Restart runtime and try again

### **Training is slow (>2 hours)**
- This means GPU isn't enabled
- Check Cell 1 output - should say "GPU Available: True"

### **Out of memory error**
- Reduce batch size in Cell 7:
  ```python
  per_device_train_batch_size=2,  # Change from 4 to 2
  ```

### **Adapter still echoes prompts**
- Training might not have converged
- Try:
  - More epochs (10 instead of 5)
  - Lower learning rate (1e-4 instead of 2e-4)
  - More training data (download Samanantar)

---

## 📈 Advanced: Better Quality

Want even better results? Use more parallel data!

### **Option 1: More FLORES Data**

Use the full FLORES test set (not just dev + devtest):
- Download from: https://github.com/facebookresearch/flores
- Add `test` split to get 3000+ pairs per language

### **Option 2: Samanantar Dataset**

Get 100k-500k pairs per language:

```python
# In Cell 2, after mounting Drive:
from datasets import load_dataset

# Load Samanantar for Hindi
dataset = load_dataset("ai4bharat/samanantar", "hi", streaming=True)

# Take first 100k pairs
pairs = []
for i, example in enumerate(dataset['train']):
    if i >= 100000:
        break
    pairs.append(f"Translate to Hindi: {example['src']}\n{example['tgt']}")

# Continue with rest of notebook...
```

**Training time**: 4-6 hours  
**Quality improvement**: Significantly better!

---

## 🎯 Success Criteria

After training, you should see:

✅ **Loss < 1.0** (preferably < 0.6)  
✅ **Proper translations in test outputs** (Cell 8)  
✅ **No prompt echoing**  
✅ **Fluent native language output**  

If you see these, your adapter is working! 🎉

---

## 📞 Need Help?

Common issues:
1. **Path errors** → Check Cell 2 path matches your Drive
2. **No GPU** → Enable in Runtime settings
3. **Slow training** → GPU not enabled
4. **Poor quality** → Need more training data or epochs

---

**Good luck with training!** 🚀

You should see MUCH better results than the monolingual training attempt!

