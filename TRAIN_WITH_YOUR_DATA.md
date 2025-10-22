# 🚀 Train BLOOMZ-560M on YOUR Data (Google Colab)

## 🎯 Goal
Train a multilingual adapter using **your actual training data** from `data/training/`.

## 📝 Step-by-Step Instructions

### **Step 1: Prepare Your Data** 
Before opening Colab, gather all your training files:
- `data/training/hi_train.txt` (Hindi)
- `data/training/bn_train.txt` (Bengali)
- `data/training/ta_train.txt` (Tamil)
- `data/training/te_train.txt` (Telugu)
- `data/training/gu_train.txt` (Gujarati)
- `data/training/mr_train.txt` (Marathi)
- `data/training/ur_train.txt` (Urdu)
- `data/training/pa_train.txt` (Punjabi)
- And any other `.txt` files you have!

### **Step 2: Open Notebook in Colab**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → Upload notebook**
3. Upload `colab_train_on_your_data.ipynb` from this project

### **Step 3: Enable GPU**
1. Click **Runtime → Change runtime type**
2. Select **GPU** (T4 is free, A100 is faster but paid)
3. Click **Save**

### **Step 4: Upload Your Training Data**

**📌 RECOMMENDED: Use Google Drive (Much Faster!)**

#### **Option A: Google Drive** ✅ (Recommended)
1. Go to [Google Drive](https://drive.google.com/)
2. Create a new folder called `multilingual_training_data`
3. Upload ALL your `.txt` files from `data/training/`:
   - `hi_train.txt`, `bn_train.txt`, `ta_train.txt`, `te_train.txt`, etc.
4. In the Colab notebook (Cell 3), keep `USE_GOOGLE_DRIVE = True`
5. The notebook will automatically mount your Drive and load the data

**⏱️ Upload time:** ~2-5 minutes (Google Drive is fast!)

#### **Option B: Direct Upload to Colab** (Slower)
1. In Colab, click the **📁 folder icon** on the left sidebar
2. Right-click in the file browser → **New folder**
3. Name it `training_data`
4. Click on the `training_data` folder
5. Click the **📄 upload** icon
6. Upload ALL your `.txt` files from `data/training/`
7. In Cell 3, change `USE_GOOGLE_DRIVE = False`

**⏱️ Upload time:** ~10-15 minutes (slower than Drive)

### **Step 5: Run Training**
1. Click **Runtime → Run all**
2. Wait for training to complete (~20-30 minutes)
3. Watch the progress bars and loss metrics

### **Step 6: Download the Trained Adapter**
1. After training completes, Cell 8 will automatically trigger a download
2. Save `gurukul_adapter.zip` to your PC
3. Extract it to `adapters/gurukul_lite/` in your project folder

### **Step 7: Test Locally**
```bash
python test_colab_adapter.py
```

## 📊 What to Expect

### Training Config (default):
- **Samples:** 5,000 total (balanced across languages)
- **Epochs:** 3
- **Batch Size:** 4
- **LoRA Rank:** 8
- **Learning Rate:** 3e-4

### Training Time:
- **T4 GPU (free):** ~25-30 minutes
- **A100 GPU (paid):** ~10-15 minutes

### Output Quality:
Since this uses YOUR actual multilingual data, the adapter should:
- ✅ Generate proper translations (not echo prompts)
- ✅ Maintain language-specific patterns
- ✅ Handle multiple Indian languages correctly

## 🔧 Customization

You can adjust training parameters in **Cell 2** of the notebook:

```python
config = {
    'max_samples': 10000,     # More samples = better quality, longer training
    'max_length': 512,        # Longer sequences = more context
    'num_epochs': 5,          # More epochs = better learning
    'batch_size': 2,          # Smaller = less VRAM, slower training
    'learning_rate': 2e-4,    # Lower = more stable, slower convergence
    'lora_r': 16,             # Higher = more capacity, more VRAM
}
```

## ❓ Troubleshooting

### "No GPU detected"
→ Go to **Runtime → Change runtime type → GPU**

### "training_data folder not found"
→ Make sure you created the folder AND uploaded files into it

### "Out of memory"
→ Reduce `batch_size` to 2 or `max_samples` to 2000

### Training is too slow
→ Make sure GPU is enabled (see above)

### Adapter still echoes prompts
→ Try:
  - Increase `max_samples` to 10,000+
  - Increase `num_epochs` to 5
  - Check your training data format (should be clean text, one sentence per line)

## 🎉 Success Criteria

After downloading and testing the adapter, you should see:
- **Hindi:** `नमस्ते दोस्त, आप कैसे हैं?`
- **Bengali:** `সুপ্রভাত।`
- **Tamil:** `நன்றி.`

NOT just echoing the English prompt!

## 💡 Pro Tips

1. **Balanced Data:** Upload roughly equal amounts of data per language
2. **Clean Data:** Remove any corrupted/garbled text before uploading
3. **More Data = Better:** If training works well, try 10k-20k samples next time
4. **Save Checkpoints:** Colab keeps checkpoints in `adapter_training/` - download them too!

---

**Good luck!** 🚀

