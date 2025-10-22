# ⚡ Quick Start: Train Adapter on Colab with Google Drive

## 🎯 **Goal**
Train a multilingual adapter on YOUR data using Google Colab (FREE GPU!) + Google Drive (easy upload)

---

## 📋 **Checklist (5 Steps)**

- [ ] **1. Upload data to Google Drive** (2-5 min)
- [ ] **2. Open notebook in Colab** (1 min)
- [ ] **3. Enable GPU** (1 min)
- [ ] **4. Run training** (20-30 min)
- [ ] **5. Download & test adapter** (5 min)

**Total time:** ~30-45 minutes

---

## 🚀 **Step 1: Upload to Google Drive** (2-5 min)

1. Go to https://drive.google.com/
2. Create folder: `multilingual_training_data`
3. Upload all `.txt` files from your `data/training/` folder
   - `hi_train.txt`, `bn_train.txt`, `ta_train.txt`, etc.
4. ✅ Done!

---

## 🚀 **Step 2: Open Notebook in Colab** (1 min)

1. Go to https://colab.research.google.com/
2. Click **File → Upload notebook**
3. Select `colab_train_on_your_data.ipynb` from your project
4. ✅ Notebook opened!

---

## 🚀 **Step 3: Enable GPU** (1 min)

1. Click **Runtime → Change runtime type**
2. Select **GPU** (T4 is free)
3. Click **Save**
4. ✅ GPU enabled!

---

## 🚀 **Step 4: Run Training** (20-30 min)

1. Click **Runtime → Run all**
2. When prompted:
   - **"Connect to Google Drive?"** → Click **Allow**
3. Watch the training progress!
4. ✅ Training complete when you see "🎉 TRAINING COMPLETE!"

---

## 🚀 **Step 5: Download & Test** (5 min)

1. Download `gurukul_adapter.zip` (auto-downloads at end)
2. Extract to `adapters/gurukul_lite/` in your project folder
3. Test it:
   ```bash
   python test_colab_adapter.py
   ```
4. ✅ Adapter working!

---

## ⚙️ **Configuration (Optional)**

### **Default Settings:**
- **Samples:** 5,000 total (balanced across languages)
- **Epochs:** 3
- **Batch Size:** 4
- **Training Time:** ~25 minutes on T4 GPU

### **To Adjust:**
Open **Cell 2** in the notebook and modify:
```python
config = {
    'max_samples': 10000,  # More samples = better quality, longer training
    'num_epochs': 5,       # More epochs = better learning
}
```

---

## 📁 **Files You Need**

From your project:
1. `colab_train_on_your_data.ipynb` - Upload to Colab
2. `data/training/*.txt` - Upload to Google Drive
3. `test_colab_adapter.py` - Run after training

---

## ❓ **Common Issues**

### **"Folder not found"**
→ Check that your Google Drive folder is named exactly: `multilingual_training_data`
→ Or update `data_folder` in Cell 3 to match your folder name

### **"No GPU detected"**
→ Runtime → Change runtime type → GPU → Save

### **"Out of memory"**
→ In Cell 2, change `'batch_size': 2` instead of 4

### **Adapter still echoes prompts**
→ Try training with more samples: `'max_samples': 10000` or `20000`

---

## ✅ **Success Criteria**

After training, your adapter should:
- ✅ Generate Hindi: `नमस्ते दोस्त`
- ✅ Generate Bengali: `সুপ্রভাত`
- ✅ Generate Tamil: `நன்றி`
- ❌ NOT just echo the English prompt!

---

## 📚 **Additional Help**

- **Detailed Google Drive Setup:** See `GOOGLE_DRIVE_SETUP.md`
- **Full Instructions:** See `TRAIN_WITH_YOUR_DATA.md`
- **Troubleshooting:** See `HOW_TO_USE_COLAB.txt`

---

## 💡 **Why This Works**

Previous adapter training failed because:
- ❌ Used fake sample data
- ❌ Trained on CPU (slow/unstable)
- ❌ Windows memory issues

This new approach:
- ✅ Uses YOUR real multilingual data
- ✅ Trains on Colab GPU (fast/stable)
- ✅ No local memory issues
- ✅ Google Drive = easy upload

---

**Ready? Let's go!** 🚀

Just follow the 5 steps above and you'll have a working adapter in ~40 minutes!

