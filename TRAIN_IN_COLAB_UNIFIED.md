# 🚀 Train Unified Adapter in Colab

## ⚠️ Why Colab?
Your laptop gets **Blue Screen of Death (BSOD)** when training all 29 languages together due to:
- High RAM/VRAM usage (2.4 GB training data)
- Extended GPU load (6-8 hours)
- Overheating risk

**Solution:** Use Google Colab's free T4 GPU!

---

## 📋 Step-by-Step Instructions

### Step 1: Prepare Training Data (On Your PC)

**You already have `training_merged.rar` - perfect!** ✅

The notebook supports: `.zip`, `.rar`, `.tar.gz`, `.7z`

If you need to compress manually:
```cmd
cd C:\pc\Project

# Option A: Windows Explorer
# Right-click data\training_merged\ → Send to → Compressed (zipped) folder

# Option B: WinRAR/7-Zip (if installed)
# Right-click data\training_merged\ → Add to archive...
```

### Step 2: Upload Notebook to Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File** → **Upload notebook**
3. Upload `notebooks/train_unified_gurukul_v2.ipynb`

### Step 3: Enable GPU

1. Click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **T4 GPU**
3. Click **Save**

### Step 4: Upload Training Data

1. In Colab, click the **📁 Files** icon on the left
2. Click the **Upload** button
3. Select `training_merged.rar` (wait for upload to complete - may take 10-15 minutes)
   - The notebook will auto-detect `.rar` format and extract it properly

### Step 5: Run All Cells

1. Click **Runtime** → **Run all**
2. The notebook will:
   - Check GPU
   - Install dependencies
   - Extract and load training data (29 languages)
   - Train the unified adapter (4-6 hours)
   - Save and test the adapter

### Step 6: Keep Colab Alive (Important!)

Colab disconnects after 30 minutes of inactivity. To prevent this:

1. Press **F12** to open browser console
2. Paste this code and press **Enter**:

```javascript
function KeepClicking(){
  console.log("Clicking");
  document.querySelector("colab-connect-button").click()
}
setInterval(KeepClicking, 60000)
```

### Step 7: Download Trained Adapter

After training completes (~4-6 hours):

1. The last cell will create `gurukul_lite_v2.zip`
2. In Files panel, right-click `gurukul_lite_v2.zip` → **Download**
3. Extract to: `C:\pc\Project\adapters\gurukul_lite_v2\`

---

## 📊 What You're Training

- **Languages:** 29 unique (21 original + 8 new)
- **Total Support:** 38 languages (includes 9 bootstrapped dialects)
- **Training Data:** 2.4 GB (~2.8 million samples)
- **Time:** 4-6 hours on T4 GPU
- **Output:** ONE unified adapter

---

## ✅ After Training

1. Extract `gurukul_lite_v2.zip` to `adapters/gurukul_lite_v2/`
2. Test with your API server
3. If it works well, replace `gurukul_lite` with `gurukul_lite_v2`
4. Commit to repository

---

## 🔧 Troubleshooting

**Q: Upload stuck at 99%?**
A: Wait patiently. Large files can take 10-15 minutes.

**Q: Colab disconnected during training?**
A: Training checkpoints are saved every 1000 steps in `gurukul_lite_v2/`. You can resume by re-running the trainer cell.

**Q: Out of memory error?**
A: Reduce `per_device_train_batch_size` from 4 to 2 in cell 9️⃣.

**Q: Training taking too long?**
A: Expected! 4-6 hours is normal for this much data. Leave it running overnight.

---

## 📝 Quick Summary

```
1. You already have: training_merged.rar ✅
2. Upload notebook to Colab
3. Enable T4 GPU
4. Upload training_merged.rar
5. Run all cells (auto-extracts .rar)
6. Keep browser alive (F12 → paste keep-alive code)
7. Wait 4-6 hours
8. Download gurukul_lite_v2.zip
9. Extract to adapters/gurukul_lite_v2/
10. Test & commit!
```

Good luck! 🎉

