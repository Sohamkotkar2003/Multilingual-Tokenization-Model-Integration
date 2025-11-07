# 🚀 Quick Start - Train in Colab

## ✅ You Have RAR - Perfect!

Your `training_merged.rar` file is ready to go! The notebook auto-detects and extracts `.rar` files.

---

## 📋 5-Minute Setup

### 1. Upload Notebook
- Go to https://colab.research.google.com/
- **File** → **Upload notebook**
- Choose: `notebooks/train_unified_gurukul_v2.ipynb`

### 2. Enable GPU
- **Runtime** → **Change runtime type**
- Set: **T4 GPU**
- Click: **Save**

### 3. Upload Your Data
- Click **📁 Files** panel
- Click **Upload**
- Select: `training_merged.rar`
- Wait: ~10-15 minutes

### 4. Run Training
- **Runtime** → **Run all**
- The notebook will:
  - Install `unrar`
  - Auto-detect your `.rar` file
  - Extract 29 languages
  - Train for 4-6 hours

### 5. Keep Alive (Important!)
Press **F12**, paste this, press **Enter**:

```javascript
function KeepClicking(){
  console.log("Clicking");
  document.querySelector("colab-connect-button").click()
}
setInterval(KeepClicking, 60000)
```

### 6. Download (After 4-6 hours)
- Right-click `gurukul_lite_v2.zip` in Files panel
- Download
- Extract to: `C:\pc\Project\adapters\gurukul_lite_v2\`

---

## 🎯 What You Get

**ONE unified adapter** supporting **38 languages**:
- 29 trained (21 original + 8 new)
- 9 bootstrapped dialects

---

## ⚠️ Troubleshooting

**Upload stuck?** → Wait patiently (large file)

**Out of memory?** → In cell 9, change `per_device_train_batch_size=4` to `2`

**Disconnected?** → Did you paste the keep-alive code? (Step 5)

---

**Full details:** See `TRAIN_IN_COLAB_UNIFIED.md`

**Go!** 🚀

