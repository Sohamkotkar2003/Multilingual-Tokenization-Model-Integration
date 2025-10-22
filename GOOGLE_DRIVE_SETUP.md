# 📁 Google Drive Setup for Colab Training

## ✅ **Why Use Google Drive?**

- **Faster:** Upload once to Drive, use in any Colab session
- **Persistent:** Data stays in Drive even if Colab disconnects
- **Reusable:** Run training multiple times without re-uploading
- **Easier:** Simple drag-and-drop upload in your browser

---

## 🚀 **Step-by-Step Setup**

### **Step 1: Upload Data to Google Drive**

1. **Open Google Drive:**
   - Go to https://drive.google.com/
   - Sign in with your Google account

2. **Create a New Folder:**
   - Click **+ New** → **Folder**
   - Name it: `multilingual_training_data`
   - Click **Create**

3. **Upload Your Training Files:**
   - Open the `multilingual_training_data` folder
   - Click **+ New** → **File upload**
   - Navigate to your project's `data/training/` folder
   - Select ALL `.txt` files:
     - `hi_train.txt` (Hindi)
     - `bn_train.txt` (Bengali)
     - `ta_train.txt` (Tamil)
     - `te_train.txt` (Telugu)
     - `gu_train.txt` (Gujarati)
     - `mr_train.txt` (Marathi)
     - `ur_train.txt` (Urdu)
     - `pa_train.txt` (Punjabi)
     - `kn_train.txt` (Kannada)
     - `ml_train.txt` (Malayalam)
     - `or_train.txt` (Odia)
     - `as_train.txt` (Assamese)
     - `ne_train.txt` (Nepali)
     - `sa_train.txt` (Sanskrit)
     - And any others!
   - Click **Open** to start upload

4. **Wait for Upload to Complete:**
   - You'll see a progress indicator in the bottom-right
   - ⏱️ Upload time: ~2-5 minutes (depending on your internet speed)

---

### **Step 2: Update Colab Notebook**

1. **Open the Colab Notebook:**
   - Upload `colab_train_on_your_data.ipynb` to Colab
   - Or open it if already uploaded

2. **Find Cell 3** (Load YOUR Training Data)

3. **Verify Settings:**
   ```python
   USE_GOOGLE_DRIVE = True  # Should be True
   data_folder = "/content/drive/MyDrive/multilingual_training_data"
   ```

4. **If You Named Your Folder Differently:**
   - Update `data_folder` to match your folder name
   - Example: If you named it `my_training_files`:
     ```python
     data_folder = "/content/drive/MyDrive/my_training_files"
     ```

---

### **Step 3: Run the Notebook**

1. **Click Runtime → Run all**
2. **When prompted "Mount Google Drive?":**
   - Click **Connect to Google Drive**
   - Sign in with your Google account
   - Click **Allow** to grant Colab access to your Drive
3. **Watch the training progress!**

---

## 📂 **Google Drive Folder Structure**

After setup, your Google Drive should look like:

```
My Drive/
└── multilingual_training_data/
    ├── hi_train.txt
    ├── bn_train.txt
    ├── ta_train.txt
    ├── te_train.txt
    ├── gu_train.txt
    ├── mr_train.txt
    ├── ur_train.txt
    ├── pa_train.txt
    ├── kn_train.txt
    ├── ml_train.txt
    ├── or_train.txt
    ├── as_train.txt
    ├── ne_train.txt
    ├── sa_train.txt
    └── ... (other languages)
```

---

## ❓ **Troubleshooting**

### **"Folder not found" error in Colab**

**Solution 1:** Check folder name
- Make sure the folder name in Cell 3 EXACTLY matches your Drive folder name
- Google Drive is case-sensitive!
- No typos or extra spaces

**Solution 2:** Check folder location
- The folder should be directly in "My Drive", not inside another folder
- If it's nested (e.g., `My Drive/Projects/multilingual_training_data/`), update the path:
  ```python
  data_folder = "/content/drive/MyDrive/Projects/multilingual_training_data"
  ```

**Solution 3:** Re-mount Drive
- Run this in a new cell:
  ```python
  from google.colab import drive
  drive.flush_and_unmount()
  drive.mount('/content/drive', force_remount=True)
  ```

### **"Permission denied" error**

- Make sure you clicked **Allow** when Colab asked for Drive access
- Try disconnecting and reconnecting:
  - Runtime → Disconnect and delete runtime
  - Runtime → Run all (start fresh)

### **Upload is taking forever**

- **Google Drive upload is usually fast (2-5 min)**
- If it's stuck:
  - Check your internet connection
  - Try uploading a few files at a time instead of all at once
  - Consider compressing files into a `.zip` first, then extract in Colab

---

## 💡 **Pro Tips**

1. **Organize by Language:**
   ```
   My Drive/
   └── multilingual_training_data/
       ├── hindi/
       │   └── hi_train.txt
       ├── bengali/
       │   └── bn_train.txt
       └── ...
   ```
   Then update Cell 3 to search recursively:
   ```python
   txt_files = glob.glob(f"{data_folder}/**/*.txt", recursive=True)
   ```

2. **Keep Backups:**
   - Google Drive = automatic backup!
   - Your training data is safe even if your PC crashes

3. **Share Across Team:**
   - Right-click folder → Share
   - Team members can access the same data for training

4. **Reuse for Multiple Trainings:**
   - Once uploaded, you can run training multiple times
   - No need to re-upload!

---

## ⏱️ **Time Comparison**

| Method | Upload Time | Setup Difficulty | Persistence |
|--------|-------------|------------------|-------------|
| **Google Drive** | 2-5 min | Easy | ✅ Permanent |
| Direct to Colab | 10-15 min | Harder | ❌ Lost on disconnect |

**Winner: Google Drive!** ✅

---

## 🎉 **You're Ready!**

Once your files are in Google Drive, you can:
- ✅ Run training in Colab
- ✅ Reuse data for future training sessions
- ✅ Share with collaborators
- ✅ Access from any device

**Happy Training!** 🚀

