# 🔧 Colab Training Troubleshooting

## ❌ Common Errors & Fixes

---

### **Error 1: "ValueError: Unable to create tensor... different lengths"**

**Full Error:**
```
ValueError: Unable to create tensor, you should probably activate truncation 
and/or padding with 'padding=True' 'truncation=True' to have batched tensors 
with the same length.
```

**Cause:** 
Sequences have different lengths and can't be batched together.

**Fix:**
✅ **ALREADY FIXED in latest notebook!**

The updated notebook (Cell 4 & Cell 6) now:
- Uses dynamic padding (more memory efficient)
- Properly handles labels with `-100` for padding tokens
- Pads to multiples of 8 for GPU optimization

**If you still see this:**
1. Re-download the latest `colab_train_on_your_data.ipynb`
2. Make sure Cell 4 has `padding=False` (let collator handle it)
3. Make sure Cell 6 uses `data_collator_with_padding`

---

### **Error 2: "CUDA out of memory"**

**Full Error:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Cause:**
Model + data don't fit in GPU memory.

**Fix:**
In **Cell 2**, reduce batch size:
```python
config = {
    'batch_size': 2,  # Reduce from 4 to 2
    # Or even 1 if still failing
}
```

Or in **Cell 6**, increase gradient accumulation:
```python
gradient_accumulation_steps=8,  # Increase from 4 to 8
```

**Trade-off:**
- Smaller batch = slower training, same results
- More gradient accumulation = same effective batch, slower training

---

### **Error 3: "No GPU detected"**

**Full Error:**
```
⚠️ WARNING: No GPU detected. Training will be VERY slow!
```

**Cause:**
GPU runtime not enabled.

**Fix:**
1. Click **Runtime → Change runtime type**
2. Select **Hardware accelerator: GPU**
3. Select **GPU type: T4** (free)
4. Click **Save**
5. Restart runtime and run again

---

### **Error 4: "Folder not found" (Google Drive)**

**Full Error:**
```
FileNotFoundError: Missing /content/drive/MyDrive/multilingual_training_data/
```

**Cause:**
Google Drive folder doesn't exist or path is wrong.

**Fix:**
1. Check your Google Drive folder name EXACTLY matches
2. Update `data_folder` in **Cell 3**:
   ```python
   data_folder = "/content/drive/MyDrive/YOUR_ACTUAL_FOLDER_NAME"
   ```

**Common mistakes:**
- Typo in folder name (case-sensitive!)
- Folder is nested: `/content/drive/MyDrive/Projects/multilingual_training_data`
- Folder name has spaces: `multilingual training data` → use underscores or quotes

---

### **Error 5: "Permission denied" (Google Drive)**

**Full Error:**
```
PermissionError: [Errno 13] Permission denied
```

**Cause:**
Colab doesn't have access to your Google Drive.

**Fix:**
1. When Cell 3 runs, you'll see: **"Permit this notebook to access your Google Drive files?"**
2. Click **Connect to Google Drive**
3. Sign in to your Google account
4. Click **Allow**

If you already denied permission:
1. Runtime → Disconnect and delete runtime
2. Runtime → Run all
3. This time, click **Allow** when prompted

---

### **Error 6: "Slow training / No progress"**

**Symptoms:**
- Training step takes >2 minutes each
- Progress bar stuck at 0%
- Loss not decreasing

**Possible causes & fixes:**

**1. CPU mode (no GPU):**
- Check Cell 1 output: should say `GPU Available: True`
- If False, see Error 3 above

**2. Too many workers:**
- In Cell 6, try:
  ```python
  dataloader_num_workers=0,  # Change from 2 to 0
  ```

**3. Large dataset:**
- Reduce samples in Cell 2:
  ```python
  'max_samples': 2000,  # Reduce from 5000
  ```

**4. Colab timeout:**
- Free Colab disconnects after 12 hours
- Upgrade to Colab Pro for longer sessions

---

### **Error 7: "Tokenization taking forever"**

**Symptoms:**
- Cell 4 stuck at "Tokenizing: 0%"

**Cause:**
Too many workers or large dataset.

**Fix:**
In **Cell 4**, reduce batch size:
```python
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=['text'],
    desc="Tokenizing",
    batch_size=10,  # Reduce from 100 to 10
    num_proc=1  # Add this: use single process
)
```

---

### **Error 8: "AttributeError: 'list' object has no attribute 'copy'"**

**Cause:**
Old tokenization code.

**Fix:**
In **Cell 4**, make sure you have:
```python
outputs['labels'] = [ids[:] for ids in outputs['input_ids']]
```

NOT:
```python
outputs['labels'] = outputs['input_ids'].copy()  # ❌ Wrong
```

---

### **Error 9: "Model echoes prompts after training"**

**Symptoms:**
- Training completes successfully
- But generated text is just the prompt repeated

**Possible causes:**

**1. Not enough training data:**
- Try 10,000+ samples instead of 5,000
- In Cell 2: `'max_samples': 10000`

**2. Not enough epochs:**
- Try 5 epochs instead of 3
- In Cell 2: `'num_epochs': 5`

**3. Bad data format:**
- Check your `.txt` files are clean text
- One sentence/paragraph per line
- No HTML tags or special formatting

**4. Learning rate too high/low:**
- Try adjusting in Cell 2:
  ```python
  'learning_rate': 2e-4,  # Default is 3e-4
  ```

---

### **Error 10: "Colab disconnected / Session crashed"**

**Cause:**
Free Colab has usage limits and can disconnect.

**Fixes:**

**Prevent disconnection:**
- Don't leave tab inactive for >30 min
- Use Colab Keep-Alive extension
- Upgrade to Colab Pro

**Resume training:**
If training was interrupted:
1. Re-run all cells
2. Training will resume from last checkpoint (saves every 500 steps)
3. Check `./adapter_training/checkpoint-XXX/` folders

---

## 🆘 Still Stuck?

### **Debug Checklist:**

- [ ] GPU enabled? (Cell 1 should show GPU name)
- [ ] Data uploaded to Drive? (Check folder exists)
- [ ] Path correct in Cell 3? (Exact folder name)
- [ ] Using latest notebook? (Re-download if unsure)
- [ ] Enough GPU memory? (Try smaller batch size)
- [ ] Clean data files? (No corrupted text)

### **Quick Test:**

Run this in a new cell to verify everything:
```python
# Test imports
import torch
print(f"GPU: {torch.cuda.is_available()}")

# Test data path
import os
data_folder = "/content/drive/MyDrive/multilingual_training_data"
print(f"Folder exists: {os.path.exists(data_folder)}")

# Test files
import glob
files = glob.glob(f"{data_folder}/*.txt")
print(f"Found {len(files)} .txt files")

# Test tokenization
sample = "Test sentence"
tokens = tokenizer(sample, return_tensors="pt")
print(f"Tokenization works: {tokens['input_ids'].shape}")
```

All should return `True` or positive numbers!

---

## 💡 Pro Tips

1. **Save checkpoints often:**
   - Cell 6: `save_steps=250` (instead of 500)
   - If Colab crashes, you lose less progress

2. **Test with small dataset first:**
   - Cell 2: `'max_samples': 500`
   - Make sure everything works before full training

3. **Monitor GPU usage:**
   ```python
   !nvidia-smi
   ```
   Shows GPU memory usage

4. **Download checkpoints during training:**
   - Don't wait until the end
   - Download intermediate checkpoints from `adapter_training/`

---

**Good luck!** 🚀

If you encounter an error not listed here, check the error message carefully - it usually tells you exactly what's wrong!

