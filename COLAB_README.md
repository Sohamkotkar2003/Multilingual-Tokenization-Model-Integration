# 🚀 Google Colab Training Script

This repository contains a standalone script that can be run in Google Colab to fine-tune a language model on multilingual data (Hindi, Sanskrit, Marathi, English).

## 📁 Files

- `colab_training_notebook.ipynb` - Main Jupyter notebook for Google Colab
- `colab_training_script.py` - Python script version (can be copied to Colab)
- `COLAB_README.md` - This instruction file

## 🚀 Quick Start in Google Colab

### Step 1: Open Google Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook or upload the provided `colab_training_notebook.ipynb`

### Step 2: Enable GPU
1. Go to `Runtime` → `Change runtime type`
2. Set `Hardware accelerator` to `GPU`
3. Click `Save`

### Step 3: Setup Your Data
1. Upload your data to Google Drive in the structure shown above
2. Make sure your files are in the correct folder: `Google Drive/Data/training/` and `Google Drive/Data/validation/`

### Step 4: Run the Notebook
1. Run each cell in sequence (Shift+Enter)
2. When prompted, authorize Google Drive access
3. The script will:
   - Install required dependencies
   - Mount your Google Drive
   - Load your training data from Google Drive
   - Configure and train the model with LoRA/PEFT
   - Save and package the trained model
   - Provide download link

### Step 5: Download Your Model
- The trained model will be automatically downloaded as a zip file
- Extract it to use with your inference scripts

## 🔧 Configuration Options

You can modify these settings in the notebook:

```python
MODEL_NAME = "AhinsaAI/ahinsa0.5-llama3.2-3B"  # Change to your preferred model
EPOCHS = 2  # Number of training epochs
BATCH_SIZE = 1  # Batch size (adjust based on GPU memory)
MAX_LENGTH = 512  # Maximum sequence length
USE_QUANTIZATION = True  # Enable 8-bit quantization
USE_PEFT = True  # Enable LoRA for efficient training
```

## 📊 Data Requirements

The script expects your data to be stored in Google Drive with the following structure:

```
Google Drive/
└── Data/
    ├── training/
    │   ├── hi_train.txt    (Hindi training data)
    │   ├── sa_train.txt    (Sanskrit training data)
    │   ├── mr_train.txt    (Marathi training data)
    │   └── en_train.txt    (English training data)
    └── validation/
        ├── hi_val.txt      (Hindi validation data)
        ├── sa_val.txt      (Sanskrit validation data)
        ├── mr_val.txt      (Marathi validation data)
        └── en_val.txt      (English validation data)
```

**Note**: If your files have different names, you can modify the file name configuration in the notebook.

## 🎯 Features

- ✅ **Memory Optimized**: Designed for Colab's GPU constraints
- ✅ **LoRA/PEFT**: Efficient fine-tuning with minimal parameters
- ✅ **8-bit Quantization**: Reduces memory usage during training
- ✅ **Progress Tracking**: Real-time training progress and loss monitoring
- ✅ **Automatic Packaging**: Creates downloadable zip file
- ✅ **Google Drive Integration**: Loads data directly from your Google Drive
- ✅ **Error Handling**: Graceful handling of memory issues

## 🔧 Troubleshooting

### Out of Memory Errors
If you encounter CUDA out of memory errors:
1. Reduce `BATCH_SIZE` to 1
2. Reduce `MAX_LENGTH` to 256 or 128
3. Ensure `USE_QUANTIZATION = True`
4. Ensure `USE_PEFT = True`

### Slow Training
- The script is optimized for Colab's free GPU (T4)
- Training time depends on your data size and epochs
- With sample data: ~5-10 minutes
- With larger datasets: 30+ minutes

### Model Download Issues
- Ensure you run the download cell after training completes
- Check your browser's download settings
- The model is packaged as a zip file for easy download

## 📝 Custom Data Configuration

To use your own data with different file names:

1. Upload your data to Google Drive in the `Data` folder
2. Modify the `TRAINING_FILES` and `VALIDATION_FILES` dictionaries in the notebook
3. Ensure your data is in UTF-8 format
4. Each file should contain one text sample per line

Example for custom file names:
```python
TRAINING_FILES = {
    "hindi": "my_hindi_data.txt",
    "sanskrit": "my_sanskrit_data.txt", 
    "marathi": "my_marathi_data.txt",
    "english": "my_english_data.txt"
}
```

## 🎉 Next Steps

After training:
1. Download the model zip file
2. Extract it to your local machine
3. Use the model with your inference scripts
4. The model supports text generation in all trained languages

## 📞 Support

If you encounter issues:
1. Check the Colab runtime logs
2. Verify GPU is enabled
3. Ensure all dependencies are installed
4. Check available disk space in Colab

Happy training! 🚀
