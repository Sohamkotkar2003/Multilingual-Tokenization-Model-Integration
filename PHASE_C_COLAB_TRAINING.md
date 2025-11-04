# Phase C: Colab Training Guide

## 🎯 Goal
Fine-tune Gurukul Lite to improve accuracy from **66.7% → 85%+** and fix English mixing issues.

---

## 📋 Prerequisites

1. **Google Account** (for Colab access)
2. **Colab Pro subscription** ($10/month) - Recommended for V100 GPU
3. **Your training data** (`data/training/*.txt` and `data/validation/*.txt`)

---

## 🚀 Quick Start (3 Methods)

### Method 1: Use the Ready-Made Script (Easiest!)

1. Open Google Colab: https://colab.research.google.com
2. Create new notebook
3. Open `colab_training_script.py` from this project
4. **Copy the ENTIRE content** and paste into Colab
5. Upload your data files when prompted
6. Run the cell
7. Wait 2-6 hours
8. Download `gurukul_lite_enhanced.zip`

### Method 2: Use the Jupyter Notebook

1. Upload `notebooks/train_gurukul_lite_enhanced.ipynb` to Colab
2. Follow the step-by-step cells
3. More detailed explanations

### Method 3: Manual Setup

See detailed instructions below.

---

## 📊 Expected Results

**Before Training (Current):**
- Accuracy: 66.7%
- English mixing in 7 languages
- Some off-topic responses

**After Training (Expected):**
- Accuracy: 85-90%
- Minimal English mixing
- Better topic adherence
- Foundation for adding more languages

---

## ⏱️ Time & Cost Estimates

| Hardware | Time | Cost | Success Rate |
|----------|------|------|--------------|
| Colab Free (T4) | 8-12 hrs | $0 | 30-40% (disconnections) |
| Colab Pro (V100) | 4-6 hrs | $10/month | 70-80% |
| Colab Pro+ (A100) | 2-3 hrs | $50/month | 95%+ |

**Recommendation:** Colab Pro ($10) - best value

---

## 📥 After Training: Deployment Steps

1. **Download** `gurukul_lite_enhanced.zip` from Colab
2. **Extract** the zip file
3. **Backup** your current `adapters/gurukul_lite/` folder
4. **Replace** with new trained files
5. **Restart** your server:
   ```bash
   python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8117
   ```
6. **Test** accuracy:
   ```bash
   python test_accuracy_detailed.py
   ```

---

## 🎯 When to Retrain

You should retrain when:
- ✅ Adding new languages to the system
- ✅ Accuracy drops below 60%
- ✅ User feedback indicates quality issues
- ✅ Adding new domains (legal, medical, technical text)

---

## 💡 Tips for Best Results

1. **Clean your training data** - Remove noisy examples
2. **Balance languages** - Don't let one language dominate
3. **Use validation set** - Monitor for overfitting
4. **Save checkpoints** - In case of disconnection
5. **Test immediately** - Verify improvements before deploying

---

## 🆘 Troubleshooting

**Problem:** Colab disconnects before training finishes
- **Solution:** Use Colab Pro or split training into smaller epochs

**Problem:** Out of memory errors
- **Solution:** Reduce batch size to 2, increase gradient_accumulation_steps to 8

**Problem:** Model not improving
- **Solution:** Check data quality, try different learning rates (1e-4, 3e-4)

---

## 📞 Support

For issues with this training:
1. Check Colab logs for error messages
2. Verify data files uploaded correctly
3. Ensure GPU is enabled (Runtime → Change runtime type → GPU)

---

**Ready to achieve 85%+ accuracy? Let's train!** 🚀

