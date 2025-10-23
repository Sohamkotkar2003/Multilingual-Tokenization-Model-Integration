# 🔥 NLLB-200 Smoke Test Guide

## Overview

The `smoke_test_nllb_colab.ipynb` notebook provides a **comprehensive quality assessment** of your trained NLLB-200 adapter across all 21 languages.

---

## What It Does

### 📊 Testing Coverage

- **21 Languages**: All languages from your training data (Assamese, Bengali, Bodo, Gujarati, Hindi, Kannada, Kashmiri, Maithili, Malayalam, Manipuri, Marathi, Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu, English)
- **10 Diverse Prompts per Language**: Common phrases and sentences
- **210 Total Tests**: Comprehensive coverage

### 🎯 Test Prompts

The notebook tests with diverse real-world phrases:
1. "Hello, how are you today?"
2. "Thank you very much for your help."
3. "What is your name?"
4. "Good morning! Have a nice day."
5. "I love learning new languages."
6. "The weather is beautiful today."
7. "Please help me with this task."
8. "Where is the nearest hospital?"
9. "This is a wonderful opportunity."
10. "Welcome to our home."

---

## How to Use

### Step 1: Upload to Google Colab

1. Go to https://colab.research.google.com/
2. Click **File** → **Upload notebook**
3. Upload `smoke_test_nllb_colab.ipynb`
4. Make sure **GPU** is enabled:
   - Click **Runtime** → **Change runtime type**
   - Set **Hardware accelerator** to **GPU (T4)**

### Step 2: Run the Notebook

1. **Cell 1-2**: Install packages (~2 minutes)
2. **Cell 3-4**: Upload your `nllb_18languages_adapter.zip` file
3. **Cell 5-6**: Load model and adapter (~3-4 minutes)
4. **Cell 7-8**: Define test configuration (instant)
5. **Cell 9-10**: Define translation function (instant)
6. **Cell 11-12**: **RUN SMOKE TESTS** (~5-10 minutes for 210 tests)
7. **Cell 13-14**: Generate markdown report (instant)
8. **Cell 15-16**: Download report automatically
9. **Cell 17-18** (Optional): Preview report in notebook
10. **Cell 19-20** (Optional): Generate performance charts

### Step 3: Review Results

The notebook will automatically download:
- **`nllb_smoke_results_YYYYMMDD_HHMMSS.md`**: Detailed markdown report
- **`performance_chart_YYYYMMDD_HHMMSS.png`**: Visual performance comparison

---

## What You Get

### 📄 Markdown Report

The generated report includes:

#### Executive Summary
- Total translation time
- Average time per translation
- Throughput (translations/second)

#### Per-Language Performance Table
```markdown
| Language | NLLB Code | Avg Time | Samples |
|----------|-----------|----------|---------|
| Hindi    | hin_Deva  | 0.45s    | 10      |
| Tamil    | tam_Taml  | 0.52s    | 10      |
| ...      | ...       | ...      | ...     |
```

#### Detailed Results
For each language and each test prompt:
- Input (English)
- Output (Target language)
- Translation time

#### Conclusion
- Overall quality assessment
- Production readiness

### 📊 Performance Chart

A beautiful bar chart showing:
- Average translation time for each language
- Overall average (red dashed line)
- Color-coded for easy comparison

---

## Expected Performance

Based on your previous testing:

- **Average Time**: 0.5-1.0s per translation
- **Total Test Time**: 5-10 minutes for all 210 tests
- **Quality**: 85-90% accuracy (based on manual review)
- **GPU Memory**: ~3-4 GB

---

## Troubleshooting

### Issue: "No GPU available"
**Solution**: 
1. Click **Runtime** → **Change runtime type**
2. Set **Hardware accelerator** to **GPU**
3. Restart notebook

### Issue: "Out of memory"
**Solution**:
1. Click **Runtime** → **Factory reset runtime**
2. Re-run from the beginning
3. Close other Colab notebooks

### Issue: "Model takes too long to load"
**Solution**:
- This is normal! NLLB-200 is 600M parameters
- First load: ~3-4 minutes
- Be patient, it will complete

### Issue: "Wrong language generated"
**Solution**:
- The notebook already handles this correctly
- Uses `forced_bos_token_id` to ensure target language
- If you see issues, the adapter may need more training

---

## Interpreting Results

### ✅ Good Signs

- **Correct Script**: Output uses the right writing system (Devanagari, Tamil, etc.)
- **Reasonable Translation**: Output makes sense in target language
- **Consistent Speed**: Similar times across languages (~0.5-1.0s)
- **No English Fallback**: Output is in target language, not English

### ⚠️ Warning Signs

- **Mixed Scripts**: Output contains multiple writing systems
- **English Words**: Target language output contains untranslated English
- **Very Slow**: Some languages take 2-3x longer than others
- **Empty Output**: Blank or very short translations

### ❌ Issues to Address

- **Wrong Language**: Gujarati input → Telugu output
- **Gibberish**: Random characters or nonsensical output
- **Always Same Output**: All prompts produce identical translation
- **Crashes**: Notebook runs out of memory or crashes

---

## Next Steps After Testing

### If Results are Good (85-90%+):

1. **Production Deployment**
   - Update your API to use NLLB-200 adapter
   - Deploy to your server
   - Start serving real traffic

2. **Documentation**
   - Add the smoke test results to your docs
   - Update API documentation with supported languages
   - Share results with stakeholders

3. **Monitoring**
   - Set up quality monitoring
   - Collect user feedback
   - Track performance metrics

### If Results Need Improvement (<85%):

1. **Train Longer**
   - Increase epochs from 3 → 5-10
   - Expected improvement: +5-10% quality

2. **Add More Data**
   - Include domain-specific parallel data
   - Mix FLORES-200 with your custom data

3. **Tune Hyperparameters**
   - Adjust learning rate
   - Modify LoRA rank
   - Try different batch sizes

4. **Try Different Base Model**
   - NLLB-200-1.3B (larger model)
   - mBART50 (alternative architecture)

---

## Comparison with Previous Results

### BLOOMZ-560M (Previous)
- **Quality**: 50-70%
- **Issues**: Chinese for Gujarati, English for Telugu
- **Speed**: 0.5-1.5s per translation

### NLLB-200-600M (Current)
- **Quality**: 87.5% (tested on 8 languages)
- **Issues**: Fixed all script issues!
- **Speed**: 0.3-0.8s per translation
- **Improvement**: +30-37 percentage points! 🎉

---

## Report Example

Here's what a typical result section looks like:

```markdown
### Hindi (hin_Deva)

#### Test 1/10

**Input (English):**
> Hello, how are you today?

**Output (Hindi):**
> नमस्ते, आप आज कैसे हैं?

**Time:** 0.52s

---

#### Test 2/10

**Input (English):**
> Thank you very much for your help.

**Output (Hindi):**
> आपकी मदद के लिए बहुत धन्यवाद।

**Time:** 0.48s

---
```

---

## Technical Details

### Model Configuration
- **Base Model**: facebook/nllb-200-distilled-600M
- **Adapter**: Your LoRA-fine-tuned adapter
- **Precision**: FP16 (float16)
- **Device**: GPU (CUDA)

### Translation Parameters
- **Beam Search**: 5 beams
- **Max Length**: 128 tokens
- **Early Stopping**: Enabled
- **Source Language**: English (eng_Latn)
- **Target Languages**: 21 Indian languages (various scripts)

### Performance Metrics
- **Per-Translation Time**: Individual translation speed
- **Average Time**: Mean across all tests
- **Throughput**: Translations per second
- **Total Time**: End-to-end test duration

---

## Support

If you encounter any issues:

1. Check the **Troubleshooting** section above
2. Review the Colab notebook's inline comments
3. Verify your adapter was uploaded correctly
4. Check GPU is enabled and available

---

## Summary

The smoke test notebook provides:
- ✅ **Comprehensive coverage**: All 21 languages
- ✅ **Diverse testing**: 10 different prompts
- ✅ **Professional reporting**: Beautiful markdown + charts
- ✅ **Easy to use**: Upload, run, download results
- ✅ **Production ready**: Validate before deployment

**Total time investment**: ~15-20 minutes
**Value**: Confidence in your adapter's quality across all languages!

---

*Last updated: 2025-01-23*

