# 📚 Parallel Translation Data Sources

## ✅ **Free & High-Quality Sources**

### **1. Samanantar (BEST for Indian Languages)** ⭐⭐⭐⭐⭐
- **What**: 49M+ English↔Indian language pairs
- **Languages**: All 22 Indian languages including Hindi, Bengali, Tamil, Telugu, etc.
- **Size**: Massive (10M+ pairs per language)
- **Quality**: Excellent
- **Link**: https://indicnlp.ai4bharat.org/samanantar/
- **Download**: https://github.com/AI4Bharat/IndicNLP

**Example data:**
```
en: Hello, how are you?
hi: नमस्ते, आप कैसे हैं?

en: Good morning
bn: সুপ্রভাত
```

---

### **2. OPUS Corpus** ⭐⭐⭐⭐
- **What**: Multi-domain parallel corpora
- **Languages**: 100+ language pairs
- **Size**: Varies (10k - 10M per pair)
- **Quality**: Good
- **Link**: https://opus.nlpl.eu/

**Subsets:**
- **OpenSubtitles**: Movie subtitles (casual/conversational)
- **Wikipedia**: Formal/encyclopedic
- **News**: Current events
- **ParaCrawl**: Web-scraped

---

### **3. IndicNLP Corpus** ⭐⭐⭐⭐
- **What**: Monolingual + some parallel data
- **Languages**: Indian languages
- **Size**: Medium
- **Quality**: Very good
- **Link**: https://github.com/ai4bharat/indicnlp_corpus

---

### **4. CCAligned** ⭐⭐⭐
- **What**: Mined parallel sentences from web
- **Languages**: 100+ languages
- **Size**: Large (millions)
- **Quality**: Medium
- **Link**: http://www.statmt.org/cc-aligned/

---

### **5. FLORES-101** ⭐⭐⭐⭐⭐
- **What**: High-quality evaluation + small training set
- **Languages**: 101 languages
- **Size**: Small (3001 sentences)
- **Quality**: Excellent (human-verified)
- **Link**: https://github.com/facebookresearch/flores

---

## 📥 **How to Download Samanantar (Recommended)**

### **Step 1: Install IndicNLP**
```bash
pip install indic-nlp-library
git clone https://github.com/AI4Bharat/samanantar.git
```

### **Step 2: Download Data**
```bash
cd samanantar
python download_samanantar.py --language hi --split train
```

**Languages:**
- `hi` - Hindi
- `bn` - Bengali
- `ta` - Tamil
- `te` - Telugu
- `gu` - Gujarati
- `mr` - Marathi
- `ur` - Urdu
- `pa` - Punjabi
- `kn` - Kannada
- `ml` - Malayalam
- `or` - Odia
- `as` - Assamese

### **Step 3: Format for Training**
```python
import pandas as pd

# Load samanantar data
df = pd.read_csv('hi-en.tsv', sep='\t', names=['english', 'hindi'])

# Create instruction format
with open('hi_translation_pairs.txt', 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        # Format: Translate to Hindi: [English]\n[Hindi]\n
        f.write(f"Translate to Hindi: {row['english']}\n{row['hindi']}\n\n")
```

---

## 🚀 **Quick Start Script**

I'll create a script to download and format Samanantar data for you:

```python
# download_samanantar.py
import urllib.request
import gzip
import shutil
from pathlib import Path

def download_samanantar_language(lang_code, output_dir='samanantar_data'):
    """
    Download Samanantar parallel data for a language
    """
    base_url = "https://storage.googleapis.com/samanantar-public/V0.2"
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Download compressed file
    url = f"{base_url}/data/en-{lang_code}.zip"
    output_file = f"{output_dir}/en-{lang_code}.zip"
    
    print(f"Downloading {lang_code}...")
    urllib.request.urlretrieve(url, output_file)
    
    # Extract
    shutil.unpack_archive(output_file, output_dir)
    print(f"✅ Downloaded and extracted {lang_code}")

# Download top languages
for lang in ['hi', 'bn', 'ta', 'te', 'gu']:
    download_samanantar_language(lang)
```

---

## 📊 **Data Size Recommendations**

| Goal | Samples Needed | Quality |
|------|---------------|---------|
| Basic translation | 10,000+ pairs | Fair |
| Good translation | 50,000+ pairs | Good |
| Excellent translation | 200,000+ pairs | Excellent |
| Production-quality | 1M+ pairs | Production |

---

## 💡 **Pro Tips**

### **1. Data Quality > Quantity**
- 10k high-quality pairs > 100k noisy pairs
- Use FLORES for evaluation

### **2. Domain Matching**
- News data → Good for formal text
- Subtitles → Good for conversation
- Wikipedia → Good for encyclopedic

### **3. Balanced Sampling**
- Sample equally from all languages
- Don't oversample one language

### **4. Data Cleaning**
```python
# Remove duplicates
df = df.drop_duplicates()

# Remove too short/long
df = df[(df['english'].str.len() > 10) & (df['english'].str.len() < 500)]

# Remove URLs, HTML
import re
df['english'] = df['english'].apply(lambda x: re.sub(r'http\S+', '', x))
```

---

## ⚡ **Fast Track (1 Hour Setup)**

1. **Download FLORES-101** (3k sentences, all languages)
   ```bash
   wget https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz
   tar -xf flores101_dataset.tar.gz
   ```

2. **Format for BLOOMZ**
   ```python
   # Create instruction pairs from FLORES
   with open('flores_eng_Latn.dev', 'r') as en, \
        open('flores_hin_Deva.dev', 'r') as hi, \
        open('train_data.txt', 'w', encoding='utf-8') as out:
       for en_line, hi_line in zip(en, hi):
           out.write(f"Translate to Hindi: {en_line.strip()}\n{hi_line.strip()}\n\n")
   ```

3. **Upload to Colab & Train** (20 min)

**Total time: ~1 hour for working translation adapter!**

---

## 🎯 **Best Strategy for YOU**

Given your time constraints:

1. **Download FLORES-101** (3k high-quality pairs) ✅ Fast
2. **Or download Samanantar subset** (10k pairs per language) ⚡ 1 hour
3. **Format as instruction pairs** 📝 Easy
4. **Train in Colab** 🚀 20-30 min
5. **See REAL improvement!** 🎉

---

## 📞 **Need Help?**

I can create a complete script to:
1. Download Samanantar
2. Format it for BLOOMZ training
3. Upload to Google Drive
4. Train in Colab

**Just ask!** 😊

