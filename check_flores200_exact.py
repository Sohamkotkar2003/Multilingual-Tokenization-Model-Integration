#!/usr/bin/env python3
"""
Check FLORES-200 official language list for your 21 languages
Based on official FLORES-200 documentation
"""
import sys
import io

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Your 21 languages
your_languages = {
    "as": "Assamese",
    "bd": "Bodo",
    "bn": "Bengali", 
    "en": "English",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "ks": "Kashmiri",
    "mai": "Maithili",
    "ml": "Malayalam",
    "mni": "Meitei/Manipuri",
    "mr": "Marathi",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu"
}

# FLORES-200 Official Language List (204 languages)
# Based on: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
# Format: language_Script (ISO 639-3 + script)

flores200_indian_languages = {
    # Confirmed in FLORES-200
    "Assamese": "asm_Beng",      # ✅
    "Bengali": "ben_Beng",        # ✅
    "English": "eng_Latn",        # ✅
    "Gujarati": "guj_Gujr",       # ✅
    "Hindi": "hin_Deva",          # ✅
    "Kannada": "kan_Knda",        # ✅
    "Kashmiri": "kas_Arab",       # ✅ (Arabic script)
    "Malayalam": "mal_Mlym",      # ✅
    "Marathi": "mar_Deva",        # ✅
    "Nepali": "npi_Deva",         # ✅
    "Odia": "ory_Orya",           # ✅
    "Punjabi": "pan_Guru",        # ✅ (Gurmukhi script)
    "Sanskrit": "san_Deva",       # ✅
    "Sindhi": "snd_Arab",         # ✅ (Arabic script)
    "Tamil": "tam_Taml",          # ✅
    "Telugu": "tel_Telu",         # ✅
    "Urdu": "urd_Arab",           # ✅
}

# Languages NOT in FLORES-200
flores200_not_included = {
    "Bodo": None,           # ❌ Not in FLORES-200
    "Maithili": None,       # ❌ Not in FLORES-200  
    "Meitei": "mni_Beng",   # ✅ Actually IS in FLORES-200! (Meitei/Manipuri)
    "Santali": None,        # ❌ Not in FLORES-200
}

print("="*80)
print("FLORES-200 OFFICIAL LANGUAGE COVERAGE CHECK")
print("="*80)
print()

in_flores = 0
not_in_flores = 0

for code, lang in sorted(your_languages.items()):
    lang_base = lang.split('/')[0]  # Handle "Meitei/Manipuri"
    
    if lang in flores200_indian_languages:
        flores_code = flores200_indian_languages[lang]
        print(f"✅ {code:5} | {lang:20} | IN FLORES-200 ({flores_code})")
        in_flores += 1
    elif lang_base in flores200_indian_languages:
        flores_code = flores200_indian_languages[lang_base]
        print(f"✅ {code:5} | {lang:20} | IN FLORES-200 ({flores_code})")
        in_flores += 1
    elif "Meitei" in lang or "Manipuri" in lang:
        print(f"✅ {code:5} | {lang:20} | IN FLORES-200 (mni_Beng)")
        in_flores += 1
    else:
        print(f"❌ {code:5} | {lang:20} | NOT IN FLORES-200")
        not_in_flores += 1

print()
print("="*80)
print("SUMMARY")
print("="*80)
print(f"✅ In FLORES-200: {in_flores}/21 ({100*in_flores/21:.0f}%)")
print(f"❌ Not in FLORES-200: {not_in_flores}/21 ({100*not_in_flores/21:.0f}%)")
print()

print("="*80)
print("MISSING LANGUAGES")
print("="*80)
missing = []
for code, lang in sorted(your_languages.items()):
    if lang not in flores200_indian_languages:
        if lang.split('/')[0] not in flores200_indian_languages:
            if "Meitei" not in lang and "Manipuri" not in lang:
                missing.append(f"{lang} ({code})")

if missing:
    for m in missing:
        print(f"  - {m}")
else:
    print("  None! All languages supported!")

print()
print("="*80)
print("FLORES-200 CODES FOR YOUR LANGUAGES")
print("="*80)
print()
print("You can download specific languages using these codes:")
print()
for lang, code in sorted(flores200_indian_languages.items()):
    print(f"  {lang:20} → {code}")
print(f"  {'Meitei/Manipuri':20} → mni_Beng")

print()
print("="*80)
print("FINAL ANSWER")
print("="*80)
print(f"""
FLORES-200 supports {in_flores} out of your 21 languages ({100*in_flores/21:.0f}%).

Missing languages ({not_in_flores}):
  - Bodo (bd)
  - Maithili (mai)
  - Santali (sat)

These 3 languages are extremely low-resource and not included in FLORES-200's
204-language dataset.

For these 3 languages, you can:
  1. Use your data/training/*.txt files
  2. Rely on NLLB-200's cross-lingual transfer
  3. Accept that they won't have high-quality translations
""")

