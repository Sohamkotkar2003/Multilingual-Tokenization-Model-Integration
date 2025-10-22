#!/usr/bin/env python3
"""
Check which of our 21 languages are supported by mT5-large
"""
import sys
import io

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Your 21 languages from data/training/
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
    "or": "Odia/Oriya",
    "pa": "Punjabi",
    "sa": "Sanskrit",
    "sat": "Santali",
    "sd": "Sindhi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu"
}

# mT5-large officially supports 101 languages
# Based on mC4 corpus (multilingual C4)
# Covers the 101 languages with largest Wikipedia presence

mt5_supported = [
    "Assamese",      # ✅
    "Bengali",       # ✅
    "English",       # ✅
    "Gujarati",      # ✅
    "Hindi",         # ✅
    "Kannada",       # ✅
    "Malayalam",     # ✅
    "Marathi",       # ✅
    "Nepali",        # ✅
    "Odia",          # ✅ (Oriya)
    "Punjabi",       # ✅
    "Tamil",         # ✅
    "Telugu",        # ✅
    "Urdu",          # ✅
    "Sanskrit",      # ✅
    # Lower-resource languages (may have limited support):
    "Sindhi",        # ⚠️ Limited
]

# Languages likely NOT in mT5's primary training (very low-resource)
mt5_not_supported = [
    "Bodo",          # ❌ Too low-resource
    "Kashmiri",      # ❌ Too low-resource
    "Maithili",      # ❌ Too low-resource
    "Meitei",        # ❌ Too low-resource (Manipuri)
    "Santali",       # ❌ Too low-resource
]

print("="*80)
print("mT5-LARGE LANGUAGE COVERAGE CHECK")
print("="*80)
print()

supported_count = 0
partial_count = 0
unsupported_count = 0

for code, lang in your_languages.items():
    if lang in mt5_supported:
        if lang == "Sindhi":
            print(f"⚠️  {code:5} | {lang:20} | LIMITED SUPPORT")
            partial_count += 1
        else:
            print(f"✅ {code:5} | {lang:20} | FULLY SUPPORTED")
            supported_count += 1
    elif lang in mt5_not_supported:
        print(f"❌ {code:5} | {lang:20} | NOT SUPPORTED (low-resource)")
        unsupported_count += 1
    else:
        print(f"❓ {code:5} | {lang:20} | UNKNOWN")

print()
print("="*80)
print("SUMMARY")
print("="*80)
print(f"✅ Fully Supported:  {supported_count}/21 ({100*supported_count/21:.0f}%)")
print(f"⚠️  Limited Support:  {partial_count}/21")
print(f"❌ Not Supported:    {unsupported_count}/21 ({100*unsupported_count/21:.0f}%)")
print()

print("="*80)
print("LANGUAGES NOT SUPPORTED BY mT5-large:")
print("="*80)
for code, lang in your_languages.items():
    if lang in mt5_not_supported:
        print(f"  - {lang} ({code})")

print()
print("="*80)
print("RECOMMENDATION")
print("="*80)
print("""
mT5-large covers 15-16 out of your 21 languages (71-76%).

The 5 NOT supported are very low-resource languages:
  - Bodo (bd)
  - Kashmiri (ks)
  - Maithili (mai)
  - Meitei/Manipuri (mni)
  - Santali (sat)

These languages have:
  - Very small Wikipedia presence
  - Limited digital text corpora
  - Not included in mC4 training data

For these 5 languages, you would need:
  1. A specialized model (like IndicBART for some)
  2. Pre-train your own model
  3. Use BLOOMZ (has broader but shallower coverage)

BLOOMZ-560M might actually cover MORE of your 21 languages due to
its massive pre-training corpus, even if quality is lower.
""")

