#!/usr/bin/env python3
"""Test all 38 languages (21 original + 8 new + 9 bootstrapped)"""

import sys
import io
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Fix Windows encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

# Complete test configuration for all 38 languages
TEST_CONFIG = {
    # ========== 21 ORIGINAL LANGUAGES (Gurukul Lite) ==========
    "hindi": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["भारत की राजधानी", "नमस्ते, मेरा नाम", "आज का मौसम"]
    },
    "bengali": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["ভারতের রাজধানী", "সুপ্রভাত, আমার নাম", "আজকের আবহাওয়া"]
    },
    "gujarati": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["ભારતની રાજધાની", "નમસ્તે, મારું નામ", "આજનું હવામાન"]
    },
    "marathi": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["भारताची राजधानी", "नमस्कार, माझे नाव", "आजचे हवामान"]
    },
    "tamil": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["இந்தியாவின் தலைநகரம்", "வணக்கம், என் பெயர்", "இன்றைய வானிலை"]
    },
    "telugu": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["భారతదేశ రాజధాని", "నమస్కారం, నా పేరు", "నేటి వాతావరణం"]
    },
    "kannada": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["ಭಾರತದ ರಾಜಧಾನಿ", "ನಮಸ್ಕಾರ, ನನ್ನ ಹೆಸರು", "ಇಂದಿನ ಹವಾಮಾನ"]
    },
    "malayalam": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["ഇന്ത്യയുടെ തലസ്ഥാനം", "നമസ്കാരം, എന്റെ പേര്", "ഇന്നത്തെ കാലാവസ്ഥ"]
    },
    "punjabi": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["ਭਾਰਤ ਦੀ ਰਾਜਧਾਨੀ", "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਮੇਰਾ ਨਾਮ", "ਅੱਜ ਦਾ ਮੌਸਮ"]
    },
    "odia": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["ଭାରତର ରାଜଧାନୀ", "ନମସ୍କାର, ମୋ ନାମ", "ଆଜିର ପାଗ"]
    },
    "assamese": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["ভাৰতৰ ৰাজধানী", "নমস্কাৰ, মোৰ নাম", "আজিৰ বতৰ"]
    },
    "urdu": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["بھارت کا دارالحکومت", "السلام علیکم، میرا نام", "آج کا موسم"]
    },
    "nepali": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["भारतको राजधानी", "नमस्ते, मेरो नाम", "आजको मौसम"]
    },
    "sanskrit": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["भारतस्य राजधानी", "नमस्ते, मम नाम", "अद्य मौसमः"]
    },
    "sindhi": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["ڀارت جي راڄڌاني", "سلام، منهنجو نالو", "اڄ جي موسم"]
    },
    "kashmiri": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["ہندوستان کی راجدھانی", "سلام، میہ ناو", "آز موسم"]
    },
    "manipuri": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["ভাৰতৰ ৰাজধানী", "খুরুমজরি, ঐগী মিং", "ঙসিগী লমদম"]
    },
    "bodo": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["भारतनि राजधानी", "जोहार, आङा मुं", "दिनैनि बाथै"]
    },
    "santali": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["ᱵᱷᱟᱨᱚᱛ ᱨᱮᱱᱟᱜ ᱨᱟᱡᱽᱜᱟᱲ", "ᱡᱚᱦᱟᱨ, ᱤᱧᱟᱜ ᱧᱩᱛᱩᱢ", "ᱛᱮᱦᱮᱧ ᱨᱤᱛᱩ"]
    },
    "maithili": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["भारतक राजधानी", "प्रणाम, हमर नाम", "आइक मौसम"]
    },
    "english": {
        "adapter_path": "adapters/gurukul_lite",
        "prompts": ["The capital of India", "Hello, my name is", "Today's weather"]
    },
    
    # ========== 8 NEW LANGUAGES (Individual Adapters) ==========
    "sinhala": {
        "adapter_path": "adapters/sinhala_lite",
        "prompts": ["ශ්‍රී ලංකාවේ අගනුවර", "සුභ උදෑසනක්", "මගේ නම"]
    },
    "tibetan": {
        "adapter_path": "adapters/tibetan_lite",
        "prompts": ["བོད་ཀྱི་རྒྱལ་ས", "ངའི་མིང་ལ", "བཀྲ་ཤིས་བདེ་ལེགས"]
    },
    "dzongkha": {
        "adapter_path": "adapters/dzongkha_lite",
        "prompts": ["འབྲུག་གི་རྒྱལ་ས", "ང་གི་མིང་", "བཀྲ་ཤིས་བདེ་ལེགས"]
    },
    "pashto": {
        "adapter_path": "adapters/pashto_lite",
        "prompts": ["زما نوم", "افغانستان پایتخت", "سلام علیکم"]
    },
    "dari": {
        "adapter_path": "adapters/dari_lite",
        "prompts": ["نام من", "پایتخت افغانستان", "سلام علیکم"]
    },
    "vietnamese": {
        "adapter_path": "adapters/vietnamese_lite",
        "prompts": ["Thủ đô của Việt Nam", "Tôi tên là", "Xin chào"]
    },
    "thai": {
        "adapter_path": "adapters/thai_lite",
        "prompts": ["เมืองหลวงของประเทศไทย", "ชื่อของฉันคือ", "สวัสดี"]
    },
    "burmese": {
        "adapter_path": "adapters/burmese_lite",
        "prompts": ["မြန်မာနိုင်ငံ၏ မြို့တော်", "ကျွန်တော့်နာမည်", "မင်္ဂလာပါ"]
    },
    
    # ========== 9 BOOTSTRAPPED LANGUAGES (Aliases) ==========
    "awadhi": {
        "adapter_path": "adapters/awadhi_lite",
        "prompts": ["भारत के राजधानी", "नमस्कार, हमार नाम", "आज के मौसम"]
    },
    "bhojpuri": {
        "adapter_path": "adapters/bhojpuri_lite",
        "prompts": ["भारत के राजधानी", "प्रणाम, हमार नाम", "आज के मौसम"]
    },
    "magahi": {
        "adapter_path": "adapters/magahi_lite",
        "prompts": ["भारत के राजधानी", "नमस्कार, हमर नाम", "आज के मौसम"]
    },
    "chhattisgarhi": {
        "adapter_path": "adapters/chhattisgarhi_lite",
        "prompts": ["भारत के राजधानी", "नमस्कार, मोर नाम", "आज के मौसम"]
    },
    "haryanvi": {
        "adapter_path": "adapters/haryanvi_lite",
        "prompts": ["भारत की राजधानी", "नमस्ते, म्हारा नाम", "आज का मौसम"]
    },
    "himachali": {
        "adapter_path": "adapters/himachali_lite",
        "prompts": ["भारत दी राजधानी", "नमस्ते, मेरा नाम", "अज्ज दा मौसम"]
    },
    "pahadi": {
        "adapter_path": "adapters/pahadi_lite",
        "prompts": ["भारतको राजधानी", "नमस्ते, मेरो नाम", "आजको मौसम"]
    },
    "mizo": {
        "adapter_path": "adapters/mizo_lite",
        "prompts": ["India ram capital", "Hello, ka hming chu", "Tun ni weather"]
    },
    "tamil_srilanka": {
        "adapter_path": "adapters/tamil_srilanka_lite",
        "prompts": ["இலங்கையின் தலைநகரம்", "வணக்கம், என் பெயர்", "இன்றைய வானிலை"]
    }
}

def evaluate_quality(text, prompt):
    """Simple quality evaluation"""
    # Check if output is longer than prompt
    if len(text) <= len(prompt) + 10:
        return "Poor"
    
    # Check for excessive gibberish (non-language characters)
    gibberish_chars = sum(1 for c in text if ord(c) > 65535 or c in '□�???')
    gibberish_ratio = gibberish_chars / max(len(text), 1)
    
    if gibberish_ratio > 0.3:
        return "Poor"
    elif gibberish_ratio > 0.1:
        return "Fair"
    
    # Check length and coherence
    if len(text) > len(prompt) + 50:
        return "Good"
    else:
        return "Fair"

def test_adapter(language, config, base_model, tokenizer):
    """Test a single adapter"""
    adapter_path = Path(config["adapter_path"])
    
    print(f"\n{'='*80}")
    print(f"🧪 Testing: {language.upper()}")
    print(f"{'='*80}")
    
    # Check if adapter exists
    if not adapter_path.exists():
        print(f"❌ Adapter not found at: {adapter_path}")
        return {"status": "NOT_FOUND", "quality": "N/A", "samples": []}
    
    print(f"📁 Loading adapter from: {adapter_path}")
    
    try:
        # Load adapter
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model.eval()
        
        print(f"✅ Adapter loaded!\n")
        
        results = []
        qualities = []
        
        # Test each prompt
        for i, prompt in enumerate(config["prompts"][:2], 1):  # Test only 2 prompts for speed
            print(f"📝 Test {i}: {prompt[:50]}...")
            
            try:
                # Generate
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=80,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        num_return_sequences=1
                    )
                
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                quality = evaluate_quality(generated_text, prompt)
                qualities.append(quality)
                
                print(f"   Quality: {quality}")
                print(f"   Output: {generated_text[:100]}...\n")
                
                results.append({
                    "prompt": prompt,
                    "output": generated_text,
                    "quality": quality
                })
            except Exception as e:
                print(f"   ❌ Generation failed: {e}\n")
                qualities.append("Error")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        # Overall quality
        if not qualities:
            overall = "Error"
        elif qualities.count("Good") >= len(qualities) / 2:
            overall = "Good"
        elif qualities.count("Poor") >= len(qualities) / 2:
            overall = "Poor"
        else:
            overall = "Fair"
        
        return {
            "status": "PASS",
            "quality": overall,
            "samples": results
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"status": "FAIL", "quality": "Error", "samples": []}

def main():
    print("="*80)
    print("🧪 TESTING ALL 38 LANGUAGES")
    print("="*80)
    print("\n21 Original + 8 New + 9 Bootstrapped\n")
    
    BASE_MODEL = "bigscience/bloomz-560m"
    
    # Load base model once
    print(f"📥 Loading base model: {BASE_MODEL}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}\n")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Base model loaded\n")
    
    # Test all languages
    results = {}
    for language, config in TEST_CONFIG.items():
        result = test_adapter(language, config, base_model, tokenizer)
        results[language] = result
    
    # Generate summary table
    print("\n" + "="*80)
    print("📊 QUALITY SUMMARY TABLE")
    print("="*80)
    print()
    print(f"{'Language':<20} {'Status':<12} {'Quality':<10} {'Category'}")
    print("-" * 80)
    
    # Sort by category
    original = []
    new_langs = []
    bootstrapped = []
    
    lang_list = list(TEST_CONFIG.keys())
    
    for i, lang in enumerate(lang_list):
        result = results[lang]
        status = result["status"]
        quality = result["quality"]
        
        if i < 21:
            category = "Original"
            original.append((lang, status, quality))
        elif i < 29:
            category = "New"
            new_langs.append((lang, status, quality))
        else:
            category = "Bootstrapped"
            bootstrapped.append((lang, status, quality))
        
        # Color coding
        if quality == "Good":
            quality_display = "✅ Good"
        elif quality == "Fair":
            quality_display = "⚠️  Fair"
        elif quality == "Poor":
            quality_display = "❌ Poor"
        else:
            quality_display = "❓ Error"
        
        print(f"{lang:<20} {status:<12} {quality_display:<15} {category}")
    
    # Statistics
    print("\n" + "="*80)
    print("📈 STATISTICS")
    print("="*80)
    
    def count_quality(lang_list):
        good = sum(1 for _, _, q in lang_list if q == "Good")
        fair = sum(1 for _, _, q in lang_list if q == "Fair")
        poor = sum(1 for _, _, q in lang_list if q == "Poor")
        error = sum(1 for _, _, q in lang_list if q not in ["Good", "Fair", "Poor"])
        return good, fair, poor, error
    
    print("\n📌 Original Languages (21):")
    g, f, p, e = count_quality(original)
    print(f"   ✅ Good: {g}  |  ⚠️  Fair: {f}  |  ❌ Poor: {p}  |  ❓ Error: {e}")
    
    print("\n📌 New Languages (8):")
    g, f, p, e = count_quality(new_langs)
    print(f"   ✅ Good: {g}  |  ⚠️  Fair: {f}  |  ❌ Poor: {p}  |  ❓ Error: {e}")
    
    print("\n📌 Bootstrapped Languages (9):")
    g, f, p, e = count_quality(bootstrapped)
    print(f"   ✅ Good: {g}  |  ⚠️  Fair: {f}  |  ❌ Poor: {p}  |  ❓ Error: {e}")
    
    # Overall
    all_langs = original + new_langs + bootstrapped
    g, f, p, e = count_quality(all_langs)
    print(f"\n📌 TOTAL (38 Languages):")
    print(f"   ✅ Good: {g}  |  ⚠️  Fair: {f}  |  ❌ Poor: {p}  |  ❓ Error: {e}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

