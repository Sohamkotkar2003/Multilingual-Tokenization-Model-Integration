#!/usr/bin/env python3
"""Test individual language adapters"""

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

# Test configurations for each language
TEST_CONFIG = {
    "sinhala": {
        "adapter_path": "adapters/sinhala_lite",
        "prompts": [
            "ශ්‍රී ලංකාවේ අගනුවර",
            "සුභ උදෑසනක්",
            "මගේ නම"
        ]
    },
    "tibetan": {
        "adapter_path": "adapters/tibetan_lite",
        "prompts": [
            "བོད་ཀྱི་རྒྱལ་ས",
            "ངའི་མིང་ལ",
            "བཀྲ་ཤིས་བདེ་ལེགས"
        ]
    },
    "dzongkha": {
        "adapter_path": "adapters/dzongkha_lite",
        "prompts": [
            "འབྲུག་གི་རྒྱལ་ས",
            "ང་གི་མིང་",
            "བཀྲ་ཤིས་བདེ་ལེགས"
        ]
    },
    "pashto": {
        "adapter_path": "adapters/pashto_lite",
        "prompts": [
            "زما نوم",
            "افغانستان پایتخت",
            "سلام علیکم"
        ]
    },
    "dari": {
        "adapter_path": "adapters/dari_lite",
        "prompts": [
            "نام من",
            "پایتخت افغانستان",
            "سلام علیکم"
        ]
    },
    "vietnamese": {
        "adapter_path": "adapters/vietnamese_lite",
        "prompts": [
            "Thủ đô của Việt Nam",
            "Tôi tên là",
            "Xin chào"
        ]
    },
    "thai": {
        "adapter_path": "adapters/thai_lite",
        "prompts": [
            "เมืองหลวงของประเทศไทย",
            "ชื่อของฉันคือ",
            "สวัสดี"
        ]
    },
    "burmese": {
        "adapter_path": "adapters/burmese_lite",
        "prompts": [
            "မြန်မာနိုင်ငံ၏ မြို့တော်",
            "ကျွန်တော့်နာမည်",
            "မင်္ဂလာပါ"
        ]
    }
}

def test_adapter(language, config, base_model, tokenizer):
    """Test a single adapter"""
    adapter_path = Path(config["adapter_path"])
    
    print(f"\n{'='*80}")
    print(f"🧪 Testing: {language.upper()}")
    print(f"{'='*80}")
    
    # Check if adapter exists
    if not adapter_path.exists():
        print(f"❌ Adapter not found at: {adapter_path}")
        return False
    
    print(f"📁 Loading adapter from: {adapter_path}")
    
    try:
        # Load adapter
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model.eval()
        
        print(f"✅ Adapter loaded successfully!\n")
        
        # Test each prompt
        for i, prompt in enumerate(config["prompts"], 1):
            print(f"📝 Test {i}/3: {prompt}")
            
            # Generate
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=100,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    num_return_sequences=1
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"🤖 Generated: {generated_text}\n")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading adapter: {e}")
        return False

def main():
    print("="*80)
    print("🧪 TESTING INDIVIDUAL LANGUAGE ADAPTERS")
    print("="*80)
    
    BASE_MODEL = "bigscience/bloomz-560m"
    
    # Load base model once
    print(f"\n📥 Loading base model: {BASE_MODEL}")
    print("   (This may take a minute...)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Base model loaded\n")
    
    # Test each adapter
    results = {}
    for language, config in TEST_CONFIG.items():
        success = test_adapter(language, config, base_model, tokenizer)
        results[language] = success
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    for language, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {language:15} - {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\n📈 Results: {passed}/{total} adapters working")
    
    if passed == total:
        print("\n🎉 All adapters are working perfectly!")
    elif passed > 0:
        print(f"\n⚠️  {total - passed} adapter(s) need attention")
    else:
        print("\n❌ No adapters found or all failed")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

