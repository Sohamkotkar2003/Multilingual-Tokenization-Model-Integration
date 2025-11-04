#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detailed Accuracy Test - Manual Quality Assessment
===================================================

Tests generation quality across all 21 languages with real-world prompts.
YOU will manually assess if outputs are accurate and on-topic.
"""

import requests
import json
import sys
import io
import time

# Force UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

LM_CORE_URL = "http://localhost:8117"

# Comprehensive test cases - 21 languages
ACCURACY_TESTS = [
    # Major Languages - More tests
    {"lang": "hindi", "prompt": "महात्मा गांधी", "expected_topic": "Mahatma Gandhi"},
    {"lang": "hindi", "prompt": "भारतीय शिक्षा प्रणाली", "expected_topic": "Indian education system"},
    {"lang": "english", "prompt": "Climate change", "expected_topic": "Climate change"},
    {"lang": "english", "prompt": "Democracy", "expected_topic": "Democracy"},
    {"lang": "tamil", "prompt": "தமிழ்நாடு", "expected_topic": "Tamil Nadu"},
    {"lang": "tamil", "prompt": "இசை", "expected_topic": "Music"},
    {"lang": "bengali", "prompt": "রবীন্দ্রনাথ ঠাকুর", "expected_topic": "Rabindranath Tagore"},
    {"lang": "bengali", "prompt": "বাংলাদেশ", "expected_topic": "Bangladesh"},
    {"lang": "telugu", "prompt": "తెలుగు భాష", "expected_topic": "Telugu language"},
    {"lang": "marathi", "prompt": "महाराष्ट्र", "expected_topic": "Maharashtra"},
    {"lang": "gujarati", "prompt": "ગુજરાત", "expected_topic": "Gujarat"},
    {"lang": "kannada", "prompt": "ಕರ್ನಾಟಕ", "expected_topic": "Karnataka"},
    {"lang": "malayalam", "prompt": "കേരളം", "expected_topic": "Kerala"},
    {"lang": "punjabi", "prompt": "ਪੰਜਾਬ", "expected_topic": "Punjab"},
    {"lang": "sanskrit", "prompt": "वेदः", "expected_topic": "Vedas"},
    {"lang": "urdu", "prompt": "شاعری", "expected_topic": "Poetry"},
    {"lang": "odia", "prompt": "ଓଡ଼ିଶା", "expected_topic": "Odisha"},
    {"lang": "assamese", "prompt": "অসম", "expected_topic": "Assam"},
    {"lang": "nepali", "prompt": "नेपाल", "expected_topic": "Nepal"},
    {"lang": "maithili", "prompt": "मैथिली भाषा", "expected_topic": "Maithili language"},
    {"lang": "bodo", "prompt": "बड़ो भाषा", "expected_topic": "Bodo language"},
]


def test_generation(prompt, language):
    """Test generation and return output"""
    try:
        start = time.time()
        response = requests.post(
            f"{LM_CORE_URL}/generate",
            json={"text": prompt, "language": language},
            timeout=30
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "output": result.get('generated_text', ''),
                "time": elapsed
            }
        return {"success": False, "error": f"HTTP {response.status_code}", "time": elapsed}
    except Exception as e:
        return {"success": False, "error": str(e), "time": 0}


def analyze_output(output, prompt, language, expected_topic):
    """Analyze output for accuracy issues"""
    issues = []
    
    # Check 1: Is output too short?
    if len(output) < 50:
        issues.append("⚠️ Too short")
    
    # Check 2: Does it contain the original prompt?
    if prompt.lower() in output.lower():
        # Good - it's responding to the prompt
        pass
    
    # Check 3: Language mixing detection (basic)
    # Check for common wrong script markers
    if language in ["hindi", "sanskrit", "marathi", "nepali"]:
        # Should be Devanagari
        if any(script in output for script in ["ஆ", "థ", "ক", "ا"]):  # Tamil, Telugu, Bengali, Arabic
            issues.append("❌ Wrong script detected")
    
    if language == "tamil":
        if any(script in output for script in ["आ", "अ", "क", "ک"]):  # Devanagari, Arabic
            issues.append("❌ Wrong script (not Tamil)")
    
    if language == "bengali":
        if any(script in output for script in ["আ", "अ", "ஆ"]) and len([c for c in output if ord(c) in range(0x0900, 0x097F)]) > 20:
            # Some Devanagari is okay, but not too much
            issues.append("⚠️ Mixed Devanagari in Bengali")
    
    # Check 4: English in non-English output
    if language != "english":
        english_count = sum(1 for c in output if ord(c) < 128 and c.isalpha())
        total_alpha = sum(1 for c in output if c.isalpha())
        if total_alpha > 0 and english_count / total_alpha > 0.6:
            issues.append("❌ Too much English")
    
    # Check 5: Chinese/Other random scripts
    if any(char in output for char in "中国的书法画展览"):
        issues.append("❌ Chinese characters detected!")
    
    return issues


print("\n" + "="*80)
print("  DETAILED ACCURACY TEST - ALL 21 LANGUAGES")
print("="*80)
print("\nThis will test generation quality on real-world topics.")
print("Each output will be analyzed for accuracy issues.\n")

# Get model info
try:
    config_response = requests.get(f"{LM_CORE_URL}/config", timeout=5)
    model_info = config_response.json().get('model', {})
    print(f"🤖 Model: {model_info.get('model_name', 'Unknown')}")
    print(f"📁 Adapter: {model_info.get('model_path', 'None')}")
    print(f"⚡ Quantization: {'Yes' if model_info.get('use_4bit_quantization') else 'No'}")
except:
    print("❌ Server not running!")
    exit(1)

print("\n" + "="*80 + "\n")

# Run tests
results = []
total_issues = 0
languages_tested = set()

for i, test in enumerate(ACCURACY_TESTS, 1):
    lang = test["lang"]
    prompt = test["prompt"]
    expected = test["expected_topic"]
    
    print(f"[{i:2d}/{len(ACCURACY_TESTS)}] {lang.capitalize():12s} | {expected:25s} ", end="", flush=True)
    
    result = test_generation(prompt, lang)
    
    if result["success"]:
        output = result["output"]
        issues = analyze_output(output, prompt, lang, expected)
        
        if len(issues) == 0:
            print(f"✅ ({result['time']:.1f}s, {len(output):3d} chars)")
        else:
            print(f"⚠️ ({result['time']:.1f}s) - {', '.join(issues)}")
            total_issues += len(issues)
        
        results.append({
            "language": lang,
            "prompt": prompt,
            "expected_topic": expected,
            "success": True,
            "output": output,
            "time": result["time"],
            "issues": issues,
            "issue_count": len(issues)
        })
        
        languages_tested.add(lang)
    else:
        print(f"❌ FAILED - {result.get('error', 'Unknown')}")
        results.append({
            "language": lang,
            "success": False,
            "error": result.get('error')
        })
    
    time.sleep(0.3)

# Summary
print("\n" + "="*80)
print("  ACCURACY SUMMARY")
print("="*80 + "\n")

successful = [r for r in results if r.get("success")]
with_issues = [r for r in results if r.get("success") and r.get("issue_count", 0) > 0]
perfect = [r for r in results if r.get("success") and r.get("issue_count", 0) == 0]

print(f"📊 Tests Run: {len(results)}")
print(f"✅ Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
print(f"🎯 Perfect (no issues): {len(perfect)} ({len(perfect)/len(results)*100:.1f}%)")
print(f"⚠️ With Issues: {len(with_issues)} ({len(with_issues)/len(results)*100:.1f}%)")
print(f"🌍 Languages Tested: {len(languages_tested)}/21")
print(f"\n⏱️  Average Time: {sum(r['time'] for r in successful)/len(successful):.2f}s")

# Accuracy by issue type
if with_issues:
    print(f"\n📋 Issue Breakdown:")
    issue_types = {}
    for r in with_issues:
        for issue in r.get("issues", []):
            issue_types[issue] = issue_types.get(issue, 0) + 1
    
    for issue, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
        print(f"   {issue}: {count} occurrences")

# Accuracy score
accuracy_score = (len(perfect) / len(results)) * 100
print(f"\n🎯 OVERALL ACCURACY SCORE: {accuracy_score:.1f}%")

if accuracy_score >= 80:
    print("   🏆 EXCELLENT - Production ready!")
elif accuracy_score >= 60:
    print("   ✅ GOOD - Acceptable for most use cases")
elif accuracy_score >= 40:
    print("   ⚠️ FAIR - Needs improvement")
else:
    print("   ❌ POOR - Significant issues")

# Save detailed results
with open("accuracy_test_detailed.json", 'w', encoding='utf-8') as f:
    json.dump({
        "model_info": model_info,
        "summary": {
            "total_tests": len(results),
            "successful": len(successful),
            "perfect": len(perfect),
            "with_issues": len(with_issues),
            "accuracy_score": accuracy_score
        },
        "results": results
    }, f, indent=2, ensure_ascii=False)

print(f"\n📄 Detailed results saved to: accuracy_test_detailed.json")
print("\n" + "="*80 + "\n")

