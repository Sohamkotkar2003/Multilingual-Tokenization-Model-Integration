#!/usr/bin/env python3
"""
Comprehensive System Test Suite for BHIV Sovereign AI Platform
Tests ALL endpoints, components, and features with detailed logging
Generates a complete test report with timestamps and results
"""

import requests
import json
import time
import sys
from datetime import datetime
from pathlib import Path

# Color codes for terminal output (works on most terminals)
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Test configuration
LM_CORE_URL = "http://localhost:8117"
SOVEREIGN_URL = "http://localhost:8116"
RESULTS_DIR = Path("test_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Generate timestamped log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = RESULTS_DIR / f"comprehensive_test_{timestamp}.log"
json_results_file = RESULTS_DIR / f"test_results_{timestamp}.json"

# Global test results
test_results = {
    "test_run_info": {
        "timestamp": datetime.now().isoformat(),
        "lm_core_url": LM_CORE_URL,
        "sovereign_url": SOVEREIGN_URL
    },
    "tests": [],
    "summary": {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "warnings": 0
    }
}

def log(message, level="INFO", color=None):
    """Log message to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_message = f"[{timestamp}] [{level}] {message}"
    
    # Console output with color
    if color:
        print(f"{color}{log_message}{Colors.ENDC}")
    else:
        print(log_message)
    
    # File output without color
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_message + '\n')

def print_banner(text):
    """Print a prominent banner"""
    border = "=" * 100
    log(border, color=Colors.HEADER)
    log(f"  {text}", color=Colors.BOLD)
    log(border, color=Colors.HEADER)

def print_section(text):
    """Print a section header"""
    log("\n" + "─" * 100, color=Colors.OKCYAN)
    log(f"  {text}", color=Colors.OKCYAN)
    log("─" * 100, color=Colors.OKCYAN)

def record_test_result(test_name, status, details, response_time=None, error=None):
    """Record test result in global results"""
    result = {
        "test_name": test_name,
        "status": status,  # "PASS", "FAIL", "WARNING"
        "timestamp": datetime.now().isoformat(),
        "response_time": response_time,
        "details": details,
        "error": error
    }
    test_results["tests"].append(result)
    test_results["summary"]["total_tests"] += 1
    
    if status == "PASS":
        test_results["summary"]["passed"] += 1
    elif status == "FAIL":
        test_results["summary"]["failed"] += 1
    elif status == "WARNING":
        test_results["summary"]["warnings"] += 1

def run_test(test_num, test_name, url, method="POST", payload=None, timeout=30, expected_keys=None):
    """
    Run a single test and log results
    
    Args:
        test_num: Test number
        test_name: Descriptive test name
        url: Full URL to test
        method: HTTP method (POST, GET)
        payload: JSON payload for POST requests
        timeout: Request timeout in seconds
        expected_keys: List of keys expected in response
    
    Returns:
        dict: Response JSON or None if failed
    """
    print_section(f"TEST {test_num}: {test_name}")
    log(f"URL: {url}")
    log(f"Method: {method}")
    if payload:
        log(f"Payload: {json.dumps(payload, ensure_ascii=False)}")
    
    start_time = time.time()
    
    try:
        if method == "POST":
            response = requests.post(url, json=payload, timeout=timeout)
        else:
            response = requests.get(url, timeout=timeout)
        
        response_time = time.time() - start_time
        
        log(f"Status Code: {response.status_code}")
        log(f"Response Time: {response_time:.3f}s")
        
        if response.status_code == 200:
            try:
                result = response.json()
                log("✅ SUCCESS", level="PASS", color=Colors.OKGREEN)
                log(f"Response Preview: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}...")
                
                # Check for expected keys
                if expected_keys:
                    missing_keys = [key for key in expected_keys if key not in result]
                    if missing_keys:
                        log(f"⚠️ WARNING: Missing expected keys: {missing_keys}", level="WARNING", color=Colors.WARNING)
                        record_test_result(test_name, "WARNING", result, response_time, f"Missing keys: {missing_keys}")
                    else:
                        log(f"✓ All expected keys present: {expected_keys}", color=Colors.OKGREEN)
                        record_test_result(test_name, "PASS", result, response_time)
                else:
                    record_test_result(test_name, "PASS", result, response_time)
                
                return result
                
            except json.JSONDecodeError as e:
                log(f"⚠️ WARNING: Response is not JSON: {response.text[:200]}", level="WARNING", color=Colors.WARNING)
                record_test_result(test_name, "WARNING", {"raw_response": response.text[:500]}, response_time, "Non-JSON response")
                return None
        else:
            log(f"❌ FAILED: HTTP {response.status_code}", level="FAIL", color=Colors.FAIL)
            log(f"Error Response: {response.text[:500]}")
            record_test_result(test_name, "FAIL", {"status_code": response.status_code, "error": response.text[:500]}, response_time, f"HTTP {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        response_time = time.time() - start_time
        log(f"❌ FAILED: Request timed out after {timeout}s", level="FAIL", color=Colors.FAIL)
        record_test_result(test_name, "FAIL", {}, response_time, "Timeout")
        return None
        
    except requests.exceptions.ConnectionError as e:
        response_time = time.time() - start_time
        log(f"❌ FAILED: Connection error - {str(e)}", level="FAIL", color=Colors.FAIL)
        record_test_result(test_name, "FAIL", {}, response_time, f"Connection error: {str(e)}")
        return None
        
    except Exception as e:
        response_time = time.time() - start_time
        log(f"❌ FAILED: Unexpected error - {str(e)}", level="FAIL", color=Colors.FAIL)
        record_test_result(test_name, "FAIL", {}, response_time, f"Exception: {str(e)}")
        return None

def main():
    """Main test execution"""
    print_banner("COMPREHENSIVE BHIV SOVEREIGN AI PLATFORM TEST SUITE")
    log(f"Test started at: {datetime.now().isoformat()}")
    log(f"Results will be saved to: {log_file}")
    log(f"JSON results will be saved to: {json_results_file}")
    
    # ========================================================================
    # SECTION 1: HEALTH CHECKS
    # ========================================================================
    print_banner("SECTION 1: HEALTH CHECKS")
    
    run_test(
        1, 
        "LM Core Health Check",
        f"{LM_CORE_URL}/health",
        method="GET",
        expected_keys=["status", "components", "supported_languages"]
    )
    
    time.sleep(1)
    
    run_test(
        2,
        "Sovereign Core Health Check",
        f"{SOVEREIGN_URL}/health",
        method="GET",
        expected_keys=["status", "components", "timestamp"]
    )
    
    time.sleep(1)
    
    # ========================================================================
    # SECTION 2: LANGUAGE DETECTION & PROCESSING
    # ========================================================================
    print_banner("SECTION 2: LANGUAGE DETECTION & PROCESSING")
    
    language_tests = [
        ("Hindi", "नमस्ते, आप कैसे हैं?", "hindi"),
        ("Tamil", "வணக்கம், எப்படி இருக்கிறீர்கள்?", "tamil"),
        ("Bengali", "নমস্কার, আপনি কেমন আছেন?", "bengali"),
        ("English", "Hello, how are you?", "english"),
        ("Sanskrit", "नमस्कारः, कथम् अस्ति?", "sanskrit"),
        ("Telugu", "నమస్కారం, మీరు ఎలా ఉన్నారు?", "telugu"),
        ("Marathi", "नमस्कार, तुम्ही कसे आहात?", "marathi"),
    ]
    
    test_num = 3
    for lang_name, text, expected_lang in language_tests:
        result = run_test(
            test_num,
            f"Language Detection - {lang_name}",
            f"{LM_CORE_URL}/language-detect",
            payload={"text": text},
            expected_keys=["language", "confidence"]
        )
        
        if result and result.get("language") == expected_lang:
            log(f"✓ Correctly detected as {expected_lang}", color=Colors.OKGREEN)
        elif result:
            log(f"⚠️ Expected {expected_lang}, got {result.get('language')}", level="WARNING", color=Colors.WARNING)
        
        test_num += 1
        time.sleep(0.5)
    
    # ========================================================================
    # SECTION 3: TEXT GENERATION
    # ========================================================================
    print_banner("SECTION 3: TEXT GENERATION (BLOOMZ Model)")
    
    generation_tests = [
        ("Hindi Generation", "नमस्ते, मुझे भारत के बारे में बताइए।", "hindi"),
        ("English Generation", "Tell me about artificial intelligence.", "english"),
        ("Tamil Generation", "செயற்கை நுண்ணறிவு பற்றி சொல்லுங்கள்.", "tamil"),
    ]
    
    for test_name, prompt, lang in generation_tests:
        run_test(
            test_num,
            test_name,
            f"{LM_CORE_URL}/generate",
            payload={"text": prompt, "language": lang},
            expected_keys=["language", "generated_text", "input_text"],
            timeout=60
        )
        test_num += 1
        time.sleep(2)
    
    # ========================================================================
    # SECTION 4: KNOWLEDGE BASE & RAG
    # ========================================================================
    print_banner("SECTION 4: KNOWLEDGE BASE & RAG QUERIES")
    
    qa_tests = [
        ("English Q&A", "What is machine learning?", "english"),
        ("Hindi Q&A", "भारत की राजधानी क्या है?", "hindi"),
        ("Educational Q&A", "Explain neural networks in simple terms.", "english"),
        ("Factual Q&A", "What is the capital of France?", "english"),
    ]
    
    for test_name, query, lang in qa_tests:
        run_test(
            test_num,
            test_name,
            f"{LM_CORE_URL}/qa",
            payload={"query": query, "language": lang, "top_k": 5},
            expected_keys=["answer", "query", "language", "confidence", "sources"],
            timeout=60
        )
        test_num += 1
        time.sleep(2)
    
    # ========================================================================
    # SECTION 5: MULTILINGUAL CONVERSATION
    # ========================================================================
    print_banner("SECTION 5: MULTILINGUAL CONVERSATION")
    
    conversation_tests = [
        ("English Conversation", "How does photosynthesis work?", "english"),
        ("Hindi Conversation", "मशीन लर्निंग क्या है?", "hindi"),
        ("Tamil Conversation", "இயற்கை மொழி செயலாக்கம் என்றால் என்ன?", "tamil"),
    ]
    
    for test_name, message, lang in conversation_tests:
        run_test(
            test_num,
            test_name,
            f"{LM_CORE_URL}/multilingual-conversation",
            payload={
                "text": message,
                "language": lang,
                "generate_response": True,
                "max_response_length": 150
            },
            expected_keys=["user_query", "kb_answer", "language", "session_id"],
            timeout=60
        )
        test_num += 1
        time.sleep(2)
    
    # ========================================================================
    # SECTION 6: KSML SEMANTIC ALIGNMENT
    # ========================================================================
    print_banner("SECTION 6: KSML SEMANTIC ALIGNMENT")
    
    ksml_tests = [
        ("Instruction Intent", "Create a new mobile application for learning languages"),
        ("Question Intent", "What is the meaning of life?"),
        ("Greeting Intent", "Hello, how are you today?"),
        ("Explanation Intent", "Explain quantum computing to me"),
        ("Command Intent", "Please show me the results"),
    ]
    
    for test_name, text in ksml_tests:
        run_test(
            test_num,
            f"KSML - {test_name}",
            f"{SOVEREIGN_URL}/align.ksml",
            payload={"text": text, "target_lang": "en"},
            expected_keys=["intent", "source_lang", "target_lang", "karma_state", "semantic_roots", "confidence"]
        )
        test_num += 1
        time.sleep(0.5)
    
    # ========================================================================
    # SECTION 7: RL FEEDBACK & POLICY
    # ========================================================================
    print_banner("SECTION 7: RL FEEDBACK & POLICY LEARNING")
    
    rl_feedback_tests = [
        ("High Reward Feedback", "Translate to English: नमस्ते", "Hello", 0.95),
        ("Medium Reward Feedback", "Create a todo app", "Here's a basic todo app structure", 0.7),
        ("Low Reward Feedback", "Random text", "Unclear response", 0.3),
        ("Perfect Reward", "What is AI?", "Artificial Intelligence is...", 1.0),
    ]
    
    for test_name, prompt, output, reward in rl_feedback_tests:
        run_test(
            test_num,
            f"RL - {test_name} (reward={reward})",
            f"{SOVEREIGN_URL}/rl.feedback",
            payload={
                "prompt": prompt,
                "output": output,
                "reward": reward,
                "user_id": "test_user",
                "session_id": f"session_{test_num}"
            },
            expected_keys=["status", "policy_updated", "reward_logged"]
        )
        test_num += 1
        time.sleep(0.5)
    
    # ========================================================================
    # SECTION 8: VAANI SPEECH COMPOSITION
    # ========================================================================
    print_banner("SECTION 8: VAANI SPEECH-READY COMPOSITION")
    
    speech_tests = [
        ("Calm Tone", "Welcome to our meditation session", "en", "calm"),
        ("Excited Tone", "Congratulations on your achievement!", "en", "excited"),
        ("Friendly Tone", "Hello! Nice to meet you.", "en", "friendly"),
        ("Serious Tone", "This is an important announcement.", "en", "serious"),
        ("Hindi Speech", "नमस्ते, आपका स्वागत है।", "hi", "friendly"),
    ]
    
    for test_name, text, lang, tone in speech_tests:
        run_test(
            test_num,
            f"Speech - {test_name}",
            f"{SOVEREIGN_URL}/compose.speech_ready",
            payload={"text": text, "language": lang, "tone": tone},
            expected_keys=["text", "tone", "lang", "prosody_hint", "audio_metadata"]
        )
        test_num += 1
        time.sleep(0.5)
    
    # ========================================================================
    # SECTION 9: COMPLETE BRIDGE PIPELINE
    # ========================================================================
    print_banner("SECTION 9: COMPLETE MULTILINGUAL REASONING BRIDGE")
    
    bridge_tests = [
        ("Simple Query", "What is Python programming?", False),
        ("Educational Query", "Teach me about machine learning", False),
        ("Translation Query", "Translate: Hello world", False),
        ("Complex Query", "Explain the difference between AI and ML in simple terms", False),
    ]
    
    for test_name, text, include_audio in bridge_tests:
        run_test(
            test_num,
            f"Bridge - {test_name}",
            f"{SOVEREIGN_URL}/bridge.reason",
            payload={
                "text": text,
                "include_audio": include_audio,
                "user_id": "test_user",
                "session_id": f"bridge_session_{test_num}"
            },
            expected_keys=["aligned_text", "ksml_metadata", "processing_time", "trace_id"],
            timeout=60
        )
        test_num += 1
        time.sleep(2)
    
    # ========================================================================
    # SECTION 10: SYSTEM STATISTICS & MONITORING
    # ========================================================================
    print_banner("SECTION 10: SYSTEM STATISTICS & MONITORING")
    
    run_test(
        test_num,
        "LM Core Statistics",
        f"{LM_CORE_URL}/stats",
        method="GET",
        expected_keys=["status", "kb_integration", "endpoints"]
    )
    test_num += 1
    time.sleep(1)
    
    run_test(
        test_num,
        "Sovereign Core Statistics",
        f"{SOVEREIGN_URL}/stats",
        method="GET",
        expected_keys=["timestamp", "components"]
    )
    test_num += 1
    time.sleep(1)
    
    run_test(
        test_num,
        "LM Core Configuration",
        f"{LM_CORE_URL}/config",
        method="GET",
        expected_keys=["api", "model", "languages"]
    )
    test_num += 1
    
    # ========================================================================
    # SECTION 11: STRESS & PERFORMANCE TESTS
    # ========================================================================
    print_banner("SECTION 11: RAPID FIRE TESTS (Performance)")
    
    log("Testing rapid successive requests...")
    rapid_fire_start = time.time()
    
    for i in range(5):
        run_test(
            test_num,
            f"Rapid Fire KSML #{i+1}",
            f"{SOVEREIGN_URL}/align.ksml",
            payload={"text": f"Test query number {i+1}", "target_lang": "en"},
            expected_keys=["intent", "karma_state"]
        )
        test_num += 1
        # No sleep - testing rapid succession
    
    rapid_fire_duration = time.time() - rapid_fire_start
    log(f"Rapid fire completed in {rapid_fire_duration:.2f}s", color=Colors.OKGREEN)
    
    # ========================================================================
    # FINAL REPORT GENERATION
    # ========================================================================
    print_banner("TEST SUITE COMPLETE - GENERATING REPORTS")
    
    # Calculate statistics
    total = test_results["summary"]["total_tests"]
    passed = test_results["summary"]["passed"]
    failed = test_results["summary"]["failed"]
    warnings = test_results["summary"]["warnings"]
    success_rate = (passed / total * 100) if total > 0 else 0
    
    # Add summary to results
    test_results["summary"]["success_rate"] = success_rate
    test_results["summary"]["duration_seconds"] = time.time() - start_time
    test_results["summary"]["end_time"] = datetime.now().isoformat()
    
    # Save JSON results
    with open(json_results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    log(f"\n✅ JSON results saved to: {json_results_file}", color=Colors.OKGREEN)
    
    # Print summary
    print_banner("TEST SUMMARY")
    log(f"Total Tests Run: {total}")
    log(f"✅ Passed: {passed}", color=Colors.OKGREEN)
    log(f"❌ Failed: {failed}", color=Colors.FAIL if failed > 0 else Colors.OKGREEN)
    log(f"⚠️ Warnings: {warnings}", color=Colors.WARNING if warnings > 0 else Colors.OKGREEN)
    log(f"Success Rate: {success_rate:.1f}%", color=Colors.OKGREEN if success_rate >= 90 else Colors.WARNING)
    log(f"Total Duration: {test_results['summary']['duration_seconds']:.2f}s")
    
    log(f"\n📄 Complete log saved to: {log_file}", color=Colors.OKBLUE)
    log(f"📊 JSON results saved to: {json_results_file}", color=Colors.OKBLUE)
    
    # Exit with appropriate code
    if failed > 0:
        log("\n❌ Some tests failed. Please review the logs above.", level="ERROR", color=Colors.FAIL)
        return 1
    elif warnings > 0:
        log("\n⚠️ All tests passed but with warnings. Review recommended.", level="WARNING", color=Colors.WARNING)
        return 0
    else:
        log("\n🎉 ALL TESTS PASSED SUCCESSFULLY!", color=Colors.OKGREEN)
        return 0

if __name__ == "__main__":
    start_time = time.time()
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        log("\n\n⚠️ Test suite interrupted by user", level="WARNING", color=Colors.WARNING)
        log(f"Partial results saved to: {log_file}")
        sys.exit(130)
    except Exception as e:
        log(f"\n\n❌ Fatal error: {str(e)}", level="ERROR", color=Colors.FAIL)
        import traceback
        log(traceback.format_exc())
        sys.exit(1)

