#!/usr/bin/env python3
"""
Test HuggingFace Dataset Streaming for Languages Without Wikipedia

Purpose:
- Verify HuggingFace datasets are accessible
- Test streaming capability
- Check data quality

Usage:
    python test_huggingface_streaming.py
"""

import sys
import io

# Force UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Test HuggingFace datasets
DATASETS_TO_TEST = {
    "Awadhi": {
        "uri": "oscar-corpus/OSCAR-2301",
        "config": "unshuffled_deduplicated_awa",
        "samples": 5
    },
    "Bhojpuri": {
        "uri": "ai4bharat/IndicCorp",
        "config": "bhojpuri", 
        "samples": 5
    },
    "Magahi": {
        "uri": "ai4bharat/IndicCorp",
        "config": "magahi",
        "samples": 5
    },
    "Chhattisgarhi": {
        "uri": "oscar-corpus/OSCAR-2301",
        "config": "unshuffled_deduplicated_hne",
        "samples": 5
    },
    "Mizo": {
        "uri": "oscar-corpus/OSCAR-2301",
        "config": "unshuffled_deduplicated_lus",
        "samples": 5
    },
}

def test_huggingface_dataset(language: str, config: dict) -> dict:
    """Test streaming from HuggingFace dataset"""
    
    print(f"\n{'='*80}")
    print(f"Testing HuggingFace: {language}")
    print(f"Dataset: {config['uri']} (config: {config['config']})")
    print(f"{'='*80}")
    
    result = {
        'language': language,
        'dataset': config['uri'],
        'config': config['config'],
        'success': False,
        'samples': 0,
        'avg_length': 0,
        'sample_texts': [],
        'error': None
    }
    
    try:
        from datasets import load_dataset
        
        print(f"Loading dataset in streaming mode...")
        
        # Load dataset in streaming mode
        dataset = load_dataset(
            config['uri'],
            config['config'],
            split='train',
            streaming=True,
            trust_remote_code=True
        )
        
        print(f"✅ Dataset loaded! Streaming samples...\n")
        
        # Stream samples
        samples = []
        for i, sample in enumerate(dataset):
            if i >= config['samples']:
                break
            
            # Extract text (different field names in different datasets)
            text = None
            for field in ['text', 'content', 'sentence', 'passage']:
                if field in sample:
                    text = sample[field]
                    break
            
            if not text:
                continue
            
            text = text.strip()
            
            # Quality check
            if len(text) < 50:
                continue
            
            samples.append(text)
            result['samples'] += 1
            
            # Show preview
            preview = text[:100].replace('\n', ' ')
            print(f"  Sample {result['samples']}: {preview}... ({len(text)} chars)")
        
        # Calculate stats
        if samples:
            result['success'] = True
            result['avg_length'] = int(sum(len(s) for s in samples) / len(samples))
            result['sample_texts'] = samples[:2]  # Keep first 2 for report
        
        print(f"\n✅ {language} test complete:")
        print(f"   Samples: {result['samples']}")
        print(f"   Avg length: {result['avg_length']} chars")
        
        return result
        
    except Exception as e:
        print(f"\n❌ {language} test FAILED: {e}")
        result['error'] = str(e)
        return result


def main():
    """Run HuggingFace streaming tests"""
    
    print("="*80)
    print("🧪 HUGGINGFACE DATASET STREAMING TEST")
    print("Testing languages without dedicated Wikipedia")
    print("="*80)
    
    results = []
    
    for language, config in DATASETS_TO_TEST.items():
        result = test_huggingface_dataset(language, config)
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY")
    print(f"{'='*80}\n")
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"Total tested: {len(results)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}\n")
    
    if successful > 0:
        print("✅ Languages with HuggingFace data:")
        for r in results:
            if r['success']:
                print(f"   - {r['language']}: {r['samples']} samples, {r['avg_length']} chars avg")
    
    if failed > 0:
        print("\n❌ Failed languages:")
        for r in results:
            if not r['success']:
                print(f"   - {r['language']}: {r.get('error', 'Unknown error')}")
    
    print(f"\n{'='*80}")
    print("✅ HuggingFace test complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

