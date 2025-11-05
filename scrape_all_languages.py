#!/usr/bin/env python3
"""
Batch Scraper for All 8 New Languages with Real Data

Purpose:
- Automatically scrape Wikipedia + News for all 8 target languages
- Run overnight to collect 500+ samples per language
- Save data ready for Colab training

Usage:
    python scrape_all_languages.py
    
Time: 2-4 hours total (can run overnight)
"""

import sys
import io
import time
from pathlib import Path
from datetime import datetime

# Force UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from scrapers.scrape_wiki import WikipediaScraper
from scrapers.scrape_news import NewsScraper

# Configuration for all 8 languages
LANGUAGES = {
    "Sinhala": {
        "code": "si",
        "wiki_pages": 500,
        "news_articles": 0,  # News RSS not working
        "news_name": "sinhala"
    },
    "Tibetan": {
        "code": "bo",
        "wiki_pages": 500,
        "news_articles": 0,
        "news_name": None
    },
    "Dzongkha": {
        "code": "dz",
        "wiki_pages": 500,
        "news_articles": 0,
        "news_name": None
    },
    "Pashto": {
        "code": "ps",
        "wiki_pages": 500,
        "news_articles": 200,
        "news_name": "pashto"
    },
    "Dari": {
        "code": "prs",
        "wiki_pages": 0,  # No Wikipedia
        "news_articles": 200,
        "news_name": "dari"
    },
    "Vietnamese": {
        "code": "vi",
        "wiki_pages": 500,
        "news_articles": 200,
        "news_name": "vietnamese"
    },
    "Thai": {
        "code": "th",
        "wiki_pages": 500,
        "news_articles": 200,
        "news_name": "thai"
    },
    "Burmese": {
        "code": "my",
        "wiki_pages": 500,
        "news_articles": 0,  # News RSS not working
        "news_name": "burmese"
    },
}

OUTPUT_DIR = Path("data/training_new")


def scrape_language(language_name: str, config: dict):
    """Scrape data for a single language"""
    
    print(f"\n{'='*80}")
    print(f"🌍 Scraping: {language_name} ({config['code']})")
    print(f"{'='*80}")
    
    start_time = time.time()
    total_samples = 0
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Scrape Wikipedia
    if config['wiki_pages'] > 0:
        print(f"\n🌐 Scraping Wikipedia: {config['wiki_pages']} pages...")
        
        try:
            scraper = WikipediaScraper(
                language_code=config['code'],
                delay=1.0,
                enable_quality_filter=True
            )
            
            output_file = OUTPUT_DIR / f"{language_name.lower()}_wiki.jsonl"
            count = scraper.save_to_jsonl(output_file, config['wiki_pages'])
            total_samples += count
            
            print(f"   ✅ Wikipedia: {count} pages saved")
            
        except Exception as e:
            print(f"   ❌ Wikipedia failed: {e}")
    
    # Scrape News
    if config['news_articles'] > 0 and config['news_name']:
        print(f"\n📰 Scraping News: {config['news_articles']} articles...")
        
        try:
            scraper = NewsScraper(
                language=config['news_name'],
                delay=2.0
            )
            
            output_file = OUTPUT_DIR / f"{language_name.lower()}_news.jsonl"
            count = scraper.save_to_jsonl(output_file, config['news_articles'])
            total_samples += count
            
            print(f"   ✅ News: {count} articles saved")
            
        except Exception as e:
            print(f"   ❌ News failed: {e}")
    
    elapsed = (time.time() - start_time) / 60
    
    print(f"\n✅ {language_name} complete!")
    print(f"   Total samples: {total_samples}")
    print(f"   Time: {elapsed:.1f} minutes")
    
    return total_samples


def main():
    """Scrape all languages"""
    
    print("="*80)
    print("🚀 BATCH SCRAPING - 8 NEW LANGUAGES")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)
    
    total_start = time.time()
    results = {}
    
    # Scrape each language
    for language_name, config in LANGUAGES.items():
        samples = scrape_language(language_name, config)
        results[language_name] = samples
        
        # Brief pause between languages
        time.sleep(5)
    
    # Summary
    total_elapsed = (time.time() - total_start) / 3600
    total_samples = sum(results.values())
    
    print(f"\n{'='*80}")
    print("📊 SCRAPING COMPLETE!")
    print(f"{'='*80}")
    print(f"Total time: {total_elapsed:.2f} hours")
    print(f"Total samples: {total_samples}")
    print(f"\nResults by language:")
    for lang, count in results.items():
        status = "✅" if count > 0 else "❌"
        print(f"   {status} {lang:15} - {count:4} samples")
    
    print(f"\n📁 All data saved to: {OUTPUT_DIR}/")
    print(f"{'='*80}\n")
    
    # Next steps
    print("📋 NEXT STEPS:")
    print("1. Check data quality in data/training_new/")
    print("2. Upload to Google Colab")
    print("3. Use FREE_COLAB_TRAINING.ipynb to train adapters")
    print("4. Train one language at a time (2-3 hours each)")
    print("\n🎯 You now have data for 8 new languages!")


if __name__ == "__main__":
    main()

