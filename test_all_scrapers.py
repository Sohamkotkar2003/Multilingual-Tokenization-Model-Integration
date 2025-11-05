#!/usr/bin/env python3
"""
Comprehensive Scraper Test for All 17 New Languages

Purpose:
- Test Wikipedia AND News scrapers for all target languages
- Generate quality report showing what data we're getting
- Identify any issues before full-scale scraping

Usage:
    python test_all_scrapers.py
"""

import sys
import io
import json
import time
from pathlib import Path
from datetime import datetime

# Force UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from scrapers.scrape_wiki import WikipediaScraper
from scrapers.scrape_news import NewsScraper

# Test configuration: 17 new languages
TEST_LANGUAGES = {
    # Indo-Aryan Languages (Hindi Belt Dialects)
    "Awadhi": {"code": "awa", "wiki_available": False, "news_available": False},
    "Bhojpuri": {"code": "bho", "wiki_available": False, "news_available": False},
    "Magahi": {"code": "mag", "wiki_available": False, "news_available": False},
    "Chhattisgarhi": {"code": "hne", "wiki_available": False, "news_available": False},
    "Haryanvi": {"code": "bgc", "wiki_available": False, "news_available": False},
    
    # South Asian Languages
    "Sinhala": {"code": "si", "wiki_available": True, "news_available": True, "news_name": "sinhala"},
    
    # Tibeto-Burman Languages
    "Tibetan": {"code": "bo", "wiki_available": True, "news_available": False},
    "Dzongkha": {"code": "dz", "wiki_available": True, "news_available": False},
    "Mizo": {"code": "lus", "wiki_available": False, "news_available": False},
    
    # Iranian Languages
    "Pashto": {"code": "ps", "wiki_available": True, "news_available": True, "news_name": "pashto"},
    "Dari": {"code": "prs", "wiki_available": False, "news_available": True, "news_name": "dari"},
    
    # Southeast Asian Languages
    "Vietnamese": {"code": "vi", "wiki_available": True, "news_available": True, "news_name": "vietnamese"},
    "Thai": {"code": "th", "wiki_available": True, "news_available": True, "news_name": "thai"},
    "Burmese": {"code": "my", "wiki_available": True, "news_available": True, "news_name": "burmese"},
}

SAMPLES_PER_LANGUAGE = 10  # Test with 10 samples each (Wikipedia)
NEWS_SAMPLES = 5  # Test with 5 news articles per language

def test_news_scraper(language_name: str, config: dict, output_dir: Path) -> dict:
    """
    Test news scraping for a single language
    
    Returns:
        Test results dictionary
    """
    result = {
        'language': language_name,
        'source': 'news',
        'success': False,
        'samples_scraped': 0,
        'avg_length': 0,
        'topics': [],
        'error': None
    }
    
    # Skip if no news feeds available
    if not config.get('news_available', False):
        result['error'] = 'No news feeds configured'
        return result
    
    print(f"\n  📰 Testing NEWS scraper for {language_name}...")
    
    try:
        # Create news scraper
        news_name = config.get('news_name', language_name.lower())
        scraper = NewsScraper(
            language=news_name,
            delay=1.0
        )
        
        if not scraper.feeds:
            result['error'] = 'No RSS feeds configured'
            print(f"     ⚠️  No RSS feeds configured for {language_name}")
            return result
        
        # Test scraping
        output_file = output_dir / f"test_news_{config['code']}.jsonl"
        
        articles = []
        for article in scraper.scrape_articles(max_articles=NEWS_SAMPLES):
            articles.append(article)
            result['samples_scraped'] += 1
            print(f"     ✅ [{result['samples_scraped']}/{NEWS_SAMPLES}] {article['title'][:50]}... ({article['char_count']} chars)")
        
        # Save articles
        if articles:
            with open(output_file, 'w', encoding='utf-8') as f:
                for article in articles:
                    f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        # Calculate stats
        result['success'] = len(articles) > 0
        result['avg_length'] = int(sum(a['char_count'] for a in articles) / len(articles)) if articles else 0
        result['topics'] = [a['title'] for a in articles[:3]]
        
        if result['success']:
            print(f"     ✅ News test: {result['samples_scraped']} articles, {result['avg_length']} chars avg")
        
        return result
        
    except Exception as e:
        print(f"     ❌ News test FAILED: {e}")
        result['error'] = str(e)
        return result


def test_language(language_name: str, config: dict, output_dir: Path) -> dict:
    """
    Test scraping for a single language (both Wikipedia and News)
    
    Returns:
        Test results dictionary
    """
    print(f"\n{'='*80}")
    print(f"Testing: {language_name} ({config['code']})")
    print(f"{'='*80}")
    
    result = {
        'language': language_name,
        'code': config['code'],
        'wiki_available': config.get('wiki_available', False),
        'news_available': config.get('news_available', False),
        'wiki_success': False,
        'news_success': False,
        'wiki_samples': 0,
        'news_samples': 0,
        'wiki_avg_length': 0,
        'news_avg_length': 0,
        'wiki_topics': [],
        'news_topics': [],
        'wiki_error': None,
        'news_error': None,
        'overall_success': False
    }
    
    # Test 1: Wikipedia Scraper
    if config.get('wiki_available', False):
        print(f"\n  🌐 Testing WIKIPEDIA scraper for {language_name}...")
        
        try:
            scraper = WikipediaScraper(
                language_code=config['code'],
                delay=0.5,
                enable_quality_filter=True
            )
            
            output_file = output_dir / f"test_wiki_{config['code']}.jsonl"
            samples = []
            
            for page in scraper.scrape_pages(max_pages=SAMPLES_PER_LANGUAGE):
                samples.append(page)
                result['wiki_samples'] += 1
                print(f"     ✅ [{result['wiki_samples']}/{SAMPLES_PER_LANGUAGE}] {page['title'][:50]}... ({page['char_count']} chars)")
            
            if samples:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for sample in samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                
                result['wiki_success'] = True
                result['wiki_avg_length'] = int(sum(s['char_count'] for s in samples) / len(samples))
                result['wiki_topics'] = [s['title'] for s in samples[:3]]
                print(f"     ✅ Wiki test: {result['wiki_samples']} samples, {result['wiki_avg_length']} chars avg")
        
        except Exception as e:
            print(f"     ❌ Wiki test FAILED: {e}")
            result['wiki_error'] = str(e)
    else:
        result['wiki_error'] = 'No dedicated Wikipedia'
        print(f"  ⚠️  No Wikipedia available for {language_name}")
    
    # Test 2: News Scraper
    news_result = test_news_scraper(language_name, config, output_dir)
    result['news_success'] = news_result['success']
    result['news_samples'] = news_result['samples_scraped']
    result['news_avg_length'] = news_result['avg_length']
    result['news_topics'] = news_result['topics']
    result['news_error'] = news_result['error']
    
    # Overall success if at least one source works
    result['overall_success'] = result['wiki_success'] or result['news_success']
    
    # Summary
    print(f"\n  📊 {language_name} Summary:")
    if result['wiki_success']:
        print(f"     ✅ Wikipedia: {result['wiki_samples']} samples ({result['wiki_avg_length']} chars avg)")
    elif result['wiki_error']:
        print(f"     ⚠️  Wikipedia: {result['wiki_error']}")
    
    if result['news_success']:
        print(f"     ✅ News: {result['news_samples']} articles ({result['news_avg_length']} chars avg)")
    elif result['news_error']:
        print(f"     ⚠️  News: {result['news_error']}")
    
    if not result['overall_success']:
        print(f"     ❌ No data sources available")
    
    return result


def generate_report(results: list, output_file: Path):
    """Generate comprehensive quality report"""
    
    report = []
    report.append("# Language Scraping Quality Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"## Summary\n")
    
    total_languages = len(results)
    successful = sum(1 for r in results if r.get('overall_success', False))
    wiki_success = sum(1 for r in results if r.get('wiki_success', False))
    news_success = sum(1 for r in results if r.get('news_success', False))
    no_sources = sum(1 for r in results if not r.get('overall_success', False))
    
    report.append(f"- **Total languages tested:** {total_languages}")
    report.append(f"- **✅ Languages with data:** {successful}")
    report.append(f"- **🌐 Wikipedia sources:** {wiki_success}")
    report.append(f"- **📰 News sources:** {news_success}")
    report.append(f"- **❌ No sources available:** {no_sources}\n")
    
    report.append("## Results by Language\n")
    report.append("| Language | Code | Wikipedia | News | Total Samples | Best Avg Length |\n")
    report.append("|----------|------|-----------|------|---------------|-----------------|")
    
    for r in results:
        wiki_status = "✅" if r.get('wiki_success') else "❌"
        news_status = "✅" if r.get('news_success') else "❌"
        
        total_samples = r.get('wiki_samples', 0) + r.get('news_samples', 0)
        
        # Get best avg length from either source
        wiki_len = r.get('wiki_avg_length', 0)
        news_len = r.get('news_avg_length', 0)
        best_len = max(wiki_len, news_len)
        
        # Format length display
        if best_len >= 3000:
            len_display = f"🌟 {best_len}"
        elif best_len >= 1500:
            len_display = f"✅ {best_len}"
        elif best_len > 0:
            len_display = f"⚠️ {best_len}"
        else:
            len_display = "❌ 0"
        
        report.append(f"| {r['language']} | `{r['code']}` | {wiki_status} | {news_status} | {total_samples} | {len_display} |")
    
    report.append("\n## Detailed Results\n")
    
    for r in results:
        report.append(f"### {r['language']} (`{r['code']}`)\n")
        
        # Wikipedia results
        if r.get('wiki_success'):
            report.append(f"**Wikipedia:** ✅ Success")
            report.append(f"- Samples: {r['wiki_samples']}")
            report.append(f"- Avg length: {r['wiki_avg_length']} chars")
            if r.get('wiki_topics'):
                report.append(f"- Sample topics: {', '.join(r['wiki_topics'][:3])}")
            report.append("")
        elif r.get('wiki_error'):
            report.append(f"**Wikipedia:** ⚠️ {r['wiki_error']}\n")
        
        # News results
        if r.get('news_success'):
            report.append(f"**News:** ✅ Success")
            report.append(f"- Articles: {r['news_samples']}")
            report.append(f"- Avg length: {r['news_avg_length']} chars")
            if r.get('news_topics'):
                report.append(f"- Sample topics: {', '.join(r['news_topics'][:3])}")
            report.append("")
        elif r.get('news_error') and r['news_error'] != 'No news feeds configured':
            report.append(f"**News:** ⚠️ {r['news_error']}\n")
        
        # Overall status
        if r.get('overall_success'):
            total = r.get('wiki_samples', 0) + r.get('news_samples', 0)
            report.append(f"**Overall:** ✅ {total} total samples from {1 if r['wiki_success'] == r['news_success'] else 2} source(s)\n")
        else:
            report.append(f"**Overall:** ❌ No data sources available\n")
    
    report.append("\n## Next Steps\n")
    report.append("### ✅ Languages Ready for Training")
    ready_langs = [r for r in results if r.get('overall_success', False)]
    for r in ready_langs:
        sources = []
        if r.get('wiki_success'):
            sources.append(f"Wiki: {r['wiki_samples']} samples, {r['wiki_avg_length']} chars")
        if r.get('news_success'):
            sources.append(f"News: {r['news_samples']} articles, {r['news_avg_length']} chars")
        report.append(f"- **{r['language']}** (`{r['code']}`) - {' | '.join(sources)}")
    
    report.append("\n### ❌ Languages Without Data Sources")
    no_data_langs = [r for r in results if not r.get('overall_success', False)]
    if no_data_langs:
        for r in no_data_langs:
            report.append(f"- **{r['language']}** (`{r['code']}`) - Need alternative data sources")
        report.append("\n**Recommendation:** Use existing Hindi/Maithili data for bootstrapping Indo-Aryan dialects")
    else:
        report.append("None! All languages have data sources ✅")
    
    # Save report
    report_text = '\n'.join(report)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return report_text


def main():
    """Run comprehensive scraper test"""
    
    print("="*80)
    print("🧪 COMPREHENSIVE SCRAPER QUALITY TEST")
    print("Testing all 17 new languages with quality filters")
    print("="*80)
    
    # Create output directory
    output_dir = Path("data/scraper_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test all languages
    results = []
    
    for language_name, config in TEST_LANGUAGES.items():
        result = test_language(language_name, config, output_dir)
        results.append(result)
        time.sleep(1)  # Brief pause between languages
    
    # Generate report
    print(f"\n{'='*80}")
    print("📊 Generating quality report...")
    print(f"{'='*80}")
    
    report_file = Path("bridge/tests/scraper_quality_report.md")
    report_file.parent.mkdir(parents=True, exist_ok=True)
    
    report_text = generate_report(results, report_file)
    
    # Print summary
    print("\n" + "="*80)
    print("📋 SUMMARY")
    print("="*80)
    print(report_text)
    
    print(f"\n{'='*80}")
    print(f"✅ Test complete!")
    print(f"   Report saved to: {report_file}")
    print(f"   Test data saved to: {output_dir}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

