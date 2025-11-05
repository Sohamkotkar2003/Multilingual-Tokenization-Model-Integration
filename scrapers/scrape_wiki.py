#!/usr/bin/env python3
"""
Wikipedia Scraper for Low-Resource Languages

Purpose:
- Fetch Wikipedia pages in target languages
- Extract clean article text (remove navigation, infoboxes, etc.)
- Save structured data for adapter training

Usage:
    python scrapers/scrape_wiki.py --language si --max_pages 100
    
Example:
    # Scrape 100 Sinhala Wikipedia pages
    python scrapers/scrape_wiki.py --language si --max_pages 100 --output data/streaming/wiki_sinhala.jsonl
"""

import requests
import re
import time
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Generator
from urllib.parse import urljoin, quote
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Wikipedia language code mapping
WIKI_LANGUAGES = {
    "sinhala": "si",
    "tibetan": "bo", 
    "dzongkha": "dz",
    "pashto": "ps",
    "dari": "fa",  # Dari uses Persian Wikipedia with filters
    "vietnamese": "vi",
    "thai": "th",
    "burmese": "my",
    "haryanvi": "bgw",
    "nepali": "ne",
}

class WikipediaScraper:
    """Scraper for Wikipedia articles in various languages"""
    
    def __init__(self, language_code: str, base_url: Optional[str] = None, delay: float = 1.0):
        """
        Initialize Wikipedia scraper
        
        Args:
            language_code: Wikipedia language code (e.g., 'si' for Sinhala)
            base_url: Optional custom base URL (default: https://{lang}.wikipedia.org)
            delay: Delay between requests in seconds (be respectful!)
        """
        self.language_code = language_code
        self.base_url = base_url or f"https://{language_code}.wikipedia.org"
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational Research Bot for Low-Resource Languages) Gurukul/1.0'
        })
        
    def get_random_pages(self, count: int = 10) -> List[str]:
        """
        Get random Wikipedia page titles
        
        Args:
            count: Number of random pages to fetch
            
        Returns:
            List of page titles
        """
        try:
            url = f"{self.base_url}/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'list': 'random',
                'rnlimit': min(count, 500),  # Wikipedia API limit
                'rnnamespace': 0  # Main namespace only (articles)
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            pages = [page['title'] for page in data.get('query', {}).get('random', [])]
            logger.info(f"Fetched {len(pages)} random page titles")
            return pages
            
        except Exception as e:
            logger.error(f"Failed to fetch random pages: {e}")
            return []
    
    def fetch_page_content(self, title: str) -> Optional[Dict[str, str]]:
        """
        Fetch content of a Wikipedia page
        
        Args:
            title: Wikipedia page title
            
        Returns:
            Dict with 'title', 'url', 'text', 'language' or None if failed
        """
        try:
            # Use Wikipedia API to get clean article text
            url = f"{self.base_url}/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'explaintext': True,  # Plain text, no HTML
                'exsectionformat': 'plain'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Extract page content
            pages = data.get('query', {}).get('pages', {})
            if not pages:
                return None
                
            page = list(pages.values())[0]
            
            # Skip if page doesn't exist or is redirect
            if 'missing' in page or 'redirect' in page:
                logger.debug(f"Page '{title}' missing or redirect, skipping")
                return None
            
            extract = page.get('extract', '').strip()
            
            # Quality filters
            if len(extract) < 100:  # Too short
                logger.debug(f"Page '{title}' too short ({len(extract)} chars), skipping")
                return None
            
            if not extract:
                return None
            
            page_url = f"{self.base_url}/wiki/{quote(title)}"
            
            return {
                'title': title,
                'url': page_url,
                'text': extract,
                'language': self.language_code,
                'source': 'wikipedia',
                'char_count': len(extract),
                'word_count': len(extract.split())
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch page '{title}': {e}")
            return None
    
    def scrape_pages(self, max_pages: int = 100) -> Generator[Dict[str, str], None, None]:
        """
        Scrape multiple Wikipedia pages
        
        Args:
            max_pages: Maximum number of pages to scrape
            
        Yields:
            Page content dictionaries
        """
        scraped = 0
        attempted = 0
        
        logger.info(f"Starting Wikipedia scrape for language '{self.language_code}' (target: {max_pages} pages)")
        
        while scraped < max_pages and attempted < max_pages * 3:  # Try 3x to account for failures
            # Fetch batch of random pages
            batch_size = min(50, max_pages - scraped)
            titles = self.get_random_pages(batch_size)
            
            if not titles:
                logger.warning("No more pages available, stopping")
                break
            
            for title in titles:
                if scraped >= max_pages:
                    break
                
                attempted += 1
                
                # Fetch page content
                content = self.fetch_page_content(title)
                
                if content:
                    scraped += 1
                    logger.info(f"✅ [{scraped}/{max_pages}] Scraped: {title} ({content['char_count']} chars)")
                    yield content
                else:
                    logger.debug(f"❌ [{attempted}] Failed: {title}")
                
                # Be respectful - delay between requests
                time.sleep(self.delay)
        
        logger.info(f"Scraping complete: {scraped} pages scraped, {attempted} pages attempted")
    
    def save_to_jsonl(self, output_path: Path, max_pages: int = 100) -> int:
        """
        Scrape pages and save to JSONL file
        
        Args:
            output_path: Path to output JSONL file
            max_pages: Maximum number of pages to scrape
            
        Returns:
            Number of pages saved
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        with output_path.open('w', encoding='utf-8') as f:
            for page in self.scrape_pages(max_pages):
                f.write(json.dumps(page, ensure_ascii=False) + '\n')
                count += 1
        
        logger.info(f"💾 Saved {count} pages to {output_path}")
        return count


def main():
    parser = argparse.ArgumentParser(description="Scrape Wikipedia for low-resource languages")
    parser.add_argument('--language', '-l', required=True, 
                       help='Language code (e.g., si, bo, ps, vi)')
    parser.add_argument('--max_pages', '-n', type=int, default=100,
                       help='Maximum number of pages to scrape (default: 100)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output JSONL file path (default: data/streaming/wiki_{language}.jsonl)')
    parser.add_argument('--delay', '-d', type=float, default=1.0,
                       help='Delay between requests in seconds (default: 1.0)')
    parser.add_argument('--base_url', '-u', type=str, default=None,
                       help='Custom Wikipedia base URL (optional)')
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        args.output = f"data/streaming/wiki_{args.language}.jsonl"
    
    # Create scraper
    scraper = WikipediaScraper(
        language_code=args.language,
        base_url=args.base_url,
        delay=args.delay
    )
    
    # Scrape and save
    output_path = Path(args.output)
    count = scraper.save_to_jsonl(output_path, args.max_pages)
    
    print(f"\n{'='*60}")
    print(f"✅ Wikipedia scraping complete!")
    print(f"   Language: {args.language}")
    print(f"   Pages scraped: {count}")
    print(f"   Output: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

