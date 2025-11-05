#!/usr/bin/env python3
"""
Social Media Text Scraper for Low-Resource Languages

Purpose:
- Scrape public social media posts in target languages
- Extract text from Twitter/X, Facebook public pages, Reddit
- Build training corpus from contemporary language use

Requirements:
    pip install tweepy praw beautifulsoup4

Usage:
    python scrapers/scrape_social.py --language bhojpuri --source twitter --max_posts 100
    
Example:
    # Scrape 100 Bhojpuri tweets
    python scrapers/scrape_social.py -l bhojpuri -s twitter -n 100
"""

import requests
import json
import re
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Generator
import time
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Language-specific hashtags and keywords
SOCIAL_KEYWORDS = {
    "bhojpuri": ["#bhojpuri", "#भोजपुरी", "bhojpuri", "भोजपुरी"],
    "awadhi": ["#awadhi", "#अवधी", "awadhi", "अवधी"],
    "magahi": ["#magahi", "#मगही", "magahi", "मगही"],
    "chhattisgarhi": ["#chhattisgarhi", "#छत्तीसगढ़ी", "chhattisgarhi"],
    "haryanvi": ["#haryanvi", "#हरियाणवी", "haryanvi", "हरियाणवी"],
    "mizo": ["#mizo", "#mizoram", "mizo", "mi zo"],
    "himachali": ["#himachali", "#हिमाचली", "himachali"],
    "pahadi": ["#pahadi", "#पहाड़ी", "pahadi", "garhwali", "kumaoni"],
}

class SocialMediaScraper:
    """
    Scraper for social media content in various languages
    
    Note: This is a basic scraper that uses public APIs and web scraping.
    For production use, you may need API keys for Twitter, Reddit, etc.
    """
    
    def __init__(self, language: str, source: str = 'web', delay: float = 2.0):
        """
        Initialize social media scraper
        
        Args:
            language: Language name (e.g., 'bhojpuri')
            source: Source platform ('twitter', 'reddit', 'web')
            delay: Delay between requests
        """
        self.language = language
        self.source = source
        self.keywords = SOCIAL_KEYWORDS.get(language.lower(), [language])
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational Research Bot) Gurukul/1.0'
        })
    
    def scrape_nitter(self, max_posts: int = 100) -> Generator[Dict[str, str], None, None]:
        """
        Scrape tweets via Nitter (public Twitter mirror)
        
        Args:
            max_posts: Maximum posts to scrape
            
        Yields:
            Post dictionaries
        """
        # Nitter instances (public Twitter mirrors)
        nitter_instances = [
            'https://nitter.net',
            'https://nitter.privacydev.net',
            'https://nitter.poast.org',
        ]
        
        scraped = 0
        
        for keyword in self.keywords:
            if scraped >= max_posts:
                break
            
            for nitter_url in nitter_instances:
                if scraped >= max_posts:
                    break
                
                try:
                    # Search on Nitter
                    search_url = f"{nitter_url}/search?f=tweets&q={keyword}"
                    logger.info(f"Searching: {search_url}")
                    
                    response = self.session.get(search_url, timeout=10)
                    
                    if response.status_code != 200:
                        continue
                    
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Find tweet content
                    tweets = soup.find_all('div', class_='tweet-content')
                    
                    for tweet in tweets:
                        if scraped >= max_posts:
                            break
                        
                        text = tweet.get_text().strip()
                        
                        # Quality filters
                        if len(text) < 20:  # Too short
                            continue
                        
                        if text.startswith('RT @'):  # Skip retweets
                            continue
                        
                        scraped += 1
                        
                        yield {
                            'text': text,
                            'language': self.language,
                            'source': 'twitter_nitter',
                            'keyword': keyword,
                            'char_count': len(text),
                            'word_count': len(text.split())
                        }
                        
                        logger.info(f"[{scraped}/{max_posts}] Tweet: {text[:50]}...")
                    
                    time.sleep(self.delay)
                    
                except Exception as e:
                    logger.debug(f"Nitter instance {nitter_url} failed: {e}")
                    continue
    
    def scrape_web_search(self, max_results: int = 100) -> Generator[Dict[str, str], None, None]:
        """
        Scrape text from general web search results
        
        Args:
            max_results: Maximum results to scrape
            
        Yields:
            Text dictionaries
        """
        logger.info(f"Web search scraping not yet implemented")
        logger.info(f"Recommendation: Use DuckDuckGo or Google search to find {self.language} blogs/forums")
        return
        yield  # Make it a generator
    
    def scrape_posts(self, max_posts: int = 100) -> Generator[Dict[str, str], None, None]:
        """
        Scrape social media posts
        
        Args:
            max_posts: Maximum posts to scrape
            
        Yields:
            Post dictionaries
        """
        if self.source == 'twitter':
            yield from self.scrape_nitter(max_posts)
        elif self.source == 'web':
            yield from self.scrape_web_search(max_posts)
        else:
            logger.warning(f"Source '{self.source}' not yet implemented")
    
    def save_to_jsonl(self, output_path: Path, max_posts: int = 100) -> int:
        """
        Scrape posts and save to JSONL file
        
        Args:
            output_path: Path to output JSONL file
            max_posts: Maximum posts to scrape
            
        Returns:
            Number of posts saved
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        with output_path.open('w', encoding='utf-8') as f:
            for post in self.scrape_posts(max_posts):
                f.write(json.dumps(post, ensure_ascii=False) + '\n')
                count += 1
        
        logger.info(f"💾 Saved {count} posts to {output_path}")
        return count


def main():
    parser = argparse.ArgumentParser(description="Scrape social media for low-resource languages")
    parser.add_argument('--language', '-l', required=True,
                       help='Language name (e.g., bhojpuri, awadhi, haryanvi)')
    parser.add_argument('--source', '-s', default='twitter',
                       choices=['twitter', 'web'],
                       help='Social media source (default: twitter)')
    parser.add_argument('--max_posts', '-n', type=int, default=100,
                       help='Maximum number of posts to scrape (default: 100)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output JSONL file path (default: data/streaming/social_{language}.jsonl)')
    parser.add_argument('--delay', '-d', type=float, default=2.0,
                       help='Delay between requests in seconds (default: 2.0)')
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        args.output = f"data/streaming/social_{args.language}.jsonl"
    
    # Create scraper
    scraper = SocialMediaScraper(
        language=args.language,
        source=args.source,
        delay=args.delay
    )
    
    # Scrape and save
    output_path = Path(args.output)
    count = scraper.save_to_jsonl(output_path, args.max_posts)
    
    print(f"\n{'='*60}")
    print(f"✅ Social media scraping complete!")
    print(f"   Language: {args.language}")
    print(f"   Posts scraped: {count}")
    print(f"   Output: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

