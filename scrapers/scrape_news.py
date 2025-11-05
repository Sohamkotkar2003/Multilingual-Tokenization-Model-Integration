#!/usr/bin/env python3
"""
News RSS/Web Scraper for Low-Resource Languages

Purpose:
- Fetch news articles from RSS feeds in target languages
- Extract article text from news websites
- Save structured data for adapter training

Usage:
    python scrapers/scrape_news.py --language vi --max_articles 100
    
Example:
    # Scrape Vietnamese news from RSS feeds
    python scrapers/scrape_news.py --language vi --max_articles 100 --output data/streaming/news_vietnamese.jsonl
"""

import requests
import time
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Generator
from datetime import datetime
import feedparser
from bs4 import BeautifulSoup
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# RSS Feed sources for different languages
NEWS_FEEDS = {
    "vietnamese": [
        "https://vnexpress.net/rss/tin-moi-nhat.rss",
        "https://vietnamnet.vn/rss/home.rss",
        "https://dantri.com.vn/rss.htm",
    ],
    "thai": [
        "https://www.thairath.co.th/rss/news",
        "https://www.bangkokpost.com/rss/data/news.xml",
    ],
    "burmese": [
        "http://www.mizzimaburmese.com/rss",
        "https://burmese.dvb.no/feed",
    ],
    "sinhala": [
        "http://sinhala.adaderana.lk/rss.php",
        "https://www.lankadeepa.lk/rss/latest_news",
    ],
    "pashto": [
        "https://www.bbc.com/pashto/index.xml",
        "https://www.rferl.org/api/zr-qtepiev",
    ],
    "dari": [
        "https://www.bbc.com/persian/index.xml",
        "https://da.azadiradio.com/api/zmgqepi_io",
    ],
}

class NewsScraper:
    """Scraper for news articles from RSS feeds"""
    
    def __init__(self, language: str, feeds: Optional[List[str]] = None, delay: float = 2.0):
        """
        Initialize news scraper
        
        Args:
            language: Language name (e.g., 'vietnamese', 'thai')
            feeds: Optional list of RSS feed URLs (uses default if None)
            delay: Delay between requests in seconds
        """
        self.language = language
        self.feeds = feeds or NEWS_FEEDS.get(language, [])
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Educational Research Bot) Gurukul/1.0'
        })
        
        if not self.feeds:
            logger.warning(f"No RSS feeds configured for language '{language}'")
    
    def fetch_rss_entries(self, feed_url: str) -> List[Dict]:
        """
        Fetch entries from an RSS feed
        
        Args:
            feed_url: RSS feed URL
            
        Returns:
            List of feed entries
        """
        try:
            logger.info(f"Fetching RSS feed: {feed_url}")
            
            # Parse RSS feed
            feed = feedparser.parse(feed_url)
            
            if feed.bozo:  # Feed has parsing errors
                logger.warning(f"RSS feed has parsing errors: {feed_url}")
            
            entries = feed.entries
            logger.info(f"Found {len(entries)} entries in feed")
            return entries
            
        except Exception as e:
            logger.error(f"Failed to fetch RSS feed {feed_url}: {e}")
            return []
    
    def extract_article_text(self, url: str) -> Optional[str]:
        """
        Extract article text from a news URL
        
        Args:
            url: Article URL
            
        Returns:
            Article text or None if failed
        """
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Try common article content selectors
            article_selectors = [
                'article',
                '.article-content',
                '.post-content',
                '.entry-content',
                '.content',
                'main',
                '#content',
            ]
            
            article_text = None
            for selector in article_selectors:
                article = soup.select_one(selector)
                if article:
                    # Get all paragraphs
                    paragraphs = article.find_all('p')
                    if paragraphs:
                        article_text = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                        break
            
            # Fallback: get all paragraphs from body
            if not article_text:
                paragraphs = soup.find_all('p')
                if paragraphs:
                    article_text = '\n\n'.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            if not article_text or len(article_text) < 100:
                return None
            
            # Clean up text
            article_text = re.sub(r'\n{3,}', '\n\n', article_text)  # Remove excessive newlines
            article_text = re.sub(r' {2,}', ' ', article_text)  # Remove excessive spaces
            
            return article_text.strip()
            
        except Exception as e:
            logger.debug(f"Failed to extract article from {url}: {e}")
            return None
    
    def scrape_articles(self, max_articles: int = 100) -> Generator[Dict[str, str], None, None]:
        """
        Scrape articles from RSS feeds
        
        Args:
            max_articles: Maximum number of articles to scrape
            
        Yields:
            Article dictionaries
        """
        scraped = 0
        attempted = 0
        
        logger.info(f"Starting news scraping for language '{self.language}' (target: {max_articles} articles)")
        
        for feed_url in self.feeds:
            if scraped >= max_articles:
                break
            
            # Fetch RSS entries
            entries = self.fetch_rss_entries(feed_url)
            
            for entry in entries:
                if scraped >= max_articles:
                    break
                
                attempted += 1
                
                # Get article URL
                article_url = entry.get('link', '')
                if not article_url:
                    continue
                
                # Get title
                title = entry.get('title', 'Untitled')
                
                # Get summary (fallback if full text extraction fails)
                summary = entry.get('summary', '')
                
                # Try to extract full article text
                article_text = self.extract_article_text(article_url)
                
                # Use summary if full text extraction failed
                if not article_text and summary:
                    article_text = BeautifulSoup(summary, 'html.parser').get_text().strip()
                
                # Quality check
                if not article_text or len(article_text) < 100:
                    logger.debug(f"❌ [{attempted}] Skipped (too short): {title}")
                    continue
                
                # Get published date
                published = entry.get('published', entry.get('updated', ''))
                
                article_data = {
                    'title': title,
                    'url': article_url,
                    'text': article_text,
                    'language': self.language,
                    'source': 'news_rss',
                    'published': published,
                    'char_count': len(article_text),
                    'word_count': len(article_text.split()),
                    'feed_url': feed_url,
                }
                
                scraped += 1
                logger.info(f"✅ [{scraped}/{max_articles}] Scraped: {title[:50]}... ({article_data['char_count']} chars)")
                yield article_data
                
                # Delay between requests
                time.sleep(self.delay)
        
        logger.info(f"Scraping complete: {scraped} articles scraped, {attempted} articles attempted")
    
    def save_to_jsonl(self, output_path: Path, max_articles: int = 100) -> int:
        """
        Scrape articles and save to JSONL file
        
        Args:
            output_path: Path to output JSONL file
            max_articles: Maximum number of articles to scrape
            
        Returns:
            Number of articles saved
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        with output_path.open('w', encoding='utf-8') as f:
            for article in self.scrape_articles(max_articles):
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
                count += 1
        
        logger.info(f"💾 Saved {count} articles to {output_path}")
        return count


def main():
    parser = argparse.ArgumentParser(description="Scrape news articles for low-resource languages")
    parser.add_argument('--language', '-l', required=True,
                       help='Language name (e.g., vietnamese, thai, sinhala)')
    parser.add_argument('--max_articles', '-n', type=int, default=100,
                       help='Maximum number of articles to scrape (default: 100)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output JSONL file path (default: data/streaming/news_{language}.jsonl)')
    parser.add_argument('--delay', '-d', type=float, default=2.0,
                       help='Delay between requests in seconds (default: 2.0)')
    parser.add_argument('--feeds', '-f', nargs='+', default=None,
                       help='Custom RSS feed URLs (optional)')
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        args.output = f"data/streaming/news_{args.language}.jsonl"
    
    # Create scraper
    scraper = NewsScraper(
        language=args.language,
        feeds=args.feeds,
        delay=args.delay
    )
    
    # Scrape and save
    output_path = Path(args.output)
    count = scraper.save_to_jsonl(output_path, args.max_articles)
    
    print(f"\n{'='*60}")
    print(f"✅ News scraping complete!")
    print(f"   Language: {args.language}")
    print(f"   Articles scraped: {count}")
    print(f"   Output: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

