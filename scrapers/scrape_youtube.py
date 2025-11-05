#!/usr/bin/env python3
"""
YouTube Subtitle Scraper for Low-Resource Languages

Purpose:
- Extract subtitles from YouTube videos in target languages
- Get natural, spoken language data (songs, movies, vlogs)
- Build training corpus for languages without Wikipedia

Requirements:
    pip install yt-dlp

Usage:
    python scrapers/scrape_youtube.py --language bhojpuri --max_videos 50
    
Example:
    # Scrape 50 Bhojpuri videos
    python scrapers/scrape_youtube.py -l bhojpuri -n 50 -o data/streaming/youtube_bhojpuri.jsonl
"""

import subprocess
import json
import re
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Generator
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# YouTube search queries for each language
YOUTUBE_QUERIES = {
    "bhojpuri": [
        "bhojpuri movie",
        "bhojpuri song",
        "bhojpuri news",
        "bhojpuri comedy",
        "भोजपुरी फिल्म",
    ],
    "awadhi": [
        "awadhi song",
        "awadhi bhajan",
        "अवधी गीत",
    ],
    "magahi": [
        "magahi song",
        "magahi folk",
        "मगही गीत",
    ],
    "chhattisgarhi": [
        "chhattisgarhi song",
        "chhattisgarhi movie",
        "छत्तीसगढ़ी फिल्म",
    ],
    "haryanvi": [
        "haryanvi song",
        "haryanvi ragni",
        "haryanvi comedy",
        "हरियाणवी गाना",
    ],
    "mizo": [
        "mizo song",
        "mizo hla thar",
        "mizo news",
    ],
    "himachali": [
        "himachali song",
        "himachali pahari geet",
    ],
    "pahadi": [
        "pahadi song",
        "garhwali kumaoni song",
    ],
}

class YouTubeScraper:
    """Scraper for YouTube subtitles in various languages"""
    
    def __init__(self, language: str, delay: float = 2.0):
        """
        Initialize YouTube scraper
        
        Args:
            language: Language name (e.g., 'bhojpuri', 'awadhi')
            delay: Delay between video requests in seconds
        """
        self.language = language
        self.queries = YOUTUBE_QUERIES.get(language.lower(), [f"{language} song"])
        self.delay = delay
        self.seen_videos = set()
        
        # Check if yt-dlp is available
        if not self._check_ytdlp():
            raise RuntimeError("yt-dlp not installed. Install with: pip install yt-dlp")
    
    def _check_ytdlp(self) -> bool:
        """Check if yt-dlp is installed"""
        try:
            result = subprocess.run(['yt-dlp', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False
    
    def search_videos(self, query: str, max_results: int = 10) -> List[str]:
        """
        Search YouTube for videos
        
        Args:
            query: Search query
            max_results: Maximum number of video URLs to return
            
        Returns:
            List of YouTube video URLs
        """
        try:
            logger.info(f"Searching YouTube: '{query}'")
            
            # Use yt-dlp to search
            cmd = [
                'yt-dlp',
                f'ytsearch{max_results}:{query}',
                '--get-id',
                '--no-warnings',
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                logger.error(f"Search failed: {result.stderr}")
                return []
            
            # Parse video IDs
            video_ids = result.stdout.strip().split('\n')
            video_urls = [f'https://www.youtube.com/watch?v={vid}' for vid in video_ids if vid]
            
            logger.info(f"Found {len(video_urls)} videos")
            return video_urls
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def extract_subtitles(self, video_url: str) -> Optional[Dict[str, str]]:
        """
        Extract subtitles from a YouTube video
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Dict with 'title', 'url', 'text', 'language' or None if failed
        """
        try:
            # Skip if already processed
            if video_url in self.seen_videos:
                return None
            
            self.seen_videos.add(video_url)
            
            logger.debug(f"Extracting from: {video_url}")
            
            # Get video info and subtitles
            cmd = [
                'yt-dlp',
                '--skip-download',
                '--write-auto-sub',
                '--sub-lang', 'hi,en',  # Try Hindi or English auto-generated subs
                '--convert-subs', 'srt',
                '--output', 'temp_subtitle',
                '--no-warnings',
                '--get-title',
                video_url
            ]
            
            # Get title first
            title_cmd = ['yt-dlp', '--get-title', '--no-warnings', video_url]
            title_result = subprocess.run(title_cmd, capture_output=True, text=True, timeout=15)
            title = title_result.stdout.strip() if title_result.returncode == 0 else "Unknown"
            
            # Try to get subtitles
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # Check if subtitle file was created
            subtitle_files = list(Path('.').glob('temp_subtitle*.srt'))
            
            if not subtitle_files:
                logger.debug(f"No subtitles available for: {title}")
                return None
            
            # Read subtitle file
            subtitle_file = subtitle_files[0]
            subtitle_text = subtitle_file.read_text(encoding='utf-8')
            
            # Clean subtitle text (remove timestamps, duplicates)
            cleaned_text = self._clean_subtitles(subtitle_text)
            
            # Clean up temp file
            subtitle_file.unlink()
            
            if len(cleaned_text) < 100:
                logger.debug(f"Subtitles too short for: {title}")
                return None
            
            logger.info(f"✅ Extracted subtitles from: {title[:50]}... ({len(cleaned_text)} chars)")
            
            return {
                'title': title,
                'url': video_url,
                'text': cleaned_text,
                'language': self.language,
                'source': 'youtube_subtitles',
                'char_count': len(cleaned_text),
                'word_count': len(cleaned_text.split())
            }
            
        except Exception as e:
            logger.debug(f"Extraction failed for {video_url}: {e}")
            
            # Clean up any temp files
            for f in Path('.').glob('temp_subtitle*'):
                try:
                    f.unlink()
                except:
                    pass
            
            return None
    
    def _clean_subtitles(self, subtitle_text: str) -> str:
        """
        Clean SRT subtitle text
        
        Args:
            subtitle_text: Raw SRT subtitle content
            
        Returns:
            Cleaned text
        """
        # Remove SRT timestamp lines
        lines = subtitle_text.split('\n')
        text_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip sequence numbers
            if line.isdigit():
                continue
            
            # Skip timestamp lines (format: 00:00:00,000 --> 00:00:00,000)
            if '-->' in line:
                continue
            
            # Skip lines with XML tags
            if '<' in line and '>' in line:
                continue
            
            text_lines.append(line)
        
        # Join and clean
        text = ' '.join(text_lines)
        
        # Remove duplicate spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove [Music], [Applause], etc.
        text = re.sub(r'\[.*?\]', '', text)
        
        return text.strip()
    
    def scrape_videos(self, max_videos: int = 50) -> Generator[Dict[str, str], None, None]:
        """
        Scrape subtitles from multiple YouTube videos
        
        Args:
            max_videos: Maximum number of videos to process
            
        Yields:
            Subtitle dictionaries
        """
        scraped = 0
        attempted = 0
        
        logger.info(f"Starting YouTube scrape for language '{self.language}' (target: {max_videos} videos)")
        
        for query in self.queries:
            if scraped >= max_videos:
                break
            
            # Search for videos
            batch_size = min(20, max_videos - scraped)
            video_urls = self.search_videos(query, batch_size)
            
            for video_url in video_urls:
                if scraped >= max_videos:
                    break
                
                attempted += 1
                
                # Extract subtitles
                subtitle_data = self.extract_subtitles(video_url)
                
                if subtitle_data:
                    scraped += 1
                    logger.info(f"[{scraped}/{max_videos}] {subtitle_data['title'][:50]}...")
                    yield subtitle_data
                else:
                    logger.debug(f"[{attempted}] No subtitles: {video_url}")
                
                # Delay between requests
                time.sleep(self.delay)
        
        logger.info(f"Scraping complete: {scraped} videos processed, {attempted} videos attempted")
    
    def save_to_jsonl(self, output_path: Path, max_videos: int = 50) -> int:
        """
        Scrape videos and save to JSONL file
        
        Args:
            output_path: Path to output JSONL file
            max_videos: Maximum number of videos to scrape
            
        Returns:
            Number of videos saved
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        count = 0
        with output_path.open('w', encoding='utf-8') as f:
            for video in self.scrape_videos(max_videos):
                f.write(json.dumps(video, ensure_ascii=False) + '\n')
                count += 1
        
        logger.info(f"💾 Saved {count} videos to {output_path}")
        return count


def main():
    parser = argparse.ArgumentParser(description="Scrape YouTube subtitles for low-resource languages")
    parser.add_argument('--language', '-l', required=True,
                       help='Language name (e.g., bhojpuri, awadhi, haryanvi)')
    parser.add_argument('--max_videos', '-n', type=int, default=50,
                       help='Maximum number of videos to scrape (default: 50)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output JSONL file path (default: data/streaming/youtube_{language}.jsonl)')
    parser.add_argument('--delay', '-d', type=float, default=2.0,
                       help='Delay between requests in seconds (default: 2.0)')
    parser.add_argument('--queries', '-q', nargs='+', default=None,
                       help='Custom search queries (optional)')
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        args.output = f"data/streaming/youtube_{args.language}.jsonl"
    
    # Override queries if provided
    if args.queries:
        YOUTUBE_QUERIES[args.language] = args.queries
    
    # Create scraper
    try:
        scraper = YouTubeScraper(
            language=args.language,
            delay=args.delay
        )
        
        # Scrape and save
        output_path = Path(args.output)
        count = scraper.save_to_jsonl(output_path, args.max_videos)
        
        print(f"\n{'='*60}")
        print(f"✅ YouTube scraping complete!")
        print(f"   Language: {args.language}")
        print(f"   Videos processed: {count}")
        print(f"   Output: {output_path}")
        print(f"{'='*60}\n")
    
    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
        print(f"   Install yt-dlp: pip install yt-dlp\n")
        return 1


if __name__ == "__main__":
    main()

