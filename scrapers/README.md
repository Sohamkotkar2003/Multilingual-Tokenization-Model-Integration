# Web Scrapers & MCP Streaming for Language Expansion

This directory contains web scrapers and streaming tools for collecting training data for low-resource languages.

## 📦 Installation

```bash
# Install scraper dependencies
pip install -r scrapers/requirements.txt
```

## 🛠️ Tools

### 1. Wikipedia Scraper (`scrape_wiki.py`)

Scrapes Wikipedia articles in various languages.

**Usage:**
```bash
# Scrape 100 Sinhala Wikipedia pages
python scrapers/scrape_wiki.py --language si --max_pages 100

# Scrape 50 Tibetan pages with custom output
python scrapers/scrape_wiki.py -l bo -n 50 -o data/tibetan_wiki.jsonl

# Slower scraping (3 second delay between requests)
python scrapers/scrape_wiki.py -l ps -n 100 --delay 3.0
```

**Output format (JSONL):**
```json
{"title": "ශ්‍රී ලංකාව", "url": "https://si.wikipedia.org/...", "text": "...", "language": "si", "source": "wikipedia", "char_count": 1234, "word_count": 234}
```

### 2. News Scraper (`scrape_news.py`)

Scrapes news articles from RSS feeds.

**Usage:**
```bash
# Scrape 100 Vietnamese news articles
python scrapers/scrape_news.py --language vietnamese --max_articles 100

# Scrape Thai news with custom feeds
python scrapers/scrape_news.py -l thai -n 50 --feeds "https://example.com/rss"

# Available languages: vietnamese, thai, burmese, sinhala, pashto, dari
```

**Output format (JSONL):**
```json
{"title": "Breaking News", "url": "https://...", "text": "...", "language": "vietnamese", "source": "news_rss", "published": "2024-11-05", "char_count": 1500}
```

### 3. MCP Stream Client (`../mcp/stream_client.py`)

Unified streaming client that coordinates all data sources.

**Usage (Python):**
```python
from mcp.stream_client import stream_language_data

# Stream 1000 samples for Sinhala (auto-fetches from all sources)
for sample in stream_language_data('sinhala', max_samples=1000):
    print(f"[{sample['source']}] {sample['text'][:100]}...")
```

**Usage (CLI testing):**
```bash
# Test streaming 10 samples
python mcp/stream_client.py --language sinhala --max_samples 10

# Save to file for inspection
python mcp/stream_client.py -l vietnamese -n 100 -o data/test_vi.jsonl
```

## 🌍 Supported Languages

### Indo-Aryan Languages
- Awadhi (`awa`)
- Bhojpuri (`bho`)
- Magahi (`mag`)
- Chhattisgarhi (`hne`)
- Haryanvi (`bgc`)
- Himachali (`him`)
- Pahadi (`pah`)

### South Asian Languages
- Sinhala (`si`)
- Tamil-Sri Lanka (`ta_LK`)

### Tibeto-Burman Languages
- Tibetan (`bo`)
- Dzongkha (`dz`)
- Mizo (`lus`)

### Iranian Languages
- Pashto (`ps`)
- Dari (`prs`)

### Southeast Asian Languages
- Vietnamese (`vi`)
- Thai (`th`)
- Burmese (`my`)

## 📝 Configuration

Edit `mcp_connectors.yml` in the project root to:
- Add new data sources
- Adjust `max_samples` per source
- Add custom HuggingFace datasets
- Add new RSS feeds

## 🔍 Quality Filters

All scrapers implement quality filters:
- Minimum text length: 100 characters
- Remove duplicates
- Clean HTML/formatting
- Skip redirects and missing pages

## ⚠️ Rate Limiting

**Be respectful!** Default delays:
- Wikipedia: 1 second between requests
- News: 2 seconds between requests

Increase delay with `--delay` flag if needed.

## 🐛 Troubleshooting

**"No module named 'datasets'"**
```bash
pip install datasets
```

**"No module named 'bs4'"**
```bash
pip install beautifulsoup4
```

**"RSS feed has parsing errors"**
- Feed URL might be wrong or changed
- Try updating RSS URLs in `scrape_news.py`

**"HuggingFace dataset not found"**
- Check dataset name in `mcp_connectors.yml`
- Some datasets require authentication

