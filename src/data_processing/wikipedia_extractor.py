"""
Wikipedia XML Dump Extractor for Multilingual Corpora

This module extracts clean text from Wikipedia XML dumps (.xml.bz2 files)
and converts them to training-ready text files for multilingual tokenizer training.

Features:
- Handles compressed XML dumps (.xml.bz2)
- Extracts article text content
- Removes Wikipedia markup and metadata
- Language-specific text cleaning
- Batch processing for multiple languages
"""

import os
import bz2
import xml.etree.ElementTree as ET
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Iterator
import tempfile
import shutil

# Import settings
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WikipediaExtractor:
    """
    Extracts clean text from Wikipedia XML dumps
    
    This class handles the complete extraction pipeline:
    1. Decompress .xml.bz2 files
    2. Parse XML structure
    3. Extract article content
    4. Clean and normalize text
    5. Save as training-ready text files
    """
    
    def __init__(self):
        self.stats = {
            "total_articles": 0,
            "total_text_length": 0,
            "languages_processed": set(),
            "errors": []
        }
        
        # Wikipedia XML namespaces
        self.namespaces = {
            'ns': 'http://www.mediawiki.org/xml/export-0.10/'
        }
        
        # Common Wikipedia markup patterns to remove
        self.markup_patterns = [
            r'<ref[^>]*>.*?</ref>',  # References
            r'<ref[^>]*/>',         # Self-closing references
            r'<gallery[^>]*>.*?</gallery>',  # Galleries
            r'<math[^>]*>.*?</math>',  # Math formulas
            r'<chem[^>]*>.*?</chem>',  # Chemistry formulas
            r'<code[^>]*>.*?</code>',  # Code blocks
            r'<pre[^>]*>.*?</pre>',   # Preformatted text
            r'<nowiki[^>]*>.*?</nowiki>',  # Nowiki blocks
            r'<syntaxhighlight[^>]*>.*?</syntaxhighlight>',  # Syntax highlighting
            r'<source[^>]*>.*?</source>',  # Source code
            r'<blockquote[^>]*>.*?</blockquote>',  # Blockquotes
            r'<div[^>]*>.*?</div>',  # Divs
            r'<span[^>]*>.*?</span>',  # Spans
            r'<table[^>]*>.*?</table>',  # Tables
            r'<tr[^>]*>.*?</tr>',    # Table rows
            r'<td[^>]*>.*?</td>',    # Table cells
            r'<th[^>]*>.*?</th>',    # Table headers
            r'<ul[^>]*>.*?</ul>',    # Unordered lists
            r'<ol[^>]*>.*?</ol>',    # Ordered lists
            r'<li[^>]*>.*?</li>',    # List items
            r'<dl[^>]*>.*?</dl>',    # Definition lists
            r'<dt[^>]*>.*?</dt>',    # Definition terms
            r'<dd[^>]*>.*?</dd>',    # Definition descriptions
            r'<p[^>]*>.*?</p>',      # Paragraphs
            r'<br[^>]*/?>',          # Line breaks
            r'<hr[^>]*/?>',          # Horizontal rules
            r'<h[1-6][^>]*>.*?</h[1-6]>',  # Headings
            r'<a[^>]*>([^<]*)</a>',  # Links (keep text)
            r'<img[^>]*/?>',         # Images
            r'<file[^>]*>.*?</file>',  # File links
            r'<category[^>]*>.*?</category>',  # Categories
            r'<includeonly>.*?</includeonly>',  # Includeonly
            r'<noinclude>.*?</noinclude>',  # Noinclude
            r'<onlyinclude>.*?</onlyinclude>',  # Onlyinclude
            r'<timeline[^>]*>.*?</timeline>',  # Timelines
            r'<poem[^>]*>.*?</poem>',  # Poems
            r'<score[^>]*>.*?</score>',  # Musical scores
            r'<graph[^>]*>.*?</graph>',  # Graphs
            r'<mapframe[^>]*>.*?</mapframe>',  # Map frames
            r'<maplink[^>]*>.*?</maplink>',  # Map links
            r'<templatedata[^>]*>.*?</templatedata>',  # Template data
            r'<translate[^>]*>.*?</translate>',  # Translate
            r'<languages[^>]*>.*?</languages>',  # Languages
            r'<indicator[^>]*>.*?</indicator>',  # Indicators
            r'<inputbox[^>]*>.*?</inputbox>',  # Input boxes
            r'<imagemap[^>]*>.*?</imagemap>',  # Image maps
            r'<hiero[^>]*>.*?</hiero>',  # Hieroglyphs
            r'<charinsert[^>]*>.*?</charinsert>',  # Character insert
            r'<ref name="[^"]*"[^>]*>.*?</ref>',  # Named references
            r'<ref name="[^"]*"[^>]*/>',  # Named self-closing references
            r'<ref group="[^"]*"[^>]*>.*?</ref>',  # Grouped references
            r'<ref group="[^"]*"[^>]*/>',  # Grouped self-closing references
        ]
        
        # Additional patterns for specific content removal
        self.content_patterns = [
            r'\[\[Category:[^\]]*\]\]',  # Categories
            r'\[\[File:[^\]]*\]\]',      # Files
            r'\[\[Image:[^\]]*\]\]',     # Images
            r'\[\[Media:[^\]]*\]\]',     # Media
            r'\[\[Template:[^\]]*\]\]',  # Templates
            r'\[\[User:[^\]]*\]\]',      # Users
            r'\[\[User talk:[^\]]*\]\]', # User talk
            r'\[\[Wikipedia:[^\]]*\]\]', # Wikipedia
            r'\[\[WP:[^\]]*\]\]',        # Wikipedia policies
            r'\[\[Help:[^\]]*\]\]',      # Help
            r'\[\[Special:[^\]]*\]\]',   # Special pages
            r'\[\[Talk:[^\]]*\]\]',      # Talk pages
            r'\[\[User talk:[^\]]*\]\]', # User talk
            r'\[\[Project:[^\]]*\]\]',   # Project pages
            r'\[\[Portal:[^\]]*\]\]',    # Portals
            r'\[\[Book:[^\]]*\]\]',      # Books
            r'\[\[Draft:[^\]]*\]\]',     # Drafts
            r'\[\[Education Program:[^\]]*\]\]',  # Education programs
            r'\[\[Grants:[^\]]*\]\]',    # Grants
            r'\[\[Incubator:[^\]]*\]\]', # Incubator
            r'\[\[Outreach:[^\]]*\]\]',  # Outreach
            r'\[\[Quality:[^\]]*\]\]',   # Quality
            r'\[\[Requested articles:[^\]]*\]\]', # Requested articles
            r'\[\[Village pump:[^\]]*\]\]', # Village pump
            r'\[\[Wikipedia:[^\]]*\]\]', # Wikipedia
            r'\[\[WP:[^\]]*\]\]',        # Wikipedia policies
            r'\[\[Help:[^\]]*\]\]',      # Help
            r'\[\[Special:[^\]]*\]\]',   # Special pages
            r'\[\[Talk:[^\]]*\]\]',      # Talk pages
            r'\[\[User talk:[^\]]*\]\]', # User talk
            r'\[\[Project:[^\]]*\]\]',   # Project pages
            r'\[\[Portal:[^\]]*\]\]',    # Portals
            r'\[\[Book:[^\]]*\]\]',      # Books
            r'\[\[Draft:[^\]]*\]\]',     # Drafts
            r'\[\[Education Program:[^\]]*\]\]',  # Education programs
            r'\[\[Grants:[^\]]*\]\]',    # Grants
            r'\[\[Incubator:[^\]]*\]\]', # Incubator
            r'\[\[Outreach:[^\]]*\]\]',  # Outreach
            r'\[\[Quality:[^\]]*\]\]',   # Quality
            r'\[\[Requested articles:[^\]]*\]\]', # Requested articles
            r'\[\[Village pump:[^\]]*\]\]', # Village pump
            r'\[\[[^\]]*\|([^\]]*)\]\]', # Links with text (keep text part)
            r'\[\[([^\]]*)\]\]',         # Simple links (keep text)
            r'\[\[[^\]]*\]\]',           # All other links
            r'{{[^}]*}}',                # Templates
            r'{{[^}]*\|[^}]*}}',        # Templates with parameters
            r'<ref[^>]*>.*?</ref>',      # References
            r'<ref[^>]*/>',              # Self-closing references
            r'<references[^>]*>.*?</references>',  # References section
            r'<references[^>]*/>',       # Self-closing references
            r'<ref name="[^"]*"[^>]*>.*?</ref>',  # Named references
            r'<ref name="[^"]*"[^>]*/>',  # Named self-closing references
            r'<ref group="[^"]*"[^>]*>.*?</ref>',  # Grouped references
            r'<ref group="[^"]*"[^>]*/>',  # Grouped self-closing references
            r'<ref name="[^"]*"[^>]*>.*?</ref>',  # Named references
            r'<ref name="[^"]*"[^>]*/>',  # Named self-closing references
            r'<ref group="[^"]*"[^>]*>.*?</ref>',  # Grouped references
            r'<ref group="[^"]*"[^>]*/>',  # Grouped self-closing references
        ]
    
    def clean_wikipedia_text(self, text: str) -> str:
        """
        Clean Wikipedia markup from text
        
        Args:
            text: Raw Wikipedia text with markup
            
        Returns:
            Cleaned text ready for training
        """
        if not text:
            return ""
        
        # Remove all markup patterns
        for pattern in self.markup_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove content patterns
        for pattern in self.content_patterns:
            text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove very short lines (likely artifacts)
        if len(text) < 10:
            return ""
        
        return text
    
    def extract_article_text(self, article_xml: str) -> str:
        """
        Extract clean text from a single article XML
        
        Args:
            article_xml: XML content of a single article
            
        Returns:
            Cleaned article text
        """
        try:
            # Parse the XML
            root = ET.fromstring(article_xml)
            
            # Find the text content
            text_elem = root.find('.//ns:revision/ns:text', self.namespaces)
            if text_elem is not None and text_elem.text:
                # Clean the Wikipedia markup
                cleaned_text = self.clean_wikipedia_text(text_elem.text)
                return cleaned_text
            
        except ET.ParseError as e:
            logger.warning(f"XML parsing error: {e}")
        except Exception as e:
            logger.warning(f"Error extracting article text: {e}")
        
        return ""
    
    def process_wikipedia_dump(self, xml_file_path: str, output_file_path: str, 
                              max_articles: int = 10000) -> Dict[str, int]:
        """
        Process a Wikipedia XML dump file
        
        Args:
            xml_file_path: Path to the .xml.bz2 file
            output_file_path: Path to save extracted text
            max_articles: Maximum number of articles to process
            
        Returns:
            Processing statistics
        """
        logger.info(f"Processing Wikipedia dump: {xml_file_path}")
        
        stats = {
            "articles_processed": 0,
            "articles_extracted": 0,
            "total_text_length": 0,
            "errors": 0
        }
        
        try:
            # Open the compressed file
            with bz2.open(xml_file_path, 'rt', encoding='utf-8') as f:
                # Read the file in chunks to handle large files
                buffer = ""
                article_buffer = ""
                in_article = False
                
                for line in f:
                    buffer += line
                    
                    # Look for article start
                    if '<ns:page>' in line:
                        in_article = True
                        article_buffer = line
                    elif in_article:
                        article_buffer += line
                        
                        # Look for article end
                        if '</ns:page>' in line:
                            in_article = False
                            
                            # Process the complete article
                            if stats["articles_processed"] < max_articles:
                                article_text = self.extract_article_text(article_buffer)
                                
                                if article_text:
                                    # Write to output file
                                    with open(output_file_path, 'a', encoding='utf-8') as out_f:
                                        out_f.write(article_text + '\n')
                                    
                                    stats["articles_extracted"] += 1
                                    stats["total_text_length"] += len(article_text)
                                
                                stats["articles_processed"] += 1
                                
                                if stats["articles_processed"] % 1000 == 0:
                                    logger.info(f"Processed {stats['articles_processed']} articles, "
                                              f"extracted {stats['articles_extracted']} with text")
                            
                            article_buffer = ""
                            
                            # Stop if we've reached the limit
                            if stats["articles_processed"] >= max_articles:
                                break
                
        except Exception as e:
            logger.error(f"Error processing {xml_file_path}: {e}")
            stats["errors"] += 1
        
        logger.info(f"Completed processing {xml_file_path}")
        logger.info(f"Articles processed: {stats['articles_processed']}")
        logger.info(f"Articles with text extracted: {stats['articles_extracted']}")
        logger.info(f"Total text length: {stats['total_text_length']:,} characters")
        
        return stats
    
    def extract_all_wikipedia_dumps(self, max_articles_per_language: int = 10000) -> Dict[str, Dict[str, int]]:
        """
        Extract text from all Wikipedia dumps in the training data directory
        
        Args:
            max_articles_per_language: Maximum articles to process per language
            
        Returns:
            Statistics for each language
        """
        logger.info("Starting Wikipedia dump extraction for all languages")
        
        all_stats = {}
        
        # Find all .xml.bz2 files in the training data directory
        training_dir = Path(settings.TRAINING_DATA_PATH)
        xml_files = list(training_dir.glob("*.xml.bz2"))
        
        if not xml_files:
            logger.warning("No .xml.bz2 files found in training data directory")
            return all_stats
        
        for xml_file in xml_files:
            # Extract language code from filename
            lang_code = xml_file.stem.replace('wiki-latest-pages-articles', '')
            
            # Create output filename
            output_file = training_dir / f"{lang_code}_extracted.txt"
            
            # Process the dump
            logger.info(f"Processing {xml_file.name} -> {output_file.name}")
            stats = self.process_wikipedia_dump(
                str(xml_file), 
                str(output_file), 
                max_articles_per_language
            )
            
            all_stats[lang_code] = stats
            self.stats["languages_processed"].add(lang_code)
        
        return all_stats
    
    def print_extraction_summary(self, all_stats: Dict[str, Dict[str, int]]):
        """Print comprehensive extraction summary"""
        logger.info("\n" + "=" * 80)
        logger.info("WIKIPEDIA EXTRACTION SUMMARY")
        logger.info("=" * 80)
        
        total_articles = 0
        total_extracted = 0
        total_text_length = 0
        
        for lang, stats in all_stats.items():
            logger.info(f"\n{lang.upper()}:")
            logger.info(f"  Articles processed: {stats['articles_processed']:,}")
            logger.info(f"  Articles extracted: {stats['articles_extracted']:,}")
            logger.info(f"  Text length: {stats['total_text_length']:,} characters")
            logger.info(f"  Success rate: {(stats['articles_extracted']/stats['articles_processed']*100):.1f}%" if stats['articles_processed'] > 0 else "  Success rate: 0%")
            
            total_articles += stats['articles_processed']
            total_extracted += stats['articles_extracted']
            total_text_length += stats['total_text_length']
        
        logger.info(f"\nTOTAL:")
        logger.info(f"  Articles processed: {total_articles:,}")
        logger.info(f"  Articles extracted: {total_extracted:,}")
        logger.info(f"  Total text length: {total_text_length:,} characters")
        logger.info(f"  Overall success rate: {(total_extracted/total_articles*100):.1f}%" if total_articles > 0 else "  Overall success rate: 0%")
        logger.info("=" * 80)


def main():
    """Main function to run Wikipedia extraction"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract text from Wikipedia XML dumps")
    parser.add_argument("--max-articles", type=int, default=10000,
                       help="Maximum articles to process per language")
    parser.add_argument("--file", type=str,
                       help="Process specific .xml.bz2 file")
    parser.add_argument("--output", type=str,
                       help="Output file path (for single file processing)")
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = WikipediaExtractor()
    
    try:
        if args.file:
            # Process single file
            if not args.output:
                args.output = args.file.replace('.xml.bz2', '_extracted.txt')
            
            logger.info(f"Processing single file: {args.file}")
            stats = extractor.process_wikipedia_dump(args.file, args.output, args.max_articles)
            logger.info(f"Extraction completed. Output saved to: {args.output}")
        else:
            # Process all files
            all_stats = extractor.extract_all_wikipedia_dumps(args.max_articles)
            extractor.print_extraction_summary(all_stats)
        
        logger.info("\nWikipedia extraction completed!")
        logger.info("\nNext steps:")
        logger.info("1. Run MCP pipeline: python src/data_processing/mcp_pipeline.py")
        logger.info("2. Train tokenizer: python src/training/train_multilingual_tokenizer.py")
        logger.info("3. Fine-tune model: python src/training/fine_tune.py")
        
    except Exception as e:
        logger.error(f"Wikipedia extraction failed: {e}")
        raise


if __name__ == "__main__":
    main()
