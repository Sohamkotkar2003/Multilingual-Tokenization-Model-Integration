#!/usr/bin/env python3
"""
Training Script with MCP Streaming Integration

This is an EXAMPLE of how to integrate MCP streaming with training.
NOTE: Current adapter training has issues, so this is for future reference.

Usage:
    python adapter_service/train_with_mcp.py --source multilingual_corpus --max_samples 5000
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Iterator, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mcp_streaming import MCPDataLoader, MCPStreamingError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def stream_training_data(source: str, max_samples: int = 5000) -> Iterator[Dict[str, Any]]:
    """
    Stream training data from MCP sources with automatic fallback
    
    Args:
        source: Source name from mcp_connectors.yml
        max_samples: Maximum samples to stream
        
    Yields:
        Dict with keys: text, language, source, metadata
    """
    loader = MCPDataLoader("mcp_connectors.yml")
    
    logger.info(f"Streaming training data from: {source}")
    logger.info(f"Max samples: {max_samples}")
    
    try:
        for sample in loader.stream(source, max_samples=max_samples):
            yield sample
    except MCPStreamingError as e:
        logger.error(f"Streaming failed: {e}")
        raise


def prepare_training_batch(samples: list, batch_size: int = 8):
    """
    Prepare batch for training (example)
    
    Args:
        samples: List of sample dicts
        batch_size: Batch size
        
    Returns:
        Batched data ready for training
    """
    texts = [s["text"] for s in samples[:batch_size]]
    languages = [s["language"] for s in samples[:batch_size]]
    
    return {
        "texts": texts,
        "languages": languages,
        "batch_size": len(texts)
    }


def train_with_streaming(source: str, max_samples: int, batch_size: int = 8):
    """
    Example training loop with MCP streaming
    
    NOTE: This is a DEMONSTRATION only. Actual training would use
    transformers Trainer or custom training loop.
    """
    logger.info("="*60)
    logger.info("MCP STREAMING TRAINING EXAMPLE")
    logger.info("="*60)
    logger.info(f"Source: {source}")
    logger.info(f"Max samples: {max_samples}")
    logger.info(f"Batch size: {batch_size}")
    
    # Stream data
    sample_buffer = []
    total_samples = 0
    total_batches = 0
    
    for sample in stream_training_data(source, max_samples):
        sample_buffer.append(sample)
        total_samples += 1
        
        # When buffer is full, create batch
        if len(sample_buffer) >= batch_size:
            batch = prepare_training_batch(sample_buffer, batch_size)
            
            # TRAINING STEP WOULD GO HERE
            # Example: loss = train_step(model, batch)
            
            total_batches += 1
            
            # Log progress
            if total_batches % 10 == 0:
                logger.info(f"Processed {total_samples} samples, {total_batches} batches")
            
            # Clear buffer
            sample_buffer = sample_buffer[batch_size:]
    
    # Process remaining samples
    if sample_buffer:
        batch = prepare_training_batch(sample_buffer, len(sample_buffer))
        total_batches += 1
    
    logger.info("="*60)
    logger.info("STREAMING TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Total batches: {total_batches}")
    logger.info(f"Avg batch size: {total_samples / max(1, total_batches):.1f}")


def demo_streaming_sources():
    """Demonstrate all available streaming sources"""
    logger.info("\n" + "="*60)
    logger.info("AVAILABLE MCP STREAMING SOURCES")
    logger.info("="*60)
    
    loader = MCPDataLoader("mcp_connectors.yml")
    sources = loader.list_sources()
    
    logger.info(f"\nFound {len(sources)} sources:")
    for i, source in enumerate(sources, 1):
        logger.info(f"  {i}. {source}")
    
    # Test streaming from first source
    if sources:
        test_source = sources[0]
        logger.info(f"\nTesting stream from: {test_source}")
        
        count = 0
        for sample in loader.stream(test_source, max_samples=5):
            count += 1
            lang = sample.get("language", "unknown")
            text = sample.get("text", "")[:60]
            logger.info(f"  {count}. [{lang}] {text}...")
        
        logger.info(f"\nSuccessfully streamed {count} samples!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train with MCP streaming")
    parser.add_argument("--source", default="multilingual_corpus",
                       help="Source name from mcp_connectors.yml")
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum samples to stream")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--demo", action="store_true",
                       help="Demo all sources instead of training")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_streaming_sources()
    else:
        train_with_streaming(args.source, args.max_samples, args.batch_size)


if __name__ == "__main__":
    main()

