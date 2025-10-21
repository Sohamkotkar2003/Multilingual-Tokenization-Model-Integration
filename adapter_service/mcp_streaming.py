#!/usr/bin/env python3
"""
MCP (Multi-Cloud Protocol) Streaming Module

Provides unified interface for streaming data from multiple sources:
- HuggingFace Datasets (streaming mode)
- S3/Cloud Storage (boto3)
- HTTP APIs (REST endpoints)
- Qdrant Vector Database
- Local filesystem fallback

Design goals:
- No large local downloads (stream on-the-fly)
- Unified iterator interface
- Automatic retry and error handling
- Memory efficient (<100MB buffer)
"""

import os
import json
import time
import logging
from typing import Iterator, Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass

import yaml

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for streaming behavior"""
    batch_size: int = 100
    buffer_size: int = 1000
    timeout: int = 30
    retry_attempts: int = 3
    max_samples: Optional[int] = None


class MCPStreamingError(Exception):
    """Base exception for MCP streaming errors"""
    pass


class HuggingFaceStreamer:
    """Stream data from HuggingFace datasets"""
    
    def __init__(self, dataset_name: str, split: str = "train", 
                 config_name: Optional[str] = None,
                 config: Optional[StreamConfig] = None):
        self.dataset_name = dataset_name
        self.split = split
        self.config_name = config_name
        self.config = config or StreamConfig()
        
    def stream(self) -> Iterator[Dict[str, Any]]:
        """Stream samples from HuggingFace dataset"""
        try:
            from datasets import load_dataset
            
            logger.info(f"Streaming from HuggingFace: {self.dataset_name}")
            
            # Load in streaming mode (no download)
            dataset = load_dataset(
                self.dataset_name,
                name=self.config_name,
                split=self.split,
                streaming=True,
                trust_remote_code=False  # Security: don't execute remote scripts
            )
            
            count = 0
            for sample in dataset:
                # Normalize to expected format
                yield {
                    "text": sample.get("text", ""),
                    "language": sample.get("language", "unknown"),
                    "source": f"hf:{self.dataset_name}",
                    "metadata": sample
                }
                
                count += 1
                if self.config.max_samples and count >= self.config.max_samples:
                    break
                    
            logger.info(f"Streamed {count} samples from HuggingFace")
            
        except Exception as e:
            logger.error(f"HuggingFace streaming error: {e}")
            raise MCPStreamingError(f"Failed to stream from HuggingFace: {e}")


class S3Streamer:
    """Stream data from S3/Cloud Storage"""
    
    def __init__(self, bucket: str, prefix: str = "", 
                 region: str = "us-east-1",
                 config: Optional[StreamConfig] = None):
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self.config = config or StreamConfig()
        
    def stream(self) -> Iterator[Dict[str, Any]]:
        """Stream samples from S3 bucket"""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            logger.info(f"Streaming from S3: s3://{self.bucket}/{self.prefix}")
            
            # Initialize S3 client
            s3 = boto3.client('s3', region_name=self.region)
            
            # List objects in bucket
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)
            
            count = 0
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    
                    # Skip non-text files
                    if not key.endswith(('.txt', '.json', '.jsonl')):
                        continue
                    
                    # Download and parse file
                    try:
                        response = s3.get_object(Bucket=self.bucket, Key=key)
                        content = response['Body'].read().decode('utf-8')
                        
                        # Parse based on file type
                        if key.endswith('.jsonl'):
                            for line in content.strip().split('\n'):
                                if line:
                                    data = json.loads(line)
                                    yield {
                                        "text": data.get("text", ""),
                                        "language": data.get("language", "unknown"),
                                        "source": f"s3://{self.bucket}/{key}",
                                        "metadata": data
                                    }
                                    count += 1
                                    if self.config.max_samples and count >= self.config.max_samples:
                                        return
                        else:
                            # Plain text
                            yield {
                                "text": content,
                                "language": "unknown",
                                "source": f"s3://{self.bucket}/{key}",
                                "metadata": {"key": key, "size": len(content)}
                            }
                            count += 1
                            if self.config.max_samples and count >= self.config.max_samples:
                                return
                                
                    except ClientError as e:
                        logger.warning(f"Failed to read s3://{self.bucket}/{key}: {e}")
                        continue
                        
            logger.info(f"Streamed {count} samples from S3")
            
        except ImportError:
            raise MCPStreamingError("boto3 not installed. Run: pip install boto3")
        except Exception as e:
            logger.error(f"S3 streaming error: {e}")
            raise MCPStreamingError(f"Failed to stream from S3: {e}")


class HTTPStreamer:
    """Stream data from HTTP API endpoints"""
    
    def __init__(self, base_url: str, endpoint: str = "/stream",
                 headers: Optional[Dict[str, str]] = None,
                 config: Optional[StreamConfig] = None):
        self.base_url = base_url.rstrip('/')
        self.endpoint = endpoint
        self.headers = headers or {}
        self.config = config or StreamConfig()
        
    def stream(self) -> Iterator[Dict[str, Any]]:
        """Stream samples from HTTP API"""
        try:
            import requests
            
            url = f"{self.base_url}{self.endpoint}"
            logger.info(f"Streaming from HTTP: {url}")
            
            # Request streaming data
            params = {
                "batch_size": self.config.batch_size,
                "max_samples": self.config.max_samples or 10000
            }
            
            response = requests.get(
                url,
                headers=self.headers,
                params=params,
                stream=True,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            count = 0
            buffer = ""
            
            # Stream response line by line
            for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                if not chunk:
                    continue
                    
                buffer += chunk
                lines = buffer.split('\n')
                buffer = lines[-1]  # Keep incomplete line
                
                for line in lines[:-1]:
                    if not line.strip():
                        continue
                        
                    try:
                        data = json.loads(line)
                        yield {
                            "text": data.get("text", ""),
                            "language": data.get("language", "unknown"),
                            "source": f"http:{url}",
                            "metadata": data
                        }
                        count += 1
                        
                        if self.config.max_samples and count >= self.config.max_samples:
                            return
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse JSON line: {line[:100]}")
                        continue
                        
            logger.info(f"Streamed {count} samples from HTTP")
            
        except Exception as e:
            logger.error(f"HTTP streaming error: {e}")
            raise MCPStreamingError(f"Failed to stream from HTTP: {e}")


class QdrantStreamer:
    """Stream data from Qdrant vector database"""
    
    def __init__(self, host: str = "localhost", port: int = 6333,
                 collection: str = "multilingual_kb",
                 config: Optional[StreamConfig] = None):
        self.host = host
        self.port = port
        self.collection = collection
        self.config = config or StreamConfig()
        
    def stream(self) -> Iterator[Dict[str, Any]]:
        """Stream samples from Qdrant collection"""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import ScrollRequest
            
            logger.info(f"Streaming from Qdrant: {self.host}:{self.port}/{self.collection}")
            
            # Initialize Qdrant client
            client = QdrantClient(host=self.host, port=self.port)
            
            # Scroll through collection (streaming)
            offset = None
            count = 0
            batch_size = min(self.config.batch_size, 100)
            
            while True:
                # Fetch batch
                results = client.scroll(
                    collection_name=self.collection,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False  # Don't need vectors for training
                )
                
                points, next_offset = results
                
                if not points:
                    break
                    
                # Yield samples
                for point in points:
                    payload = point.payload or {}
                    yield {
                        "text": payload.get("text", ""),
                        "language": payload.get("language", "unknown"),
                        "source": f"qdrant:{self.collection}",
                        "metadata": {
                            "id": point.id,
                            "payload": payload
                        }
                    }
                    count += 1
                    
                    if self.config.max_samples and count >= self.config.max_samples:
                        return
                        
                # Update offset for next batch
                offset = next_offset
                if offset is None:
                    break
                    
            logger.info(f"Streamed {count} samples from Qdrant")
            
        except ImportError:
            raise MCPStreamingError("qdrant-client not installed. Run: pip install qdrant-client")
        except Exception as e:
            logger.error(f"Qdrant streaming error: {e}")
            raise MCPStreamingError(f"Failed to stream from Qdrant: {e}")


class LocalFileStreamer:
    """Fallback: Stream data from local filesystem"""
    
    def __init__(self, path: str, config: Optional[StreamConfig] = None):
        self.path = Path(path)
        self.config = config or StreamConfig()
        
    def stream(self) -> Iterator[Dict[str, Any]]:
        """Stream samples from local files"""
        try:
            logger.info(f"Streaming from local: {self.path}")
            
            count = 0
            
            if self.path.is_file():
                # Single file
                files = [self.path]
            elif self.path.is_dir():
                # All text files in directory
                files = list(self.path.glob("**/*.txt")) + list(self.path.glob("**/*.jsonl"))
            else:
                raise MCPStreamingError(f"Path not found: {self.path}")
                
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if file_path.suffix == '.jsonl':
                            # JSONL format
                            for line in f:
                                if line.strip():
                                    data = json.loads(line)
                                    yield {
                                        "text": data.get("text", ""),
                                        "language": data.get("language", "unknown"),
                                        "source": f"local:{file_path}",
                                        "metadata": data
                                    }
                                    count += 1
                                    if self.config.max_samples and count >= self.config.max_samples:
                                        return
                        else:
                            # Plain text
                            content = f.read()
                            if content.strip():
                                yield {
                                    "text": content,
                                    "language": "unknown",
                                    "source": f"local:{file_path}",
                                    "metadata": {"file": str(file_path)}
                                }
                                count += 1
                                if self.config.max_samples and count >= self.config.max_samples:
                                    return
                                    
                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")
                    continue
                    
            logger.info(f"Streamed {count} samples from local files")
            
        except Exception as e:
            logger.error(f"Local streaming error: {e}")
            raise MCPStreamingError(f"Failed to stream from local: {e}")


class MCPDataLoader:
    """
    Unified MCP data loader with fallback strategy
    
    Usage:
        loader = MCPDataLoader("mcp_connectors.yml")
        for sample in loader.stream("multilingual_corpus", max_samples=1000):
            print(sample["text"], sample["language"])
    """
    
    def __init__(self, config_path: str = "mcp_connectors.yml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load MCP connector configuration"""
        if not self.config_path.exists():
            logger.warning(f"Config not found: {self.config_path}, using defaults")
            return {}
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
            
    def stream(self, source_name: str, max_samples: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """
        Stream data from specified source with automatic fallback
        
        Args:
            source_name: Name of source in config (e.g., "multilingual_corpus")
            max_samples: Maximum number of samples to stream
            
        Yields:
            Dict with keys: text, language, source, metadata
        """
        stream_config = StreamConfig(
            batch_size=self.config.get('streaming_config', {}).get('batch_size', 100),
            buffer_size=self.config.get('streaming_config', {}).get('buffer_size', 1000),
            timeout=self.config.get('streaming_config', {}).get('timeout', 30),
            retry_attempts=self.config.get('streaming_config', {}).get('retry_attempts', 3),
            max_samples=max_samples
        )
        
        # Try each source type in order
        errors = []
        
        # 1. Try HuggingFace
        hf_sources = self.config.get('hf_sources', {})
        if source_name in hf_sources:
            try:
                source_cfg = hf_sources[source_name]
                streamer = HuggingFaceStreamer(
                    dataset_name=source_cfg['dataset_name'],
                    split=source_cfg.get('split', 'train'),
                    config_name=source_cfg.get('config_name'),
                    config=stream_config
                )
                yield from streamer.stream()
                return  # Success!
            except Exception as e:
                errors.append(f"HuggingFace: {e}")
                logger.warning(f"HuggingFace streaming failed: {e}")
                
        # 2. Try S3
        s3_sources = self.config.get('s3_sources', {})
        if source_name in s3_sources:
            try:
                source_cfg = s3_sources[source_name]
                streamer = S3Streamer(
                    bucket=source_cfg['bucket'],
                    prefix=source_cfg.get('prefix', ''),
                    region=source_cfg.get('region', 'us-east-1'),
                    config=stream_config
                )
                yield from streamer.stream()
                return  # Success!
            except Exception as e:
                errors.append(f"S3: {e}")
                logger.warning(f"S3 streaming failed: {e}")
                
        # 3. Try HTTP
        http_sources = self.config.get('http_sources', {})
        if source_name in http_sources:
            try:
                source_cfg = http_sources[source_name]
                streamer = HTTPStreamer(
                    base_url=source_cfg['base_url'],
                    endpoint=source_cfg.get('endpoint', '/stream'),
                    headers=source_cfg.get('headers', {}),
                    config=stream_config
                )
                yield from streamer.stream()
                return  # Success!
            except Exception as e:
                errors.append(f"HTTP: {e}")
                logger.warning(f"HTTP streaming failed: {e}")
                
        # 4. Try Qdrant
        qdrant_sources = self.config.get('qdrant_sources', {})
        if source_name in qdrant_sources:
            try:
                source_cfg = qdrant_sources[source_name]
                streamer = QdrantStreamer(
                    host=source_cfg.get('host', 'localhost'),
                    port=source_cfg.get('port', 6333),
                    collection=source_cfg['collection'],
                    config=stream_config
                )
                yield from streamer.stream()
                return  # Success!
            except Exception as e:
                errors.append(f"Qdrant: {e}")
                logger.warning(f"Qdrant streaming failed: {e}")
                
        # 5. Fallback to local files
        logger.warning(f"All remote sources failed for '{source_name}', trying local fallback")
        local_paths = [
            Path("data/training"),
            Path("data/validation"),
            Path("cache"),
        ]
        
        for local_path in local_paths:
            if local_path.exists():
                try:
                    streamer = LocalFileStreamer(str(local_path), config=stream_config)
                    yield from streamer.stream()
                    return  # Success!
                except Exception as e:
                    errors.append(f"Local: {e}")
                    logger.warning(f"Local fallback failed: {e}")
                    
        # All sources failed
        error_msg = f"All streaming sources failed for '{source_name}':\n" + "\n".join(errors)
        raise MCPStreamingError(error_msg)
        
    def list_sources(self) -> List[str]:
        """List all available data sources"""
        sources = []
        sources.extend(self.config.get('hf_sources', {}).keys())
        sources.extend(self.config.get('s3_sources', {}).keys())
        sources.extend(self.config.get('http_sources', {}).keys())
        sources.extend(self.config.get('qdrant_sources', {}).keys())
        return sources


# Convenience function for quick usage
def stream_data(source_name: str = "multilingual_corpus", 
                max_samples: Optional[int] = None,
                config_path: str = "mcp_connectors.yml") -> Iterator[Dict[str, Any]]:
    """
    Quick streaming function
    
    Example:
        for sample in stream_data("multilingual_corpus", max_samples=100):
            print(sample["text"])
    """
    loader = MCPDataLoader(config_path)
    return loader.stream(source_name, max_samples)


if __name__ == "__main__":
    # Test streaming
    import sys
    import io
    
    # Fix Windows console encoding
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    logging.basicConfig(level=logging.INFO)
    
    source = sys.argv[1] if len(sys.argv) > 1 else "multilingual_corpus"
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"\n=== Testing MCP Streaming: {source} (max {max_samples} samples) ===\n")
    
    try:
        loader = MCPDataLoader()
        print(f"Available sources: {loader.list_sources()}\n")
        
        for i, sample in enumerate(loader.stream(source, max_samples=max_samples), 1):
            print(f"{i}. [{sample['language']}] from {sample['source']}")
            print(f"   {sample['text'][:100]}...")
            print()
            
    except MCPStreamingError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

