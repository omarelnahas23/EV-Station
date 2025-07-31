import json
import re
import yaml
from datasets import Dataset
import os
import logging
from typing import List, Dict, Any, Set

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_text(text):
    """Clean and normalize text data."""
    # Replace non-breaking spaces with regular spaces
    text = text.replace("\u00A0", " ")
    # Collapse runs of whitespace
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    # Replace multiple newlines with a single newline
    text = re.sub(r"\n{2,}", "\n", text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

def normalize_text(text: str) -> str:
    """Normalise text for training."""
    text = text.lower()
    # Remove extra spaces again
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def deduplicate_texts(texts) -> List[str]:
    """Remove duplicate text entries while preserving order."""
    seen: Set[str] = set()
    unique_texts: List[str] = []
    for txt in texts:
        cleaned = txt.strip()
        if not cleaned:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            unique_texts.append(cleaned)
    logger.info("Deduplicated %d texts down to %d", len(list(texts)), len(unique_texts))
    return unique_texts

def deduplicate_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate a list of record dictionaries by their JSON representation."""
    seen: Set[str] = set()
    unique_records: List[Dict[str, Any]] = []
    for rec in records:
        rec_key = str(rec)
        if rec_key not in seen:
            seen.add(rec_key)
            unique_records.append(rec)
    logger.info("Deduplicated %d records down to %d", len(records), len(unique_records))
    return unique_records

def filter_quality(data_item, min_length=50, max_length=5000):
    """Filter data based on quality criteria."""
    text = data_item['text']
    if len(text) < min_length or len(text) > max_length:
        return False
    # Check if text has meaningful content (not just numbers/symbols)
    if len(re.findall(r'\b[a-zA-Z]{3,}\b', text)) < 5:
        return False
    return True

def deduplicate_data(data):
    """Remove duplicate entries based on text similarity."""
    seen_texts = set()
    deduplicated = []
    
    for item in data:
        # Create a normalized version for comparison
        normalized = clean_text(item['text']).lower()
        # Use first 100 chars as fingerprint
        fingerprint = normalized[:100]
        
        if fingerprint not in seen_texts:
            seen_texts.add(fingerprint)
            deduplicated.append(item)
    
    return deduplicated

def chunk_text(text, chunk_size=512, overlap=50):
    """Split long text into smaller chunks with overlap."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:  # Only include meaningful chunks
            chunks.append(chunk)
    
    return chunks

def process_data(config):
    """Main data processing pipeline."""
    # Load raw data
    raw_data_path = os.path.join(config['data_dir'], 'raw_data.json')
    
    if not os.path.exists(raw_data_path):
        logger.error(f"Raw data file not found: {raw_data_path}")
        return []
        
    with open(raw_data_path, 'r') as f:
        raw_data = json.load(f)
    
    logger.info(f"Processing {len(raw_data)} raw data items")
    processed_data = []
    
    for item in raw_data:
        # Handle different data types
        if item.get('metadata', {}).get('type') == 'structured_data':
            # For structured data, use record deduplication
            if isinstance(item.get('content'), list):
                unique_records = deduplicate_records(item['content'])
                item['content'] = unique_records
        
        # Clean text
        cleaned_text = clean_text(item['text'])
        
        # Create updated item
        processed_item = {
            'text': cleaned_text,
            'metadata': item['metadata']
        }
        
        # Filter by quality
        if filter_quality(processed_item):
            # Chunk long texts
            chunks = chunk_text(cleaned_text)
            for i, chunk in enumerate(chunks):
                chunk_item = {
                    'text': chunk,
                    'metadata': {
                        **item['metadata'],
                        'chunk_id': i,
                        'total_chunks': len(chunks)
                    }
                }
                processed_data.append(chunk_item)
    
    # Deduplicate
    processed_data = deduplicate_data(processed_data)
    
    # Save processed data
    processed_path = os.path.join(config['data_dir'], 'processed_data.json')
    with open(processed_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    logger.info(f"Processed {len(processed_data)} data items")
    return processed_data

if __name__ == '__main__':
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    process_data(config) 