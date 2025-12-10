"""
Utilities for saving and loading chunks.
"""
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def save_chunks(chunks: List[Any], output_path: Path | str):
    """
    Save chunks to JSON file.
    
    Args:
        chunks: List of Chunk objects
        output_path: Path to save JSON file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(chunks)} chunks to {output_path}...")
    
    chunk_dicts = [chunk.to_dict() for chunk in chunks]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunk_dicts, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Chunks saved successfully")


def load_chunks(input_path: Path | str) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    Load chunks from JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Tuple of (contents, metadatas)
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {input_path}")
    
    logger.info(f"Loading chunks from {input_path}...")
    
    with open(input_path, "r", encoding="utf-8") as f:
        chunk_dicts = json.load(f)
    
    contents = [chunk["content"] for chunk in chunk_dicts]
    metadatas = [chunk["metadata"] for chunk in chunk_dicts]
    
    logger.info(f"Loaded {len(contents)} chunks")
    
    return contents, metadatas


def get_chunk_statistics(metadatas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics from chunk metadata.
    
    Args:
        metadatas: List of metadata dictionaries
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        "total_chunks": len(metadatas),
        "chunk_types": {},
        "total_text_length": 0,
        "avg_text_length": 0,
        "total_tables": 0,
        "total_images": 0,
    }
    
    for metadata in metadatas:
        # Count chunk types
        chunk_type = metadata.get("chunk_type", "unknown")
        stats["chunk_types"][chunk_type] = stats["chunk_types"].get(chunk_type, 0) + 1
        
        # Sum text lengths
        text_length = metadata.get("text_length", 0)
        stats["total_text_length"] += text_length
        
        # Count tables and images
        stats["total_tables"] += metadata.get("num_tables", 0)
        stats["total_images"] += metadata.get("num_images", 0)
    
    # Calculate average
    if stats["total_chunks"] > 0:
        stats["avg_text_length"] = stats["total_text_length"] / stats["total_chunks"]
    
    return stats


def print_chunk_statistics(metadatas: List[Dict[str, Any]]):
    """Print formatted chunk statistics."""
    stats = get_chunk_statistics(metadatas)
    
    logger.info("="*80)
    logger.info("CHUNK STATISTICS")
    logger.info("="*80)
    logger.info(f"Total chunks: {stats['total_chunks']}")
    logger.info(f"Total text length: {stats['total_text_length']:,} characters")
    logger.info(f"Average text length: {stats['avg_text_length']:.0f} characters")
    logger.info("")
    logger.info("Chunk type distribution:")
    for chunk_type, count in stats["chunk_types"].items():
        percentage = (count / stats["total_chunks"]) * 100
        logger.info(f"  • {chunk_type}: {count} ({percentage:.1f}%)")
    logger.info("")
    logger.info(f"Total tables detected: {stats['total_tables']}")
    logger.info(f"Total images detected: {stats['total_images']}")
    logger.info("="*80)