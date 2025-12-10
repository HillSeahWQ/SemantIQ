"""
Base classes for document chunking.
Provides abstract interface for extensibility.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum
from pathlib import Path


class ChunkType(Enum):
    """Types of chunks produced by chunkers."""
    TEXT = "text"
    TABLE = "table"
    IMAGE_HEAVY_PAGE = "image_heavy_page"
    MIXED = "mixed"
    CODE = "code"  # For future code chunking


@dataclass
class ChunkMetadata:
    """Base metadata class that all chunkers should use/extend."""
    source_file: str
    chunk_id: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        result = {"source_file": str(self.source_file), "chunk_id": self.chunk_id}
        
        # Add all other attributes
        for key, value in self.__dict__.items():
            if key not in result:
                if isinstance(value, Enum):
                    result[key] = value.value
                elif isinstance(value, Path):
                    result[key] = str(value)
                else:
                    result[key] = value
        
        return result


@dataclass
class Chunk:
    """Represents a processed document chunk."""
    content: str
    metadata: ChunkMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary format."""
        return {
            "content": self.content,
            "metadata": self.metadata.to_dict()
        }


class BaseChunker(ABC):
    """
    Abstract base class for document chunkers.
    All chunkers should inherit from this class.
    """
    
    @abstractmethod
    def chunk(self, file_path: str | Path) -> List[Chunk]:
        """
        Process a file and generate chunks.
        
        Args:
            file_path: Path to the file to chunk
            
        Returns:
            List of Chunk objects with content and metadata
        """
        pass
    
    @abstractmethod
    def get_metadata_schema(self) -> Dict[str, type]:
        """
        Return the metadata schema for this chunker.
        Used for automatic vector DB schema generation.
        
        Returns:
            Dictionary mapping field names to their types
        """
        pass
    
    def chunk_directory(self, directory: str | Path, extensions: List[str]) -> List[Chunk]:
        """
        Chunk all files in a directory with specified extensions.
        
        Args:
            directory: Path to directory
            extensions: List of file extensions to process (e.g., [".pdf", ".txt"])
            
        Returns:
            List of all chunks from all files
        """
        directory = Path(directory)
        all_chunks = []
        
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in extensions:
                chunks = self.chunk(path)
                all_chunks.extend(chunks)
        
        return all_chunks