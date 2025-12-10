"""
Chunking module for processing various document types.
"""
from .base import BaseChunker, Chunk, ChunkMetadata, ChunkType
from .pdf_chunker import MultimodalPDFChunker, PDFChunkMetadata
from .word_doc_chunker import WordDocumentChunker, WordChunkMetadata
from .code_chunker import CodeChunker, CodeChunkMetadata

__all__ = [
    'BaseChunker',
    'Chunk',
    'ChunkMetadata',
    'ChunkType',
    'MultimodalPDFChunker',
    'PDFChunkMetadata',
    'WordDocumentChunker',
    'WordChunkMetadata',
    'CodeChunker',
    'CodeChunkMetadata'
]