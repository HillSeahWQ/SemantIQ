"""
Embedding manager for handling different embedding models.
Provides unified interface for text and code embeddings.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from openai import OpenAI
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts into vectors.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of shape (num_texts, embedding_dim)
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass


class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embeddings wrapper."""
    
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        batch_size: int = 64,
        normalize: bool = True
    ):
        """
        Initialize OpenAI embedder.
        
        Args:
            model: OpenAI embedding model name
            batch_size: Batch size for API calls
            normalize: Whether to L2-normalize embeddings
        """
        self._model = model
        self.batch_size = batch_size
        self.normalize = normalize
        self.client = OpenAI()
        
        # Set dimensions based on model
        self.dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536
        }
        
        logger.info(f"Initialized OpenAI embedder: {model}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts using OpenAI API."""
        logger.info(f"Embedding {len(texts)} texts with {self._model}...")
        
        vectors = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding"):
            batch = texts[i:i + self.batch_size]
            
            response = self.client.embeddings.create(
                model=self._model,
                input=batch
            )
            
            batch_vectors = [d.embedding for d in response.data]
            batch_vectors = np.array(batch_vectors, dtype=np.float32)
            
            if self.normalize:
                norms = np.linalg.norm(batch_vectors, axis=1, keepdims=True)
                batch_vectors = batch_vectors / np.maximum(norms, 1e-12)
            
            vectors.append(batch_vectors)
        
        embeddings = np.vstack(vectors)
        logger.info(f"Finished embedding. Shape: {embeddings.shape}")
        return embeddings
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.dimensions.get(self._model, 1536)
    
    @property
    def model_name(self) -> str:
        return self._model


class SentenceTransformerEmbedder(BaseEmbedder):
    """Sentence Transformers embeddings wrapper."""
    
    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        normalize: bool = True
    ):
        """
        Initialize Sentence Transformer embedder.
        
        Args:
            model: Model name from sentence-transformers
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
        """
        self._model = model
        self.batch_size = batch_size
        self.normalize = normalize
        
        logger.info(f"Loading Sentence Transformer model: {model}")
        self.model = SentenceTransformer(model)
        logger.info(f"Model loaded successfully")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed texts using Sentence Transformers."""
        logger.info(f"Embedding {len(texts)} texts with {self._model}...")
        
        vectors = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding"):
            batch = texts[i:i + self.batch_size]
            vec = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize
            )
            vectors.append(vec)
        
        embeddings = np.vstack(vectors).astype("float32")
        logger.info(f"Finished embedding. Shape: {embeddings.shape}")
        return embeddings
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    @property
    def model_name(self) -> str:
        return self._model


class EmbeddingManager:
    """
    Factory class for creating embedders based on configuration.
    Supports easy extension for different embedding types (text, code, etc.).
    """
    
    @staticmethod
    def create_embedder(
        provider: str,
        embedding_type: str = "text",
        config: Dict[str, Any] = None
    ) -> BaseEmbedder:
        """
        Create an embedder based on provider and type.
        
        Args:
            provider: "openai" or "sentence_transformers"
            embedding_type: "text" or "code" (for future extension)
            config: Configuration dictionary with model settings
            
        Returns:
            BaseEmbedder instance
        """
        if config is None:
            config = {}
        if embedding_type == "text":
            if provider == "openai":
                return OpenAIEmbedder(
                    model=config.get("model", "text-embedding-3-large"),
                    batch_size=config.get("batch_size", 64),
                    normalize=config.get("normalize", True)
                )
            elif provider == "sentence_transformers":
                return SentenceTransformerEmbedder(
                    model=config.get("model", "sentence-transformers/all-MiniLM-L6-v2"),
                    batch_size=config.get("batch_size", 64),
                    normalize=config.get("normalize", True)
                )
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")
    
    @staticmethod
    def embed_chunks(chunks: List[str], embedder: BaseEmbedder) -> np.ndarray:
        """
        Convenience method to embed a list of chunk texts.
        
        Args:
            chunks: List of text strings
            embedder: BaseEmbedder instance
            
        Returns:
            numpy array of embeddings
        """
        return embedder.embed(chunks)