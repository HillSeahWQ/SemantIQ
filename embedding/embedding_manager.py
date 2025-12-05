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


class CodeEmbedder(BaseEmbedder):
    """
    Specialized embedder for code using code-specific models.
    Supports models trained on code repositories.
    """
    
    def __init__(
        self,
        model: str = "microsoft/codebert-base",
        batch_size: int = 32,
        normalize: bool = True,
        add_language_prefix: bool = True
    ):
        """
        Initialize Code embedder.
        
        Args:
            model: Model name (CodeBERT, GraphCodeBERT, etc.)
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
            add_language_prefix: Whether to prepend language name to code
        """
        self._model = model
        self.batch_size = batch_size
        self.normalize = normalize
        self.add_language_prefix = add_language_prefix
        
        logger.info(f"Loading Code Transformer model: {model}")
        self.model = SentenceTransformer(model)
        logger.info(f"Code model loaded successfully")
    
    def embed(self, texts: List[str], languages: List[str] = None) -> np.ndarray:
        """
        Embed code snippets.
        
        Args:
            texts: List of code strings to embed
            languages: Optional list of programming languages for each code snippet
            
        Returns:
            numpy array of embeddings
        """
        logger.info(f"Embedding {len(texts)} code snippets with {self._model}...")
        
        # Optionally prepend language tags for better context
        if self.add_language_prefix and languages:
            texts = [f"# Language: {lang}\n{code}" for lang, code in zip(languages, texts)]
        
        vectors = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding Code"):
            batch = texts[i:i + self.batch_size]
            vec = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize
            )
            vectors.append(vec)
        
        embeddings = np.vstack(vectors).astype("float32")
        logger.info(f"Finished embedding code. Shape: {embeddings.shape}")
        return embeddings
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    @property
    def model_name(self) -> str:
        return self._model


class OpenAICodeEmbedder(BaseEmbedder):
    """
    OpenAI embeddings optimized for code.
    Uses text-embedding models with code-specific preprocessing.
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        batch_size: int = 64,
        normalize: bool = True,
        add_language_prefix: bool = True
    ):
        """
        Initialize OpenAI code embedder.
        
        Args:
            model: OpenAI embedding model name
            batch_size: Batch size for API calls
            normalize: Whether to L2-normalize embeddings
            add_language_prefix: Whether to prepend language name to code
        """
        self._model = model
        self.batch_size = batch_size
        self.normalize = normalize
        self.add_language_prefix = add_language_prefix
        self.client = OpenAI()
        
        # Set dimensions based on model
        self.dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536
        }
        
        logger.info(f"Initialized OpenAI code embedder: {model}")
    
    def embed(self, texts: List[str], languages: List[str] = None) -> np.ndarray:
        """
        Embed code snippets using OpenAI API.
        
        Args:
            texts: List of code strings to embed
            languages: Optional list of programming languages for each code snippet
            
        Returns:
            numpy array of embeddings
        """
        logger.info(f"Embedding {len(texts)} code snippets with {self._model}...")
        
        # Optionally prepend language tags for better context
        if self.add_language_prefix and languages:
            texts = [f"# Language: {lang}\n{code}" for lang, code in zip(languages, texts)]
        
        vectors = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding Code"):
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
        logger.info(f"Finished embedding code. Shape: {embeddings.shape}")
        return embeddings
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return self.dimensions.get(self._model, 1536)
    
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
            provider: "openai", "sentence_transformers", or "code_transformers"
            embedding_type: "text" or "code"
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
                raise ValueError(f"Unknown text embedding provider: {provider}")
                
        elif embedding_type == "code":
            if provider == "openai":
                return OpenAICodeEmbedder(
                    model=config.get("model", "text-embedding-3-large"),
                    batch_size=config.get("batch_size", 64),
                    normalize=config.get("normalize", True),
                    add_language_prefix=config.get("add_language_prefix", True)
                )
            elif provider == "code_transformers":
                return CodeEmbedder(
                    model=config.get("model", "microsoft/codebert-base"),
                    batch_size=config.get("batch_size", 32),
                    normalize=config.get("normalize", True),
                    add_language_prefix=config.get("add_language_prefix", True)
                )
            else:
                raise ValueError(f"Unknown code embedding provider: {provider}")
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
    
    @staticmethod
    def embed_chunks(
        chunks: List[str], 
        embedder: BaseEmbedder,
        languages: List[str] = None
    ) -> np.ndarray:
        """
        Convenience method to embed a list of chunk texts.
        
        Args:
            chunks: List of text strings
            embedder: BaseEmbedder instance
            languages: Optional list of programming languages (for code embeddings)
            
        Returns:
            numpy array of embeddings
        """
        # Check if embedder supports language parameter
        if isinstance(embedder, (CodeEmbedder, OpenAICodeEmbedder)) and languages:
            return embedder.embed(chunks, languages=languages)
        else:
            return embedder.embed(chunks)