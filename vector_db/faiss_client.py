"""
FAISS vector database client for ingestion and querying.
Handles indexing, metadata storage, and search operations.
"""
import json
import logging
import pickle
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from pathlib import Path

logger = logging.getLogger(__name__)


class FAISSClient:
    """Client for FAISS vector database operations."""
    
    def __init__(
        self,
        index_dir: str = "data/faiss_indices",
        index_name: str = "default"
    ):
        """
        Initialize FAISS client.
        
        Args:
            index_dir: Directory to store FAISS indices
            index_name: Name of the index
        """
        self.index_dir = Path(index_dir)
        self.index_name = index_name
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.index = None
        self.metadata = []
        self.contents = []
        self.dimension = None
        self._index_path = self.index_dir / f"{index_name}.index"
        self._metadata_path = self.index_dir / f"{index_name}_metadata.pkl"
        self._contents_path = self.index_dir / f"{index_name}_contents.pkl"
    
    def create_index(
        self,
        embedding_dim: int,
        index_type: str = "Flat",
        metric_type: str = "IP",
        index_params: Optional[Dict[str, Any]] = None,
        drop_existing: bool = False
    ):
        """
        Create a FAISS index.
        
        Args:
            embedding_dim: Dimension of embedding vectors
            index_type: Type of FAISS index (Flat, IVF, HNSW)
            metric_type: Similarity metric (IP for inner product, L2 for euclidean)
            index_params: Index-specific parameters
            drop_existing: If True, overwrite existing index
        """
        if drop_existing or not self._index_exists():
            logger.info(f"Creating new FAISS index: {self.index_name}")
            self.dimension = embedding_dim
            
            # Create appropriate index based on type
            if index_type == "Flat":
                if metric_type == "IP":
                    self.index = faiss.IndexFlatIP(embedding_dim)
                else:  # L2
                    self.index = faiss.IndexFlatL2(embedding_dim)
                logger.info(f"Created Flat index with {metric_type} metric")
            
            elif index_type == "IVF":
                # IVF index for faster search on large datasets
                params = index_params or {}
                nlist = params.get("nlist", 100)
                
                if metric_type == "IP":
                    quantizer = faiss.IndexFlatIP(embedding_dim)
                    self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_INNER_PRODUCT)
                else:
                    quantizer = faiss.IndexFlatL2(embedding_dim)
                    self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist, faiss.METRIC_L2)
                
                logger.info(f"Created IVF index with {nlist} clusters and {metric_type} metric")
            
            elif index_type == "HNSW":
                # HNSW index for high-performance search
                params = index_params or {}
                m = params.get("M", 32)
                
                self.index = faiss.IndexHNSWFlat(embedding_dim, m)
                if metric_type == "IP":
                    self.index.metric_type = faiss.METRIC_INNER_PRODUCT
                else:
                    self.index.metric_type = faiss.METRIC_L2
                
                logger.info(f"Created HNSW index with M={m} and {metric_type} metric")
            
            else:
                raise ValueError(f"Unsupported index type: {index_type}")
            
            self.metadata = []
            self.contents = []
        else:
            logger.info(f"Loading existing index: {self.index_name}")
            self.load_index()
    
    def ingest_data(
        self,
        embeddings: np.ndarray,
        contents: List[str],
        metadatas: List[Dict[str, Any]],
        normalize: bool = False
    ):
        """
        Ingest data into FAISS index.
        
        Args:
            embeddings: numpy array of embeddings (n_samples, dimension)
            contents: List of content strings
            metadatas: List of metadata dictionaries
            normalize: If True, normalize embeddings to unit length (for cosine similarity)
        """
        if self.index is None:
            raise ValueError("Index not created. Call create_index() first.")
        
        logger.info(f"Preparing {len(metadatas)} records for insertion...")
        
        # Convert to float32 (FAISS requirement)
        embeddings = embeddings.astype('float32')
        
        # Normalize if requested (for cosine similarity with IP metric)
        if normalize:
            faiss.normalize_L2(embeddings)
            logger.info("Embeddings normalized for cosine similarity")
        
        # Train IVF index if needed
        if isinstance(self.index, faiss.IndexIVF) and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
            logger.info("IVF index trained")
        
        # Add vectors to index
        logger.info("Adding vectors to FAISS index...")
        self.index.add(embeddings)
        
        # Store metadata and contents
        self.contents.extend(contents)
        
        # Add ID to each metadata entry
        start_id = len(self.metadata)
        for i, metadata in enumerate(metadatas):
            metadata_with_id = {"id": start_id + i, **metadata}
            self.metadata.append(metadata_with_id)
        
        logger.info(f"Data ingestion complete. Total vectors: {self.index.ntotal}")
        
        # Save index
        self.save_index()
    
    def search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5,
        normalize: bool = False,
        search_params: Optional[Dict] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for similar vectors in FAISS.
        
        Args:
            query_embeddings: numpy array of query embeddings
            top_k: Number of results to return
            normalize: If True, normalize query embeddings
            search_params: Search parameters (e.g., nprobe for IVF)
            output_fields: Fields to return in results (None = all fields)
            
        Returns:
            List of results per query
        """
        if self.index is None:
            logger.info("No index in memory, loading from disk...")
            self.load_index()
        
        if self.index is None:
            raise ValueError("No index found. Please run ingestion first.")
        
        logger.info(f"Searching {len(query_embeddings)} queries (top_k={top_k})...")
        
        # Convert to float32
        query_embeddings = query_embeddings.astype('float32')
        
        # Normalize if requested
        if normalize:
            faiss.normalize_L2(query_embeddings)
        
        # Set search parameters for IVF index
        if isinstance(self.index, faiss.IndexIVF) and search_params:
            nprobe = search_params.get("nprobe", 10)
            self.index.nprobe = nprobe
            logger.info(f"Using nprobe={nprobe} for IVF search")
        
        # Perform search
        distances, indices = self.index.search(query_embeddings, top_k)
        
        # Format results
        all_results = []
        for query_idx in range(len(query_embeddings)):
            query_results = []
            for rank in range(top_k):
                idx = indices[query_idx][rank]
                distance = distances[query_idx][rank]
                
                # Skip invalid indices
                if idx == -1:
                    continue
                
                # Build result dictionary
                result_dict = {
                    "id": int(idx),
                    "score": float(distance),
                    "content": self.contents[idx] if idx < len(self.contents) else ""
                }
                
                # Add metadata fields
                if idx < len(self.metadata):
                    metadata = self.metadata[idx]
                    if output_fields:
                        for field in output_fields:
                            if field in metadata:
                                result_dict[field] = metadata[field]
                    else:
                        result_dict.update(metadata)
                
                # Add preview if not in metadata
                if "preview" not in result_dict and "content" in result_dict:
                    result_dict["preview"] = result_dict["content"][:200]
                
                query_results.append(result_dict)
            
            all_results.append(query_results)
        
        logger.info("Search complete")
        return all_results
    
    def save_index(self):
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return
        
        logger.info(f"Saving index to {self._index_path}")
        faiss.write_index(self.index, str(self._index_path))
        
        logger.info(f"Saving metadata to {self._metadata_path}")
        with open(self._metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        logger.info(f"Saving contents to {self._contents_path}")
        with open(self._contents_path, 'wb') as f:
            pickle.dump(self.contents, f)
        
        logger.info("Index, metadata, and contents saved successfully")
    
    def load_index(self):
        """Load FAISS index and metadata from disk."""
        if not self._index_exists():
            logger.warning(f"No saved index found at {self._index_path}")
            return False
        
        logger.info(f"Loading index from {self._index_path}")
        self.index = faiss.read_index(str(self._index_path))
        self.dimension = self.index.d
        
        logger.info(f"Loading metadata from {self._metadata_path}")
        with open(self._metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        logger.info(f"Loading contents from {self._contents_path}")
        with open(self._contents_path, 'rb') as f:
            self.contents = pickle.load(f)
        
        logger.info(f"Index loaded: {self.index.ntotal} vectors, {len(self.metadata)} metadata entries")
        return True
    
    def _index_exists(self) -> bool:
        """Check if index files exist on disk."""
        return (
            self._index_path.exists() and 
            self._metadata_path.exists() and 
            self._contents_path.exists()
        )
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        if self.index is None:
            self.load_index()
        
        if self.index is None:
            return {"error": "No index found"}
        
        stats = {
            "name": self.index_name,
            "num_vectors": int(self.index.ntotal),
            "dimension": int(self.dimension or self.index.d),
            "num_metadata": len(self.metadata),
            "num_contents": len(self.contents),
            "index_type": type(self.index).__name__,
            "index_path": str(self._index_path)
        }
        
        return stats
    
    def delete_index(self):
        """Delete the index and associated files."""
        logger.info(f"Deleting index: {self.index_name}")
        
        if self._index_path.exists():
            self._index_path.unlink()
        if self._metadata_path.exists():
            self._metadata_path.unlink()
        if self._contents_path.exists():
            self._contents_path.unlink()
        
        self.index = None
        self.metadata = []
        self.contents = []
        self.dimension = None
        
        logger.info("Index deleted")