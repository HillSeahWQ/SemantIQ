"""
Milvus vector database client for ingestion and querying.
Handles schema generation, indexing, and search operations.
"""
import json
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from pymilvus import (
    connections, 
    FieldSchema, 
    CollectionSchema, 
    DataType, 
    Collection, 
    utility
)

logger = logging.getLogger(__name__)


class MilvusClient:
    """Client for Milvus vector database operations."""
    
    # Type mapping from Python types to Milvus DataTypes
    TYPE_MAPPING = {
        int: DataType.INT64,
        float: DataType.FLOAT,
        str: DataType.VARCHAR,
        bool: DataType.BOOL,
    }
    
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        alias: str = "default"
    ):
        """
        Initialize Milvus client.
        
        Args:
            host: Milvus host address
            port: Milvus port
            alias: Connection alias
        """
        self.host = host
        self.port = port
        self.alias = alias
        self._connected = False
    
    def connect(self, reset: bool = False):
        """
        Establish connection to Milvus.
        
        Args:
            reset: If True, disconnect and reconnect
        """
        if reset:
            self.disconnect()
        
        if not any(c[0] == self.alias and c[1] for c in connections.list_connections()):
            connections.connect(alias=self.alias, host=self.host, port=self.port)
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            self._connected = True
        else:
            logger.info(f"Already connected to Milvus")
            self._connected = True
    
    def disconnect(self):
        """Disconnect from Milvus."""
        if any(c[0] == self.alias and c[1] for c in connections.list_connections()):
            connections.disconnect(self.alias)
            logger.info("Disconnected from Milvus")
            self._connected = False
    
    def create_collection_from_schema(
        self,
        collection_name: str,
        metadata_schema: Dict[str, type],
        embedding_dim: int,
        description: str = "",
        drop_existing: bool = False
    ) -> Collection:
        """
        Create a Milvus collection with automatic schema generation.
        
        Args:
            collection_name: Name of the collection
            metadata_schema: Dictionary mapping field names to Python types
            embedding_dim: Dimension of embedding vectors
            description: Collection description
            drop_existing: If True, drop existing collection
            
        Returns:
            Collection object
        """
        if not self._connected:
            self.connect()
        
        # Drop existing if requested
        if drop_existing and utility.has_collection(collection_name):
            logger.info(f"Dropping existing collection: {collection_name}")
            utility.drop_collection(collection_name)
        
        # Check if collection exists
        if utility.has_collection(collection_name):
            logger.info(f"Using existing collection: {collection_name}")
            return Collection(collection_name)
        
        # Build schema
        logger.info(f"Creating collection: {collection_name}")
        fields = self._build_fields_from_schema(metadata_schema, embedding_dim)
        
        schema = CollectionSchema(
            fields=fields,
            description=description or f"Auto-generated schema for {collection_name}"
        )
        
        collection = Collection(name=collection_name, schema=schema)
        logger.info(f"Collection created with {len(fields)} fields")
        
        return collection
    
    def _build_fields_from_schema(
        self,
        metadata_schema: Dict[str, type],
        embedding_dim: int
    ) -> List[FieldSchema]:
        """
        Build Milvus field schemas from metadata schema.
        
        Args:
            metadata_schema: Dictionary mapping field names to types
            embedding_dim: Dimension of embedding vectors
            
        Returns:
            List of FieldSchema objects
        """
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="preview", dtype=DataType.VARCHAR, max_length=500),
        ]
        
        # Add metadata fields
        for field_name, field_type in metadata_schema.items():
            if field_name in ["id", "embedding", "content", "preview"]:
                continue  # Skip already added fields
            
            milvus_type = self.TYPE_MAPPING.get(field_type, DataType.VARCHAR)
            
            if milvus_type == DataType.VARCHAR:
                # Determine appropriate max_length
                max_length = 65535 if field_name in ["table_content", "image_details", "content"] else 500
                fields.append(
                    FieldSchema(name=field_name, dtype=milvus_type, max_length=max_length)
                )
            else:
                fields.append(
                    FieldSchema(name=field_name, dtype=milvus_type)
                )
        
        return fields
    
    def ingest_data(
        self,
        collection_name: str,
        embeddings: np.ndarray,
        contents: List[str],
        metadatas: List[Dict[str, Any]],
        index_config: Dict[str, Any]
    ):
        """
        Ingest data into Milvus collection.
        
        Args:
            collection_name: Name of the collection
            embeddings: numpy array of embeddings
            contents: List of content strings
            metadatas: List of metadata dictionaries
            index_config: Index configuration with metric_type, index_type, params
        """
        if not self._connected:
            self.connect()
        
        collection = Collection(collection_name)
        
        logger.info(f"Preparing {len(metadatas)} records for insertion...")
        
        # Prepare insert data
        insert_data = []
        for i in range(len(metadatas)):
            record = { # "id" filled by default
                "embedding": embeddings[i].tolist(),
                "content": contents[i],
                "preview": contents[i][:200],
            }
            
            # Add metadata fields
            for key, value in metadatas[i].items():
                # Convert lists/dicts to JSON strings for storage
                if isinstance(value, (list, dict)):
                    record[key] = json.dumps(value)
                else:
                    record[key] = value
            
            insert_data.append(record)
        
        # Insert data
        logger.info("Inserting data into Milvus...")
        collection.insert(insert_data)
        logger.info("Data insertion complete")
        
        # CRITICAL: Flush data to make it persistent and visible
        logger.info("Flushing data to disk...")
        collection.flush()
        logger.info("Data flushed and persisted")
        
        # Create index
        logger.info(f"Creating index (type: {index_config['index_type']}, metric: {index_config['metric_type']})...")
        index_params = {
            "metric_type": index_config["metric_type"],
            "index_type": index_config["index_type"],
            "params": index_config["params"]
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info("Index created")
        
        # Load collection
        collection.load()
        logger.info(f"Collection {collection_name} loaded and ready for search")
        
         # Verify row count
        try:
            total_rows = collection.num_entities
            logger.info(f"Total rows in collection '{collection_name}': {total_rows:,}")
            if total_rows < len(metadatas):
                logger.warning(
                    f"Row count ({total_rows}) is less than expected ({len(metadatas)}). "
                    "Some records may not have been inserted successfully."
                )
        except Exception as e:
            logger.error(f"[ERROR] - Failed to count entities in collection '{collection_name}': {e}")
    
    def search(
        self,
        collection_name: str,
        query_embeddings: np.ndarray,
        top_k: int = 5,
        metric_type: str = "IP",
        search_params: Optional[Dict] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for similar vectors in Milvus.
        
        Args:
            collection_name: Name of the collection
            query_embeddings: numpy array of query embeddings
            top_k: Number of results to return
            metric_type: Similarity metric (IP, L2, COSINE)
            search_params: Index-specific search parameters
            output_fields: Fields to return in results
            
        Returns:
            List of results per query
        """
        if not self._connected:
            self.connect()
        
        collection = Collection(collection_name)
        collection.load()
        
        if search_params is None:
            search_params = {}
        
        if output_fields is None:
            output_fields = ["source_file", "chunk_id", "preview", "content"]
        
        logger.info(f"Searching {len(query_embeddings)} queries (top_k={top_k})...")
        
        results = collection.search(
            data=query_embeddings.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            metric_type=metric_type,
            output_fields=output_fields
        )
        
        # Format results
        all_results = []
        for hits in results:
            query_results = []
            for hit in hits:
                result_dict = {
                    "id": hit.id,
                    "score": hit.score,
                }
                # Add all output fields
                for field in output_fields:
                    value = hit.entity.get(field)
                    # Try to parse JSON strings back to objects
                    if isinstance(value, str) and field in ["table_content", "image_details", "bounding_boxes"]:
                        try:
                            value = json.loads(value)
                        except:
                            pass
                    result_dict[field] = value
                
                query_results.append(result_dict)
            all_results.append(query_results)
        
        logger.info(f"Search complete")
        return all_results
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics about a collection."""
        if not self._connected:
            self.connect()
        
        if not utility.has_collection(collection_name):
            logger.warning(f"Collection {collection_name} does not exist")
            return {}
        
        collection = Collection(collection_name)
        collection.load()
        
        stats = {
            "name": collection_name,
            "num_entities": collection.num_entities,
            "schema": {field.name: str(field.dtype) for field in collection.schema.fields}
        }
        
        return stats