# Extension Guide

How to extend SemantIQ with new document types, embeddings, and vector databases.

---

## Adding New Document Types

### Step 1: Create Chunker Class

Create `chunking/my_document_chunker.py`:

```python
from chunking.base import BaseChunker
from typing import List, Dict, Any
from pathlib import Path

class MyDocumentChunker(BaseChunker):
    """Chunker for MyDocument files."""
    
    def __init__(self, **config):
        super().__init__()
        self.config = config
    
    def chunk(self, file_path: Path) -> tuple[List[str], List[Dict[str, Any]]]:
        """
        Chunk a MyDocument file.
        
        Args:
            file_path: Path to document
            
        Returns:
            Tuple of (contents, metadatas)
        """
        contents = []
        metadatas = []
        
        # Your chunking logic here
        # ...
        
        return contents, metadatas
    
    def get_metadata_schema(self) -> Dict[str, type]:
        """
        Define metadata schema for vector database.
        
        Returns:
            Dictionary mapping field names to Python types
        """
        return {
            "source_file": str,
            "chunk_id": str,
            "chunk_type": str,
            "custom_field": str
            # Add your fields
        }
```

### Step 2: Update Configuration

Add to `config.py`:

```python
CHUNKING_CONFIG = {
    "pdf": {
        "chunker_class": "MultimodalPDFChunker",
        # ...
    },
    "mydocument": {
        "chunker_class": "MyDocumentChunker",
        "custom_param": "value"
    }
}
```

### Step 3: Register Chunker

Update `scripts/run_chunking.py` to include your chunker.

### Step 4: Test

```bash
# Add MyDocument files to INPUT_DIR
uv run scripts/run_chunking.py
```

---

## Adding New Embedding Models

### Step 1: Create Embedder Class

Create `embedding/my_embedder.py`:

```python
from embedding.base import BaseEmbedder
import numpy as np
from typing import List

class MyEmbedder(BaseEmbedder):
    """Custom embedding model."""
    
    def __init__(self, model_name: str, **config):
        super().__init__(model_name)
        self.config = config
        # Initialize your model
        # self.model = ...
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            numpy array of shape (len(texts), dimension)
        """
        # Your embedding logic
        embeddings = []
        
        for text in texts:
            # embedding = self.model.encode(text)
            # embeddings.append(embedding)
            pass
        
        return np.array(embeddings)
    
    def get_dimension(self) -> int:
        """Return embedding dimension."""
        return 768  # Your dimension
```

### Step 2: Register in EmbeddingManager

Update `embedding/embedding_manager.py`:

```python
from embedding.my_embedder import MyEmbedder

class EmbeddingManager:
    @staticmethod
    def create_embedder(provider: str, embedding_type: str, config: dict):
        # ...
        if embedding_type == "NEW_EMBEDDING_TYPE_FOR_EXAMPLE_CODE_FILES":
            if provider == "MY_CODE_EMBEDDING_MODEL_PROVIDER":
                return MyEmbedder(
                    model_name=config["model"],
                    **config
                )
```

### Step 3: Add Configuration

Add to `config.py`:

```python
EMBEDDING_CONFIG = {
    "text": {
        # ...
        "my_provider": {
            "model": "my-model-name",
            "batch_size": 64,
            "custom_param": "value"
        }
    }
}

ACTIVE_EMBEDDING_PROVIDER = "my_provider"
```

### Step 4: Test

```bash
uv run scripts/ingest_to_faiss.py
```

---

## Adding New Vector Databases

### Step 1: Create Client Class

Create `vector_db/my_vectordb_client.py`:

```python
from typing import List, Dict, Any, Optional
import numpy as np

class MyVectorDBClient:
    """Client for MyVectorDB."""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        # Initialize connection
    
    def create_collection(
        self,
        collection_name: str,
        embedding_dim: int,
        metadata_schema: Dict[str, type],
        drop_existing: bool = False
    ):
        """Create a collection/index."""
        pass
    
    def ingest_data(
        self,
        collection_name: str,
        embeddings: np.ndarray,
        contents: List[str],
        metadatas: List[Dict[str, Any]]
    ):
        """Ingest data."""
        pass
    
    def search(
        self,
        collection_name: str,
        query_embeddings: np.ndarray,
        top_k: int = 5,
        output_fields: Optional[List[str]] = None
    ) -> List[List[Dict[str, Any]]]:
        """Search for similar vectors."""
        pass
    
    def get_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics."""
        pass
```

### Step 2: Create Ingestion Script

Create `scripts/ingest_to_myvectordb.py`:

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import get_embedding_config, ACTIVE_EMBEDDING_PROVIDER, ACTIVE_EMBEDDING_TYPE
from embedding.embedding_manager import EmbeddingManager
from vector_db.my_vectordb_client import MyVectorDBClient
from utils.storage import load_chunks
from utils.logger import get_logger

logger = get_logger(__name__)

def main():
    # Load chunks
    chunks_path = get_chunk_output_path()
    contents, metadatas = load_chunks(chunks_path)
    
    # Generate embeddings
    embed_config = get_embedding_config()
    embedder = EmbeddingManager.create_embedder(
        provider=ACTIVE_EMBEDDING_PROVIDER,
        embedding_type=ACTIVE_EMBEDDING_TYPE,
        config=embed_config
    )
    embeddings = embedder.embed(contents)
    
    # Connect and ingest
    client = MyVectorDBClient(host="localhost", port=12345)
    client.create_collection(
        collection_name="my_collection",
        embedding_dim=embedder.get_dimension(),
        metadata_schema={k: type(v) for k, v in metadatas[0].items()},
        drop_existing=True
    )
    client.ingest_data(
        collection_name="my_collection",
        embeddings=embeddings,
        contents=contents,
        metadatas=metadatas
    )
    
    logger.info("Ingestion complete")

if __name__ == "__main__":
    main()
```

### Step 3: Create Query Script

Create `scripts/query_myvectordb.py`:

```python
# Similar structure to ingest script
# Use client.search() to query
```

### Step 4: Add Configuration

Add to `config.py`:

```python
MY_VECTORDB_CONFIG = {
    "connection": {
        "host": "localhost",
        "port": 12345
    },
    "collection": {
        "name": "my_collection"
    },
    "search": {
        "top_k": 5
    }
}

ACTIVE_VECTOR_DB = "my_vectordb"
```

### Step 5: Update Unified Query Script

Update `scripts/query_vector_db.py` to support your database.

---

## Adding Hybrid Search

### Example: BM25 + Dense Retrieval

```python
from vector_db.milvus_client import MilvusClient
from rank_bm25 import BM25Okapi
import numpy as np

class HybridSearcher:
    def __init__(self, vector_client, documents):
        self.vector_client = vector_client
        
        # Initialize BM25
        tokenized_docs = [doc.split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        self.documents = documents
    
    def search(self, query: str, top_k: int = 10, alpha: float = 0.5):
        """
        Hybrid search combining BM25 and dense retrieval.
        
        Args:
            query: Query string
            top_k: Number of results
            alpha: Weight for dense scores (1-alpha for BM25)
        """
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query.split())
        
        # Dense retrieval scores
        dense_results = self.vector_client.search(query, top_k=top_k)
        
        # Combine scores
        # ... (implement score fusion)
        
        return combined_results
```

---

## Adding Code Embeddings

### Step 1: Create Code Embedder

```python
from embedding.base import BaseEmbedder

class CodeEmbedder(BaseEmbedder):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # Use code-specific model
        # e.g., microsoft/codebert-base
    
    def embed(self, texts: List[str]) -> np.ndarray:
        # Implement code embedding
        pass
```

### Step 2: Add Configuration

```python
EMBEDDING_CONFIG = {
    "code": {
        "openai": {
            "model": "text-embedding-3-large"
        }
    }
}
```

### Step 3: Create Code Chunker

```python
class CodeChunker(BaseChunker):
    def chunk(self, file_path: Path):
        # Chunk by functions, classes, etc.
        pass
```

---

## Custom Metadata Fields

### Extend Metadata Dataclass

```python
from dataclasses import dataclass

@dataclass
class MyCustomMetadata:
    source_file: str
    chunk_id: str
    custom_field1: str
    custom_field2: int
    custom_field3: float
```

### Use in Chunker

```python
def chunk(self, file_path: Path):
    # ...
    metadata = {
        "source_file": str(file_path),
        "chunk_id": chunk_id,
        "custom_field1": "value",
        "custom_field2": 123,
        "custom_field3": 0.456
    }
    metadatas.append(metadata)
```

Schema is automatically generated for vector databases.

---

## Best Practices

### 1. Follow Existing Patterns
- Study existing implementations (PDF chunker, OpenAI embedder, etc.)
- Use the same interfaces and patterns
- Maintain consistency

### 2. Error Handling
```python
def embed(self, texts: List[str]) -> np.ndarray:
    try:
        embeddings = self.model.encode(texts)
        return np.array(embeddings)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise
```

### 3. Logging
```python
logger.info("Starting process...")
logger.debug(f"Processing {len(items)} items")
logger.warning("Unusual condition detected")
logger.error("Operation failed")
```

### 4. Configuration Validation
```python
def __init__(self, **config):
    required = ["model", "batch_size"]
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required config: {key}")
```

### 5. Testing
```python
def test_my_embedder():
    embedder = MyEmbedder(model_name="test-model")
    texts = ["test text 1", "test text 2"]
    embeddings = embedder.embed(texts)
    
    assert embeddings.shape == (2, embedder.get_dimension())
    assert embeddings.dtype == np.float32
```

---

## Examples

See existing implementations:
- **PDF Chunker**: `chunking/pdf_chunker.py`
- **OpenAI Embedder**: `embedding/embedding_manager.py`
- **Milvus Client**: `vector_db/milvus_client.py`
- **FAISS Client**: `vector_db/faiss_client.py`

---

## Next Steps

1. Identify what you want to extend
2. Review existing implementation
3. Create your extension following the patterns
4. Add configuration
5. Test thoroughly
6. Update documentation

For questions, open an issue on GitHub.