# Configuration Guide

Complete reference for all configuration options in SemantIQ.

All configuration is centralized in `config.py`. Edit this file to customize your pipeline.

---

## Quick Configuration

### Essential Settings

```python
# Input documents
INPUT_DIR = DATA_DIR / "your-documents"

# Active components
ACTIVE_EMBEDDING_PROVIDER = "openai"  # or "sentence_transformers"
ACTIVE_EMBEDDING_TYPE = "text"
ACTIVE_VECTOR_DB = "faiss"  # or "milvus"
```

---

## Chunking Configuration

Configure how documents are processed and split.

```python
CHUNKING_CONFIG = {
    "pdf": {
        "chunker_class": "MultimodalPDFChunker",
        "image_coverage_threshold": 0.15,  # 15% triggers vision processing
        "vision_model": "gpt-4o",
        "log_level": "INFO"
    }
}
```

### Parameters

- **`image_coverage_threshold`**: Percentage of page covered by images to trigger vision processing
  - Range: 0.0 to 1.0
  - Default: 0.15 (15%)
  - Lower = more vision processing, higher cost
  
- **`vision_model`**: OpenAI vision model for image-heavy pages
  - Options: `"gpt-4o"`, `"gpt-4-vision-preview"`
  
- **`log_level`**: Logging verbosity
  - Options: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`

---

## Embedding Configuration

Configure embedding models for vectorization.

### OpenAI Embeddings

```python
EMBEDDING_CONFIG = {
    "text": {
        "openai": {
            "model": "text-embedding-3-large",  # or "text-embedding-3-small"
            "batch_size": 64,
            "normalize": True,
            "dimensions": 3072  # 3072 for large, 1536 for small
        }
    }
}
```

**Parameters:**
- **`model`**: Model name
  - `"text-embedding-3-large"`: 3072 dimensions, best quality
  - `"text-embedding-3-small"`: 1536 dimensions, faster/cheaper
  
- **`batch_size`**: Number of texts to embed at once
  - Default: 64
  - Reduce if hitting rate limits or memory issues
  
- **`normalize`**: Normalize embeddings to unit length
  - Required for cosine similarity with inner product metric
  
- **`dimensions`**: Output dimension size
  - Must match model capabilities

### Sentence Transformers

```python
EMBEDDING_CONFIG = {
    "text": {
        "sentence_transformers": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 64,
            "normalize": True,
            "dimensions": 384
        }
    }
}
```

**Parameters:**
- **`model`**: HuggingFace model identifier
  - Popular options:
    - `"sentence-transformers/all-MiniLM-L6-v2"` (384d, fast)
    - `"sentence-transformers/all-mpnet-base-v2"` (768d, quality)
  
- Other parameters same as OpenAI

### Active Provider

```python
ACTIVE_EMBEDDING_PROVIDER = "openai"  # or "sentence_transformers"
ACTIVE_EMBEDDING_TYPE = "text"
```

---

## Vector Database Configuration

### FAISS Configuration

```python
FAISS_CONFIG = {
    "index": {
        "index_dir": "data/faiss_indices",
        "name": "document_embeddings",
        "index_type": "Flat",      # Flat, IVF, HNSW
        "metric_type": "IP",       # IP, L2
        "normalize": True,         # For cosine similarity
        "use_gpu": False,
        "gpu_id": 0,
        "params": {
            # Index-specific parameters
        }
    },
    "search": {
        "top_k": 5,
        "params": {}
    }
}
```

**Index Parameters:**
- **`index_dir`**: Directory for index files
- **`name`**: Index name (filename prefix)
- **`index_type`**: Index algorithm
  - `"Flat"`: Exact search, best accuracy
  - `"IVF"`: Fast approximate search
  - `"HNSW"`: Best speed/accuracy balance
  
- **`metric_type`**: Distance metric
  - `"IP"`: Inner product (for normalized vectors = cosine)
  - `"L2"`: Euclidean distance
  
- **`normalize`**: Normalize embeddings for cosine similarity
- **`use_gpu`**: Enable GPU acceleration (requires faiss-gpu)
- **`gpu_id`**: GPU device ID

**Index Type Specific Parameters:**

IVF:
```python
"params": {
    "nlist": 100  # Number of clusters
}
"search": {
    "params": {"nprobe": 10}  # Clusters to search
}
```

HNSW:
```python
"params": {
    "M": 32  # Number of connections per layer
}
```

### Milvus Configuration

```python
MILVUS_CONFIG = {
    "connection": {
        "host": "localhost",
        "port": "19530",
        "alias": "default"
    },
    "collection": {
        "name": "document_embeddings",
        "description": "Document embeddings with metadata"
    },
    "index": {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 1024}
    },
    "search": {
        "top_k": 5,
        "params": {}
    }
}
```

**Connection Parameters:**
- **`host`**: Milvus server address
- **`port`**: Milvus server port (default: 19530)
- **`alias`**: Connection alias

**Collection Parameters:**
- **`name`**: Collection name
- **`description`**: Collection description

**Index Parameters:**
- **`index_type`**: Index algorithm
  - `"IVF_FLAT"`: Balanced performance
  - `"HNSW"`: Best accuracy
  - `"IVF_PQ"`: Fastest, uses quantization
  
- **`metric_type`**: Distance metric
  - `"IP"`: Inner product
  - `"L2"`: Euclidean distance
  - `"COSINE"`: Cosine similarity
  
- **`params`**: Index-specific parameters
  - IVF: `{"nlist": 1024}`
  - HNSW: `{"M": 16, "efConstruction": 200}`

**Search Parameters:**
- **`top_k`**: Number of results to return
- **`params`**: Search-specific parameters
  - IVF: `{"nprobe": 10}`

---

## Evaluation Configuration

```python
EVALUATION_CONFIG = {
    "metrics": [
        "precision@k",
        "recall@k", 
        "f1@k",
        "hit_rate@k",
        "mrr",
        "map",
        "ndcg@k"
    ],
    "k_values": [1, 3, 5, 10],
    "default_ground_truth": str(GROUND_TRUTH_DIR / "default_queries.json")
}
```

**Parameters:**
- **`metrics`**: List of metrics to compute
- **`k_values`**: K values for @k metrics
- **`default_ground_truth`**: Default ground truth file path

---

## Experiment Configuration

Track experiment metadata.

```python
EXPERIMENT_CONFIG = {
    "name": "my_experiment",
    "description": "Testing different chunking strategies",
    "version": "1.0.0",
    "tags": ["pdf", "multimodal", "faiss", "openai"]
}
```

---

## Logging Configuration

```python
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "rag_pipeline.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"]
    }
}
```

**Key Settings:**
- **Console level**: `"INFO"` for important messages only
- **File level**: `"DEBUG"` for detailed logging
- **`maxBytes`**: Max log file size before rotation
- **`backupCount`**: Number of backup log files to keep

---

## Path Configuration

Default paths can be customized:

```python
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "your-documents"
CHUNKS_DIR = DATA_DIR / "chunks"
LOGS_DIR = PROJECT_ROOT / "logs"

# Evaluation paths
EVAL_DIR = DATA_DIR / "evaluation"
GROUND_TRUTH_DIR = EVAL_DIR / "ground_truth"
QUERY_RESULTS_DIR = EVAL_DIR / "query_results"
EVAL_RESULTS_DIR = EVAL_DIR / "eval_results"
```

---

## Configuration Recipes

### Development Setup (Fast, Local)

```python
ACTIVE_VECTOR_DB = "faiss"
ACTIVE_EMBEDDING_PROVIDER = "sentence_transformers"

EMBEDDING_CONFIG = {
    "text": {
        "sentence_transformers": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 128
        }
    }
}

FAISS_CONFIG = {
    "index": {
        "index_type": "Flat",
        "metric_type": "IP",
        "normalize": True
    }
}
```

### Production Setup (Quality, Scalable)

```python
ACTIVE_VECTOR_DB = "milvus"
ACTIVE_EMBEDDING_PROVIDER = "openai"

EMBEDDING_CONFIG = {
    "text": {
        "openai": {
            "model": "text-embedding-3-large",
            "batch_size": 64,
            "normalize": True
        }
    }
}

MILVUS_CONFIG = {
    "index": {
        "index_type": "HNSW",
        "metric_type": "IP",
        "params": {"M": 16, "efConstruction": 200}
    }
}
```

### Large Dataset Setup (10M+ vectors)

```python
ACTIVE_VECTOR_DB = "milvus"

MILVUS_CONFIG = {
    "index": {
        "index_type": "IVF_PQ",
        "metric_type": "IP",
        "params": {
            "nlist": 4096,
            "m": 8
        }
    },
    "search": {
        "params": {"nprobe": 32}
    }
}
```

---

## Helper Functions

```python
# Get active embedding config
embed_config = get_embedding_config()

# Get chunk output path
chunk_path = get_chunk_output_path()

# Get collection/index name
collection_name = get_collection_name()

# Get vector DB config
vdb_config = get_vector_db_config()

# Get evaluation paths
eval_paths = get_eval_paths()
```

---

## Environment Variables

Create `.env` file:

```bash
# Required for OpenAI
OPENAI_API_KEY=your_key_here

# Optional: Custom paths
# DATA_DIR=/custom/data/path
# LOGS_DIR=/custom/logs/path
```

Load in scripts:
```python
from dotenv import load_dotenv
load_dotenv()
```

---