"""
Configuration file for RAG pipeline experiments.
Modify these settings to experiment with different configurations.
"""
from pathlib import Path
from typing import Dict, Any

# ============================================================================
# PROJECT PATHS
# ============================================================================
INPUT_FOLDER_NAME = "test" # TODO: Edit input folder name - decides: chunks output file name, indices output file name

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / INPUT_FOLDER_NAME
CHUNKS_DIR = DATA_DIR / "chunks"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "default",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / f"{INPUT_FOLDER_NAME}_rag_pipeline.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file"]
    }
}

# ============================================================================
# CHUNKING CONFIGURATION
# ============================================================================
CHUNKING_CONFIG = {
    "pdf": {
        "image_coverage_threshold": 0.15,  # 15% triggers vision processing
        "vision_model": "gpt-4o",
        "log_level": "INFO"
    },
    "word_doc": {
        "vision_model": "gpt-4o"
    },
    # Word Document Chunker Configuration
    "word": {
        "max_chunk_size": 1000,            # Maximum characters per chunk
        "vision_model": "gpt-4o",          # Vision model for image processing
        "process_images": True,            # Whether to process images with vision model
        "log_level": "INFO"                # Logging level
    },
    
    # Code Chunker Configuration
    "code": {
        "max_chunk_size": 1500,            # Maximum characters per chunk
        "min_chunk_size": 100,             # Minimum characters per chunk
        "include_imports": True,           # Include imports in context
        "include_docstrings": True,        # Include docstrings
        "overlap_lines": 3,                # Lines of overlap between chunks
        "log_level": "INFO"                # Logging level
    },
    
    # ========================================================================
    # SECTION: Add more document type configurations here
    # ========================================================================
    #
    # # Markdown Chunker Configuration
    # "markdown": {
    #     "chunk_by_headers": True,        # Chunk by markdown headers
    #     "max_chunk_size": 1000,          # Maximum characters per chunk
    #     "preserve_code_blocks": True,    # Keep code blocks intact
    #     "log_level": "INFO"
    # },
    #
    # # Text Chunker Configuration
    # "text": {
    #     "chunk_size": 1000,              # Characters per chunk
    #     "chunk_overlap": 200,            # Overlap between chunks
    #     "separator": "\n\n",             # Chunk separator
    #     "log_level": "INFO"
    # },
    #
    # # HTML Chunker Configuration
    # "html": {
    #     "chunk_by_sections": True,       # Chunk by HTML sections
    #     "max_chunk_size": 1000,          # Maximum characters per chunk
    #     "extract_tables": True,          # Extract HTML tables
    #     "log_level": "INFO"
    # },
    #
    # ========================================================================
}

# ============================================================================
# EMBEDDING CONFIGURATION
# ============================================================================
EMBEDDING_CONFIG = {
    "text": {
        # OpenAI embeddings
        "openai": {
            "model": "text-embedding-3-large",  # or "text-embedding-3-small"
            "batch_size": 64,
            "normalize": True,
            "dimensions": 3072  # 3072 for large, 1536 for small
        },
        # Sentence Transformers
        "sentence_transformers": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "batch_size": 64,
            "normalize": True,
            "dimensions": 384
        }
    },
    # Code embeddings
    "code": {
        # OpenAI embeddings for code (with code-specific preprocessing)
        "openai": {
            "model": "text-embedding-3-large",
            "batch_size": 64,
            "normalize": True,
            "dimensions": 3072,
            "add_language_prefix": True  # Prepend language name for better context
        },
        # Code-specific transformer models
        "code_transformers": {
            # Options:
            # - "microsoft/codebert-base" (768 dim)
            # - "microsoft/graphcodebert-base" (768 dim) 
            # - "Salesforce/codet5-base" (768 dim)
            # - "huggingface/CodeBERTa-small-v1" (768 dim)
            "model": "microsoft/codebert-base",
            "batch_size": 32,
            "normalize": True,
            "dimensions": 768,
            "add_language_prefix": True
        }
    }
}

# Current embedding provider to use
ACTIVE_EMBEDDING_PROVIDER = "openai"  # or "sentence_transformers", "code_transformer"
ACTIVE_EMBEDDING_TYPE = "text"

# ============================================================================
# VECTOR DATABASE SELECTION
# ============================================================================
ACTIVE_VECTOR_DB = "faiss"  # Options: "milvus", "faiss"

# ============================================================================
# VECTOR DATABASE CONFIGURATION - MILVUS
# ============================================================================
MILVUS_CONFIG = {
    "connection": {
        "host": "localhost",
        "port": "19530",
        "alias": "default"
    },
    "collection": {
        "name": "X_document_embeddings",
        "description": "Document embeddings with full chunk metadata"
    },
    "index": {
        "index_type": "IVF_FLAT",  # Options: HNSW, IVF_FLAT, IVF_PQ, etc.
        "metric_type": "IP",  # Options: IP (inner product), L2, COSINE
        "params": {
            "nlist": 1024  # For IVF_FLAT
            # For HNSW: {"M": 16, "efConstruction": 200}
        }
    },
    "search": {
        "top_k": 5,
        "params": {}  # Index-specific search params, e.g., {"nprobe": 10} for IVF
    }
}

# ============================================================================
# VECTOR DATABASE CONFIGURATION - FAISS
# ============================================================================
FAISS_CONFIG = {
    "index": {
        "index_dir": str(DATA_DIR / "faiss_indices"),
        "name": f"{INPUT_FOLDER_NAME}",
        "index_type": "Flat",  # Options: Flat, IVF, HNSW
        "metric_type": "IP",  # Options: IP (inner product), L2
        "normalize": True,  # True for cosine similarity with IP metric
        "params": {
            # For IVF: {"nlist": 100}
            # For HNSW: {"M": 32}
        }
    },
    "search": {
        "top_k": 5,
        "params": {
            # For IVF: {"nprobe": 10}
        }
    }
}
# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================
QUERIES = [ # Modify
    "How much does X cover for surgeries?",
    "What hospitals are covered?"
]
# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================
EXPERIMENT_CONFIG = {
    "name": f"{INPUT_FOLDER_NAME}_exp",
    "description": "Initial RAG pipeline with PDF multimodal chunking",
    "version": "1.0.0",
    "tags": ["pdf", "multimodal", "milvus", "openai"]
}
# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================
EVAL_DIR = DATA_DIR / "evaluation"
GROUND_TRUTH_DIR = EVAL_DIR / "ground_truth"
QUERY_RESULTS_DIR = EVAL_DIR / "query_results"
EVAL_RESULTS_DIR = EVAL_DIR / "eval_results"

# Ensure directories exist
GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
QUERY_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
    "default_ground_truth": str(GROUND_TRUTH_DIR / "default_queries.json"),
    "results_naming": "{experiment_name}_{vector_db}_{embedding_provider}_{timestamp}.json"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_embedding_config() -> Dict[str, Any]:
    """Get active embedding configuration."""
    return EMBEDDING_CONFIG[ACTIVE_EMBEDDING_TYPE][ACTIVE_EMBEDDING_PROVIDER]

def get_chunk_output_path(experiment_name: str = None) -> Path:
    """Get output path for chunks."""
    if experiment_name:
        return CHUNKS_DIR / f"{experiment_name}_chunks.json"
    return CHUNKS_DIR / f"{EXPERIMENT_CONFIG['name']}_chunks.json"

def get_collection_name() -> str:
    """Get Milvus collection name."""
    return MILVUS_CONFIG["collection"]["name"]

def get_vector_db_config() -> Dict[str, Any]:
    """Get active vector database configuration."""
    if ACTIVE_VECTOR_DB == "milvus":
        return MILVUS_CONFIG
    elif ACTIVE_VECTOR_DB == "faiss":
        return FAISS_CONFIG
    else:
        raise ValueError(f"Unknown vector database: {ACTIVE_VECTOR_DB}")
    
def get_eval_paths() -> Dict[str, Path]:
    """Get evaluation-related paths."""
    return {
        "ground_truth_dir": GROUND_TRUTH_DIR,
        "query_results_dir": QUERY_RESULTS_DIR,
        "eval_results_dir": EVAL_RESULTS_DIR
    }
    
    
def resolve_chunks_path(chunks_arg: str = None) -> Path:
    """
    Intelligently resolve chunks file path.
    
    Supports:
    - None: Use default from config
    - "output.json": Assumes data/chunks/output.json
    - "chunks/output.json": Resolves to data/chunks/output.json
    - "data/chunks/output.json": Uses as-is
    - "/absolute/path/output.json": Uses as-is
    
    Args:
        chunks_arg: Chunks file path argument from command line
        
    Returns:
        Resolved Path object
    """
    if chunks_arg is None:
        return get_chunk_output_path()
    
    chunks_path = Path(chunks_arg)
    
    # If absolute path, use as-is
    if chunks_path.is_absolute():
        return chunks_path
    
    # If file exists at given path, use it
    if chunks_path.exists():
        return chunks_path
    
    # Smart resolution:
    # 1. If just filename (no directory parts), assume it's in data/chunks/
    if len(chunks_path.parts) == 1:
        return CHUNKS_DIR / chunks_path
    
    # 2. If starts with "chunks/", prepend with DATA_DIR
    if chunks_path.parts[0] == "chunks":
        return DATA_DIR / chunks_path
    
    # 3. If starts with "data/", use from project root
    if chunks_path.parts[0] == "data":
        return PROJECT_ROOT / chunks_path
    
    # 4. Otherwise, assume relative to DATA_DIR
    return DATA_DIR / chunks_path