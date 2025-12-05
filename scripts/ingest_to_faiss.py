"""
FAISS ingestion script - loads chunks, generates embeddings, and ingests to FAISS.
Run this script after chunking to ingest documents into the FAISS vector database.
"""
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    get_embedding_config,
    get_chunk_output_path,
    FAISS_CONFIG,
    ACTIVE_EMBEDDING_PROVIDER,
    ACTIVE_EMBEDDING_TYPE,
    EXPERIMENT_CONFIG,
    DATA_DIR
)
from embedding.embedding_manager import EmbeddingManager
from vector_db.faiss_client import FAISSClient
from utils.storage import load_chunks, print_chunk_statistics
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest chunks to FAISS vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast run - minimal arguments
  python ingest_to_faiss.py --chunks data/chunks/output.json --index my_index
  
  # Advanced run - with hyperparameters
  python ingest_to_faiss.py \\
    --chunks data/chunks/output.json \\
    --index my_index \\
    --embedding-model text-embedding-3-small \\
    --index-type HNSW \\
    --normalize
        """
    )
    
    # === FAST RUN ARGUMENTS ===
    parser.add_argument(
        "--chunks",
        type=str,
        help="Path to chunks JSON file (default: data/chunks/<experiment>_chunks.json)"
    )
    parser.add_argument(
        "--index",
        type=str,
        help=f"FAISS index name (default: from config, currently '{FAISS_CONFIG['index']['name']}')"
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing index before creating new one"
    )
    
    # === ADVANCED RUN ARGUMENTS ===
    parser.add_argument(
        "--embedding-provider",
        type=str,
        choices=["openai", "sentence_transformers", "code_transformers"],
        help=f"Embedding provider (default: from config, currently '{ACTIVE_EMBEDDING_PROVIDER}')"
    )
    parser.add_argument(
        "--embedding-type",
        type=str,
        choices=["text", "code"],
        help=f"Embedding documents types (default: from config, currently '{ACTIVE_EMBEDDING_TYPE}')"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        help="Embedding model name (default: from config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Embedding batch size (default: from config)"
    )
    parser.add_argument(
        "--index-type",
        type=str,
        choices=["Flat", "IVF", "HNSW"],
        help=f"FAISS index type (default: from config, currently '{FAISS_CONFIG['index']['index_type']}')"
    )
    parser.add_argument(
        "--metric-type",
        type=str,
        choices=["IP", "L2"],
        help=f"Distance metric (default: from config, currently '{FAISS_CONFIG['index']['metric_type']}')"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize embeddings for cosine similarity (default: from config)"
    )
    
    return parser.parse_args()


def main():
    """Run FAISS ingestion pipeline."""
    args = parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # === RESOLVE PATHS ===
    if args.chunks:
        chunks_file = Path(args.chunks)
        if not chunks_file.is_absolute():
            chunks_file = DATA_DIR / args.chunks
    else:
        chunks_file = get_chunk_output_path()
    
    index_name = args.index if args.index else FAISS_CONFIG["index"]["name"]
    
    # === BUILD CONFIGS ===
    # Embedding config
    embedding_provider = args.embedding_provider or ACTIVE_EMBEDDING_PROVIDER
    embedding_type = args.embedding_type or ACTIVE_EMBEDDING_TYPE
    embed_config = get_embedding_config().copy()
    
    if args.embedding_model:
        embed_config["model"] = args.embedding_model
    if args.batch_size:
        embed_config["batch_size"] = args.batch_size
    
    # FAISS config
    faiss_config = FAISS_CONFIG["index"].copy()
    
    if args.index_type:
        faiss_config["index_type"] = args.index_type
    if args.metric_type:
        faiss_config["metric_type"] = args.metric_type
    if args.normalize:
        faiss_config["normalize"] = True
    
    # === LOG CONFIGURATION ===
    logger.info("="*80)
    logger.info("FAISS INGESTION PIPELINE")
    logger.info("="*80)
    logger.info(f"Experiment: {EXPERIMENT_CONFIG['name']}")
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Chunks file: {chunks_file}")
    logger.info(f"  Index name: {index_name}")
    logger.info(f"  Embedding provider: {embedding_provider}")
    logger.info(f"  Embedding model: {embed_config.get('model')}")
    logger.info(f"  Index type: {faiss_config['index_type']}")
    logger.info(f"  Metric type: {faiss_config['metric_type']}")
    logger.info(f"  Normalize: {faiss_config.get('normalize', False)}")
    logger.info("")
    
    # === VALIDATION ===
    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        logger.info("Run chunking first: python scripts/run_chunking.py")
        return 1
    
    try:
        # === LOAD CHUNKS ===
        logger.info("STEP 1: LOADING CHUNKS")
        logger.info("-"*80)
        contents, metadatas = load_chunks(chunks_file)
        logger.info(f"Loaded {len(contents)} chunks")
        print_chunk_statistics(metadatas)
        
        # === GENERATE EMBEDDINGS ===
        logger.info("")
        logger.info("STEP 2: GENERATING EMBEDDINGS")
        logger.info("-"*80)
        
        embedder = EmbeddingManager.create_embedder(
            provider=embedding_provider,
            embedding_type=embedding_type,
            config=embed_config
        )
        
        logger.info(f"Model: {embedder.model_name}")
        logger.info(f"Dimension: {embedder.get_dimension()}")
        
        embeddings = embedder.embed(contents)
        logger.info(f"Generated embeddings: {embeddings.shape}")
        
        # === INITIALIZE FAISS ===
        logger.info("")
        logger.info("STEP 3: INITIALIZING FAISS")
        logger.info("-"*80)
        
        client = FAISSClient(
            index_dir=faiss_config["index_dir"],
            index_name=index_name
        )
        
        # === CREATE INDEX ===
        logger.info("")
        logger.info("STEP 4: CREATING INDEX")
        logger.info("-"*80)
        
        client.create_index(
            embedding_dim=embedder.get_dimension(),
            index_type=faiss_config["index_type"],
            metric_type=faiss_config["metric_type"],
            index_params=faiss_config.get("params", {}),
            drop_existing=args.drop_existing
        )
        
        # === INGEST DATA ===
        logger.info("")
        logger.info("STEP 5: INGESTING DATA")
        logger.info("-"*80)
        
        client.ingest_data(
            embeddings=embeddings,
            contents=contents,
            metadatas=metadatas,
            normalize=faiss_config.get("normalize", False)
        )
        
        # === VERIFY ===
        logger.info("")
        logger.info("STEP 6: VERIFYING INGESTION")
        logger.info("-"*80)
        
        stats = client.get_index_stats()
        logger.info(f"Total vectors: {stats['num_vectors']:,}")
        logger.info(f"Index type: {stats['index_type']}")
        
        # === SUCCESS ===
        logger.info("")
        logger.info("="*80)
        logger.info("[SUCCESS] - INGESTION COMPLETE")
        logger.info("="*80)
        logger.info(f"Index: {index_name}")
        logger.info(f"Vectors: {stats['num_vectors']:,}")
        logger.info("")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"[ERROR] - Ingestion failed: {e}")
        logger.exception("Full error traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())