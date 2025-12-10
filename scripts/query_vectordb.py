"""
Unified script for querying vector databases (Milvus or FAISS).
Automatically uses the active vector database from config.
Supports saving results for evaluation.
"""
"""
Unified query script with command-line interface.
Supports querying both FAISS and Milvus with optional result saving.
"""
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Optional

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    get_embedding_config,
    ACTIVE_VECTOR_DB,
    MILVUS_CONFIG,
    FAISS_CONFIG,
    ACTIVE_EMBEDDING_PROVIDER,
    ACTIVE_EMBEDDING_TYPE,
    EXPERIMENT_CONFIG,
    DATA_DIR,
    QUERY_RESULTS_DIR
)
from embedding.embedding_manager import EmbeddingManager
from vector_db.milvus_client import MilvusClient
from vector_db.faiss_client import FAISSClient
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Query vector database (FAISS or Milvus)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast run - simple query
  python query_vector_db.py --query "What is covered by insurance?"
  
  # Fast run - multiple queries
  python query_vector_db.py --query "query 1" "query 2" "query 3"
  
  # Fast run - from file
  python query_vector_db.py --queries-file data/evaluation/queries.json
  
  # Advanced run - with options
  python query_vector_db.py \\
    --query "What is covered?" \\
    --top-k 10 \\
    --save-results data/evaluation/query_results/output.json \\
    --vector-db faiss
        """
    )
    
    # === FAST RUN ARGUMENTS ===
    parser.add_argument(
        "--query",
        type=str,
        nargs="+",
        help="Query string(s) to search"
    )
    parser.add_argument(
        "--queries-file",
        type=str,
        help="JSON file with queries (format: {'queries': [{'query_id': '...', 'query_text': '...'}]})"
    )
    parser.add_argument(
        "--save-results",
        type=str,
        help="Save query results to JSON file (for evaluation)"
    )
    
    # === ADVANCED RUN ARGUMENTS ===
    parser.add_argument(
        "--vector-db",
        type=str,
        choices=["faiss", "milvus"],
        help=f"Vector database to use (default: from config, currently '{ACTIVE_VECTOR_DB}')"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Number of results to return (default: from config)"
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        choices=["openai", "sentence_transformers"],
        help=f"Embedding provider (default: from config, currently '{ACTIVE_EMBEDDING_PROVIDER}')"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        help="Embedding model name (default: from config)"
    )
    
    # FAISS-specific
    parser.add_argument(
        "--faiss-index",
        type=str,
        help=f"FAISS index name (default: from config)"
    )
    
    # Milvus-specific
    parser.add_argument(
        "--milvus-collection",
        type=str,
        help=f"Milvus collection name (default: from config)"
    )
    parser.add_argument(
        "--milvus-host",
        type=str,
        help="Milvus host (default: from config)"
    )
    parser.add_argument(
        "--milvus-port",
        type=str,
        help="Milvus port (default: from config)"
    )
    
    return parser.parse_args()


def load_queries_from_file(file_path: Path):
    """Load queries from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    queries = [q["query_text"] for q in data["queries"]]
    query_ids = [q["query_id"] for q in data["queries"]]
    
    return queries, query_ids


def save_results(results: dict, output_path: Path):
    """Save query results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving results to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info("Results saved successfully")


def main():
    """Run query pipeline."""
    args = parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # === PARSE QUERIES ===
    if args.queries_file:
        queries_file = Path(args.queries_file)
        if not queries_file.is_absolute():
            queries_file = DATA_DIR / args.queries_file
        
        if not queries_file.exists():
            logger.error(f"Queries file not found: {queries_file}")
            return 1
        
        queries, query_ids = load_queries_from_file(queries_file)
    elif args.query:
        # Join all query arguments into individual queries
        queries = args.query
        query_ids = [f"q{i+1}" for i in range(len(queries))]
    else:
        # Default example queries
        logger.warning("No queries provided. Using default examples.")
        queries = [
            "How much does X cover for surgeries",
            "What are the hospitals covered?"
        ]
        query_ids = ["q1", "q2"]
    
    # === BUILD CONFIG ===
    vector_db = args.vector_db or ACTIVE_VECTOR_DB
    embedding_provider = args.embedding_provider or ACTIVE_EMBEDDING_PROVIDER
    
    embed_config = get_embedding_config().copy()
    if args.embedding_model:
        embed_config["model"] = args.embedding_model
    
    # === LOG CONFIGURATION ===
    logger.info("="*80)
    logger.info(f"QUERYING {vector_db.upper()} VECTOR DATABASE")
    logger.info("="*80)
    logger.info(f"Experiment: {EXPERIMENT_CONFIG['name']}")
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Vector DB: {vector_db}")
    logger.info(f"  Embedding provider: {embedding_provider}")
    logger.info(f"  Embedding model: {embed_config.get('model')}")
    logger.info(f"  Number of queries: {len(queries)}")
    logger.info(f"  Top K: {args.top_k or 'from config'}")
    logger.info("")
    
    try:
        # === GENERATE QUERY EMBEDDINGS ===
        logger.info("Generating query embeddings...")
        embedder = EmbeddingManager.create_embedder(
            provider=embedding_provider,
            embedding_type=ACTIVE_EMBEDDING_TYPE,
            config=embed_config
        )
        
        query_embeddings = embedder.embed(queries)
        logger.info(f"Embedded {len(queries)} queries")
        
        # === QUERY VECTOR DATABASE ===
        output_fields = [
            "source_file",
            "id",
            "page_number",
            "chunk_type",
            "preview",
            "content"
        ]
        
        if vector_db == "faiss":
            results = _query_faiss(
                query_embeddings=query_embeddings,
                top_k=args.top_k,
                output_fields=output_fields,
                index_name=args.faiss_index
            )
            index_name = args.faiss_index or FAISS_CONFIG["index"]["name"]
        elif vector_db == "milvus":
            results = _query_milvus(
                query_embeddings=query_embeddings,
                top_k=args.top_k,
                output_fields=output_fields,
                collection_name=args.milvus_collection,
                host=args.milvus_host,
                port=args.milvus_port
            )
            index_name = args.milvus_collection or MILVUS_CONFIG["collection"]["name"]
        else:
            logger.error(f"Unknown vector database: {vector_db}")
            return 1
        
        # === DISPLAY RESULTS ===
        _display_results(queries, results)
        
        # === BUILD OUTPUT ===
        structured_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "vector_db": vector_db,
                "embedding_provider": embedding_provider,
                "embedding_model": embedder.model_name,
                "collection_name": index_name,
                "top_k": args.top_k or "default",
                "num_queries": len(queries),
                "experiment_name": EXPERIMENT_CONFIG.get("name", "unknown")
            },
            "queries": []
        }
        
        for query_id, query_text, query_results in zip(query_ids, queries, results):
            structured_results["queries"].append({
                "query_id": query_id,
                "query_text": query_text,
                "results": query_results
            })
        
        # === SAVE RESULTS ===
        if args.save_results:
            save_path = Path(args.save_results)
            if not save_path.is_absolute():
                save_path = DATA_DIR / args.save_results
            save_results(structured_results, save_path)
        else:
            # Suggest save location
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suggested_path = QUERY_RESULTS_DIR / f"query_results_{vector_db}_{timestamp}.json"
            logger.info(f"\nTo save results for evaluation, run with:")
            logger.info(f"  --save-results {suggested_path}")
        
        logger.info("")
        logger.info("[SUCCESS] - Query complete")
        
        return 0
        
    except Exception as e:
        logger.error(f"[ERROR] - Query failed: {e}")
        logger.exception("Full error traceback:")
        return 1


def _query_faiss(query_embeddings, top_k, output_fields, index_name=None):
    """Query FAISS database."""
    index_name = index_name or FAISS_CONFIG["index"]["name"]
    index_dir = FAISS_CONFIG["index"]["index_dir"]
    
    client = FAISSClient(
        index_dir=index_dir,
        index_name=index_name
    )
    
    logger.info(f"Loading FAISS index: {index_name}")
    if not client.load_index():
        raise FileNotFoundError(
            f"Index '{index_name}' not found. "
            f"Please run ingestion first: python scripts/ingest_to_faiss.py"
        )
    
    search_config = FAISS_CONFIG["search"]
    top_k = top_k or search_config["top_k"]
    normalize = FAISS_CONFIG["index"].get("normalize", False)
    
    logger.info(f"Searching (top_k={top_k})...")
    
    return client.search(
        query_embeddings=query_embeddings,
        top_k=top_k,
        normalize=normalize,
        search_params=search_config.get("params", {}),
        output_fields=output_fields
    )


def _query_milvus(query_embeddings, top_k, output_fields, collection_name=None, host=None, port=None):
    """Query Milvus database."""
    milvus_conn = MILVUS_CONFIG["connection"].copy()
    
    if host:
        milvus_conn["host"] = host
    if port:
        milvus_conn["port"] = port
    
    collection_name = collection_name or MILVUS_CONFIG["collection"]["name"]
    
    client = MilvusClient(
        host=milvus_conn["host"],
        port=milvus_conn["port"],
        alias=milvus_conn["alias"]
    )
    
    logger.info(f"Connecting to Milvus at {milvus_conn['host']}:{milvus_conn['port']}")
    client.connect()
    
    search_config = MILVUS_CONFIG["search"]
    top_k = top_k or search_config["top_k"]
    
    logger.info(f"Searching collection '{collection_name}' (top_k={top_k})...")
    
    results = client.search(
        collection_name=collection_name,
        query_embeddings=query_embeddings,
        top_k=top_k,
        metric_type=MILVUS_CONFIG['index']['metric_type'],
        search_params=search_config["params"],
        output_fields=output_fields
    )
    
    client.disconnect()
    return results


def _display_results(queries, results):
    """Display search results."""
    logger.info("")
    logger.info("="*80)
    logger.info("SEARCH RESULTS")
    logger.info("="*80)
    
    for i, (query, query_results) in enumerate(zip(queries, results)):
        logger.info("")
        logger.info(f"Query {i+1}: {query}")
        logger.info("-"*80)
        
        if not query_results:
            logger.info("  No results found")
            continue
        
        for rank, hit in enumerate(query_results, start=1):
            logger.info(f"\n  Rank {rank} (Score: {hit['score']:.4f})")
            logger.info(f"  Source: {hit.get('source_file', 'N/A')}")
            logger.info(f"  Page: {hit.get('page_number', 'N/A')}")
            logger.info(f"  Preview: {hit.get('preview', '')[:150]}...")
        
        logger.info("")
    
    logger.info("="*80)


if __name__ == "__main__":
    load_dotenv()
    sys.exit(main())