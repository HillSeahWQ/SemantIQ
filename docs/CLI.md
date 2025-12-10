# Command-Line Interface Guide

Complete guide for using SemantIQ's CLI interface.

## Overview

All entry point scripts support **two modes**:

1. **Fast Run**: Minimal arguments (just file paths)
2. **Advanced Run**: Full control over hyperparameters

**Default behavior**: All scripts read from `data/` subdirectories and use config file defaults.

---

## Quick Reference

### Chunking
```bash
# Fast: minimal arguments
uv run scripts/run_chunking.py --input data/YOUR-RAW-DOCUMENTS-FOLDER --output chunks/YOUR-CHUNKS-FILENAME.json

# Advanced: with hyperparameters
uv run scripts/run_chunking.py --input data/YOUR-RAW-DOCUMENTS-FOLDER --output chunks/YOUR-CHUNKS-FILENAME.json --image-threshold 0.2 --vision-model gpt-4o
```

### Ingestion (FAISS)
```bash
# Fast
uv run scripts/ingest_to_faiss.py --chunks chunks/YOUR-CHUNKS-FILENAME.json --index YOUR-FAISS-INDICES-NAME

# Advanced
uv run scripts/ingest_to_faiss.py --chunks chunks/YOUR-CHUNKS-FILENAME.json --index YOUR-FAISS-INDICES-NAME --embedding-model text-embedding-3-small --index-type HNSW
```

### Ingestion (Milvus)
```bash
# Fast
uv run scripts/ingest_to_milvus.py --chunks chunks/YOUR-CHUNKS-FILENAME.json --collection [YOUR-COLLECTION-NAME]

# Advanced
uv run scripts/ingest_to_milvus.py --chunks chunks/YOUR-CHUNKS-FILENAME.json --collection YOUR-COLLECTION-NAME --embedding-model text-embedding-3-small --index-type HNSW --host localhost --port 19530
```

### Querying

- Auto detects active vector DB with default ingested locations (default faiss pkl files/milvus db configs) if unspecified

```bash
# Fast: single query
uv run scripts/query_vectordb.py --query "What is covered by insurance?" --save-results evaluation/query_results/[YOUR-RESULTS-NAME].json

# Fast: multiple queries
uv run scripts/query_vectordb.py --query "query 1" "query 2" "query 3" --save-results evaluation/query_results/[YOUR-RESULTS-NAME].json

# Fast: from file
uv run scripts/query_vectordb.py --queries-file evaluation/queries.json --save-results evaluation/query_results/output.json

# Advanced
uv run scripts/query_vectordb.py --query "What is covered?" --top-k 10 --vector-db faiss --embedding-model text-embedding-3-small
```

---

## Path Resolution

### Relative Paths
All paths support **relative paths from `data/`** directory:

```bash
# These are equivalent:
--input raw-documents
--input data/raw-documents
--input /absolute/path/to/raw-documents

# These are equivalent:
--chunks chunks/output.json
--chunks data/chunks/output.json
--chunks /absolute/path/to/chunks/output.json
```

### Standard Directory Structure
```
data/
├── raw-documents/          # Input documents
├── chunks/                 # Chunked output
├── faiss_indices/          # FAISS indices
└── evaluation/
    ├── ground_truth/       # Ground truth files
    ├── query_results/      # Query results
    └── eval_results/       # Evaluation results
```

---

## Detailed Usage

### 1. run_chunking.py

Chunk documents into processable pieces.

**Fast Run:**
```bash
uv run scripts/run_chunking.py \
  --input raw-documents \
  --output chunks/my_chunks.json
```

**Advanced Run:**
```bash
uv run scripts/run_chunking.py \
  --input raw-documents \
  --output chunks/my_chunks.json \
  --image-threshold 0.25 \
  --vision-model gpt-4o \
  --log-level DEBUG
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input` | path | `INPUT_DIR` from config | Directory with documents |
| `--output` | path | `data/chunks/<experiment>_chunks.json` | Output JSON file |
| `--image-threshold` | float | From config (0.15) | Vision trigger threshold (0.0-1.0) |
| `--vision-model` | choice | From config (gpt-4o) | Vision model: gpt-4o, gpt-4-vision-preview |
| `--log-level` | choice | From config (INFO) | DEBUG, INFO, WARNING, ERROR |

**Example Workflows:**

```bash
# Process specific document folder
uv run scripts/run_chunking.py --input contracts/2024 --output chunks/contracts_2024.json

# Reduce vision processing (lower cost)
uv run scripts/run_chunking.py --input documents --image-threshold 0.3

# Debug mode
uv run scripts/run_chunking.py --input documents --log-level DEBUG
```

---

### 2. ingest_to_faiss.py

Ingest chunks to FAISS vector database.

**Fast Run:**
```bash
uv run scripts/ingest_to_faiss.py \
  --chunks chunks/my_chunks.json \
  --index my_experiment
```

**Advanced Run:**
```bash
uv run scripts/ingest_to_faiss.py \
  --chunks chunks/my_chunks.json \
  --index my_experiment \
  --embedding-provider openai \
  --embedding-model text-embedding-3-small \
  --batch-size 32 \
  --index-type HNSW \
  --metric-type IP \
  --normalize \
  --use-gpu
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--chunks` | path | From config | Chunks JSON file |
| `--index` | str | From config | Index name |
| `--drop-existing` | flag | False | Drop existing index |
| `--embedding-provider` | choice | From config | openai, sentence_transformers |
| `--embedding-model` | str | From config | Model name |
| `--batch-size` | int | From config | Embedding batch size |
| `--index-type` | choice | From config (Flat) | Flat, IVF, HNSW |
| `--metric-type` | choice | From config (IP) | IP, L2 |
| `--normalize` | flag | From config | Normalize for cosine similarity |
| `--use-gpu` | flag | From config (False) | Use GPU acceleration |

**Example Workflows:**

```bash
# Quick test with small model
uv run scripts/ingest_to_faiss.py --chunks chunks/test.json --index test --embedding-model text-embedding-3-small

# Production with HNSW
uv run scripts/ingest_to_faiss.py --chunks chunks/prod.json --index prod --index-type HNSW --normalize

# GPU-accelerated
uv run scripts/ingest_to_faiss.py --chunks chunks/large.json --index large --use-gpu --index-type IVF

# Replace existing index
uv run scripts/ingest_to_faiss.py --chunks chunks/updated.json --index my_index --drop-existing
```

---

### 3. ingest_to_milvus.py

Ingest chunks to Milvus vector database.

**Fast Run:**
```bash
uv run scripts/ingest_to_milvus.py \
  --chunks chunks/my_chunks.json \
  --collection my_collection
```

**Advanced Run:**
```bash
uv run scripts/ingest_to_milvus.py \
  --chunks chunks/my_chunks.json \
  --collection my_collection \
  --host localhost \
  --port 19530 \
  --embedding-provider openai \
  --embedding-model text-embedding-3-large \
  --index-type HNSW \
  --metric-type IP
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--chunks` | path | From config | Chunks JSON file |
| `--collection` | str | From config | Collection name |
| `--drop-existing` | flag | False | Drop existing collection |
| `--host` | str | From config (localhost) | Milvus host |
| `--port` | str | From config (19530) | Milvus port |
| `--embedding-provider` | choice | From config | openai, sentence_transformers |
| `--embedding-model` | str | From config | Model name |
| `--batch-size` | int | From config | Embedding batch size |
| `--index-type` | choice | From config | IVF_FLAT, HNSW, IVF_PQ |
| `--metric-type` | choice | From config (IP) | IP, L2, COSINE |

**Example Workflows:**

```bash
# Connect to remote Milvus
uv run scripts/ingest_to_milvus.py --chunks chunks/data.json --collection prod --host milvus.example.com --port 19530

# Use high-quality embeddings
uv run scripts/ingest_to_milvus.py --chunks chunks/data.json --collection quality --embedding-model text-embedding-3-large

# Production-optimized index
uv run scripts/ingest_to_milvus.py --chunks chunks/large.json --collection prod --index-type HNSW --metric-type IP
```

---

### 4. query_vectordb.py

Query vector database (automatically detects FAISS or Milvus from config).

**Fast Run - Single Query:**
```bash
uv run scripts/query_vectordb.py --query "What is covered by insurance?"
```

**Fast Run - Multiple Queries:**
```bash
uv run scripts/query_vectordb.py --query "query 1" "query 2" "query 3"
```

**Fast Run - From File:**
```bash
uv run scripts/query_vectordb.py \
  --queries-file evaluation/queries.json \
  --save-results evaluation/query_results/experiment1.json
```

**Advanced Run:**
```bash
uv run scripts/query_vectordb.py \
  --query "What is covered by insurance?" \
  --vector-db faiss \
  --top-k 10 \
  --embedding-provider openai \
  --embedding-model text-embedding-3-large \
  --save-results evaluation/query_results/test.json
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--query` | str+ | Example queries | Query string(s) |
| `--queries-file` | path | None | JSON file with queries |
| `--save-results` | path | None | Save results for evaluation |
| `--vector-db` | choice | From config | faiss, milvus |
| `--top-k` | int | From config (5) | Number of results |
| `--embedding-provider` | choice | From config | openai, sentence_transformers |
| `--embedding-model` | str | From config | Model name |
| `--faiss-index` | str | From config | FAISS index name |
| `--milvus-collection` | str | From config | Milvus collection name |
| `--milvus-host` | str | From config | Milvus host |
| `--milvus-port` | str | From config | Milvus port |

**Queries File Format:**
```json
{
  "queries": [
    {
      "query_id": "q1",
      "query_text": "What is covered by insurance?"
    },
    {
      "query_id": "q2",
      "query_text": "Which hospitals are in-network?"
    }
  ]
}
```

**Example Workflows:**

```bash
# Quick test query
uv run scripts/query_vectordb.py --query "test query"

# Compare FAISS vs Milvus
uv run scripts/query_vectordb.py --query "test" --vector-db faiss --save-results evaluation/query_results/faiss.json
uv run scripts/query_vectordb.py --query "test" --vector-db milvus --save-results evaluation/query_results/milvus.json

# Batch evaluation queries
uv run scripts/query_vectordb.py --queries-file evaluation/queries.json --save-results evaluation/query_results/batch1.json

# More results
uv run scripts/query_vectordb.py --query "comprehensive query" --top-k 20
```

---

## Complete Workflows

### Workflow 1: Basic Pipeline

```bash
# 1. Chunk documents
uv run scripts/run_chunking.py \
  --input raw-documents \
  --output chunks/experiment1.json

# 2. Ingest to FAISS
uv run scripts/ingest_to_faiss.py \
  --chunks chunks/experiment1.json \
  --index experiment1

# 3. Query
uv run scripts/query_vectordb.py \
  --query "What is the policy?"
```

### Workflow 2: Compare Embeddings

```bash
# Chunk once
uv run scripts/run_chunking.py --input documents --output chunks/data.json

# Test with small model
uv run scripts/ingest_to_faiss.py \
  --chunks chunks/data.json \
  --index test_small \
  --embedding-model text-embedding-3-small

# Test with large model
uv run scripts/ingest_to_faiss.py \
  --chunks chunks/data.json \
  --index test_large \
  --embedding-model text-embedding-3-large

# Query both
uv run scripts/query_vectordb.py --query "test" --faiss-index test_small
uv run scripts/query_vectordb.py --query "test" --faiss-index test_large
```

### Workflow 3: Evaluation Pipeline

```bash
# 1. Chunk
uv run scripts/run_chunking.py --input documents --output chunks/eval.json

# 2. Ingest
uv run scripts/ingest_to_faiss.py --chunks chunks/eval.json --index eval

# 3. Query and save results
uv run scripts/query_vectordb.py \
  --queries-file evaluation/ground_truth/queries.json \
  --save-results evaluation/query_results/experiment1.json

# 4. Evaluate (see EVALUATION.md)
uv run scripts/evaluate_retrieval.py \
  --results evaluation/query_results/experiment1.json \
  --ground-truth evaluation/ground_truth/gt.json
```

---

## Tips and Best Practices

### 1. Use Relative Paths

Always use relative paths from `data/`:
```bash
# Good
--input raw-documents --output chunks/output.json

# Also works, but longer
--input data/raw-documents --output data/chunks/output.json
```

### 2. Organize by Experiment

```bash
# Organize outputs by experiment name
--output chunks/experiment1_chunks.json
--index experiment1
--save-results evaluation/query_results/experiment1_results.json
```

### 3. Config File as Default

Leave most hyperparameters in config file:
```bash
# Minimal CLI - uses config defaults
uv run scripts/ingest_to_faiss.py --chunks chunks/data.json --index myindex

# Only override when testing
uv run scripts/ingest_to_faiss.py --chunks chunks/data.json --index test --batch-size 16
```

### 4. Use --help

Every script has built-in help:
```bash
uv run scripts/run_chunking.py --help
uv run scripts/ingest_to_faiss.py --help
uv run scripts/query_vectordb.py --help
```

### 5. Incremental Development

```bash
# Start small
uv run scripts/run_chunking.py --input test_docs --output chunks/test.json
uv run scripts/ingest_to_faiss.py --chunks chunks/test.json --index test

# Iterate quickly
uv run scripts/query_vectordb.py --query "test"

# Scale up when ready
uv run scripts/run_chunking.py --input all_documents --output chunks/production.json
```

---

## Error Handling

### File Not Found

```bash
# Check path
ls data/chunks/

# Use absolute path if needed
--chunks /absolute/path/to/chunks.json
```

### Config Errors

```bash
# Override config with CLI
--embedding-model text-embedding-3-small

# Check config file
cat config.py | grep EMBEDDING_CONFIG
```

### Connection Errors (Milvus)

```bash
# Check Milvus is running
docker ps | grep milvus

# Override connection
--host localhost --port 19530
```

---

## Next Steps

- See [Evaluation Guide](EVALUATION.md) for evaluation CLI
- See [Configuration Guide](CONFIGURATION.md) for config file defaults
- See [Troubleshooting](TROUBLESHOOTING.md) for common issues