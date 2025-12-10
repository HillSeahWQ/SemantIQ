# SemantIQ — A Framework for Multimodal Semantic Retrieval Experiments

## Table of Contents
1. [Overview](#1-overview)  
2. [Key Features](#2-key-features)  
3. [Goal](#3-goal)  
4. [Current Implementations](#4-current-implementations)  
   - [4.1 File Types](#41-file-types)  
   - [4.2 Embedding Models](#42-embedding-models)  
   - [4.3 Vector Database Support](#43-vector-database-support)  
   - [4.4 Retrieval Evaluation Metrics](#44-retrieval-evaluation-metrics)  
5. [Quick Start](#5-quick-start)  
   - [5.0 Prerequisites](#50-prerequisites)  
   - [5.1 Installation](#51-installation)  
   - [5.2 Start Milvus](#52-start-milvus-if-vector-db-choice--milvus)  
   - [5.3 Configure Pipeline](#53-configure-pipeline)  
   - [5.4 Run Ingestion Pipeline](#54-run-ingestion-pipeline)  
   - [5.5 Query Your Data](#55-query-your-data)  
   - [5.6 Detailed Evaluation with Metrics](#56-detailed-evaluation-with-metrics)  
6. [Project Structure](#6-project-structure)  
7. [Documentation](#7-documentation)  
8. [Use Cases](#8-use-cases)  
   - [8.1 Compare Embedding Models](#81-compare-embedding-models)  
   - [8.2 Compare Vector Databases](#82-compare-vector-databases)  
   - [8.3 Evaluate Retrieval Quality](#83-evaluate-retrieval-quality)  
   - [8.4 Extensions](#84-extensions)  
9. [Extending the Pipeline](#9-extending-the-pipeline)  
10. [Troubleshooting](#10-troubleshooting)  
11. [Performance Tips](#11-performance-tips)  
12. [Support](#12-support)

---

## 1. Overview
**SemantIQ** is a modular and extensible framework designed to benchmark and optimize **semantic retrieval** of systems like RAG, Recommendation Systems, ... etc.

It enables rapid experimentation and evaluation across all stages of the retrieval pipeline — from **document loading and multimodal chunking**, to **embedding generation** and **vector database indexing**.

---

## 2. Key Features

- **Flexible Document Ingestion** — Supports multiple file types (PDF, DOCX, code files, images, tables, etc.) with customizable chunking strategies.  
- **Pluggable Embedding Models** — Swap between text and multimodal embedding models for comparative evaluation.  
- **Configurable Vector Stores** — Test across FAISS, Milvus, Chroma, and others with adjustable indexing strategies and hyperparameters.  
- **Evaluation & Logging** — Built-in tools for reproducible experiments, retrieval quality metrics, and performance tracking.  

---

## 3. Goal
To systematically study how **chunking**, **embedding**, and **vector indexing** choices affect semantic retrieval quality and efficiency in RAG pipelines.

---

## 4. Current Implementations

### 4.1 File Types

**PDF**
- Multimodal: Text, Images, Tables  
- Vision-Enhanced Chunking: Automatic vision processing for image-heavy pages (based on area threshold)  
- Table Extraction: Preserves tables in Markdown format  
- Chunking by page  

*Extensible: Easily add DOCX, code files, HTML, and more.*

---

### 4.2 Embedding Models

**OpenAI Embeddings**  
- `text-embedding-3-large` / `text-embedding-3-small`

**Sentence Transformers (HuggingFace)**  
- Compatible with a variety of transformer-based models  

*Extensible: Add code or domain-specific embeddings.*

---

### 4.3 Vector Database Support

**Milvus**  
- Distributed vector database with production-ready features  
- Supports multiple index types (HNSW, IVF_FLAT, IVF_PQ)  
- Requires Docker setup (see below)  
- Best for: Production deployments, large-scale datasets  

**FAISS (Facebook AI Similarity Search)**  
- File-based local vector storage  
- CPU and GPU support available  
- Multiple index types (Flat, IVF, HNSW)  
- Best for: Development, experimentation, small-to-medium datasets  
- No server setup required  

*Extensible: Add Chroma, Pinecone, Weaviate, or other vector databases with configurable parameters.*

---

### 4.4 Retrieval Evaluation Metrics
- @K: Precision, Recall, F1, Hit Rate, nDCG

---

## 5. Quick Start

### 5.0 Prerequisites

Before getting started, ensure you have the following installed:

| Requirement | Version | Notes |
|--------------|----------|--------|
| **Python** | ≥ 3.12 | Recommended to use a virtual environment |
| **uv** | Latest | Fast Python package manager ([uv documentation](https://docs.astral.sh/uv/)) |
| **Docker** | Latest | Required for running local vector databases or external services |

---

### 5.1 Installation

```bash
# Clone repository
git clone <repo>
cd SemantIQ

# Install dependencies
uv sync

# Setup environment variables
cat > .env << EOF
OPENAI_API_KEY=your_openai_key_here
EOF
```

---

### 5.2 Start Milvus (If Vector DB choice = Milvus)

```bash
# Using Docker Compose (recommended)
# Download docker-compose.yml from Milvus documentation
docker-compose -f docker-compose-milvus-standalone.yml up -d

# Verify Milvus is running
curl http://localhost:19530/healthz
```

---

### 5.3 Configure Data Directory

Place your documents folder (`your-documents-folder-name`), containing all data files (PDFs, Markdown, code, docs, etc.), inside a `YOUR_INPUT_FOLDER_NAME` in the top-level `data` directory in the root of this repository.

Example structure:
SemantiQ/data/your-documents-folder-name

---

### 5.4 Configure Pipeline 

Edit `config.py`:

```python
# Set your input directory to documents folder
INPUT_FOLDER_NAME = "YOUR_INPUT_FOLDER_NAME"

# Choose embedding model - provider to use
ACTIVE_EMBEDDING_PROVIDER = "openai"  # or "sentence_transformers", "code_transformers"

# Choose desired supported vector database
ACTIVE_VECTOR_DB = "faiss"  # Options: "milvus", "faiss"
```

---

### 5.5 Run Ingestion Pipeline
- See **[CLI commands Guide](docs/CLI.md)** on details for more advanced configurable runs

```bash
# Step 1: Chunk documents (run once)
uv run scripts/run_chunking.py --input data/[YOUR-RAW-DOCUMENTS-FOLDER] --output chunks/[YOUR-CHUNKS-FILENAME].json

# Step 2: Embed and ingest to your chosen vector database

# For FAISS:
uv run scripts/ingest_to_faiss.py --chunks chunks/[YOUR-CHUNKS-FILENAME].json --index [YOUR-FAISS-INDICES-NAME] --embedding-type [YOUR-EMBEDDING-TYPE] --embedding-provider [YOUR-EMBEDDING-PROVIDER] --embedding-model [YOUR-EMBEDDING-MODEL-CHOICE-FROM-EMBEDDING-PROVIDER]

# For Milvus:
uv run scripts/ingest_to_milvus.py --chunks chunks/[YOUR-CHUNKS-FILENAME].json --collection [YOUR-COLLECTION-NAME] --embedding-type [YOUR-EMBEDDING-TYPE] --embedding-provider [YOUR-EMBEDDING-PROVIDER] --embedding-model [YOUR-EMBEDDING-MODEL-CHOICE-FROM-EMBEDDING-PROVIDER]
```

1. Decide what data types (example .doc, .pdf are text files; .py, .java are code files) 
   --embedding-type examples: text, code, ...
2. Decide what embedding model provider to use for the specific data types
   --embedding-provider examples: openai, sentence_transformers, ...
3. Decide what specifc embedding model from the provider to use
   --embedding-model examples: for openai: , for sentence_transformers: , ...

e.g. for .doc, .docx, .pdf chunks, if we have openai implemented embedding models we do: 
```bash
uv run scripts/ingest_to_faiss.py --chunks chunks/[YOUR-CHUNKS-FILENAME].json --index [YOUR-FAISS-INDICES-NAME] --embedding-type text --embedding-provider openai --embedding-model text-embedding-3-large
```

e.g. for .py, .java, .rs code chunks, if we have sentence transformers implemented embedding models we do:
```bash
uv run scripts/ingest_to_faiss.py --chunks chunks/[YOUR-CHUNKS-FILENAME].json --index [YOUR-FAISS-INDICES-NAME] --embedding-type code --embedding-provider code_transformers --embedding-model microsoft/codebert-base
```
---

### 5.6 Query Your Data

- Auto detects active vector DB with default ingested locations (default faiss pkl files/milvus db configs) if unspecified

```bash
# Run example queries - (automatically uses ACTIVE_VECTOR_DB)
uv run scripts/query_vectordb.py --query "query 1" "query 2" "query 3"  --save-results evaluation/query_results/[YOUR-RESULTS-NAME].json
```

Or programmatically:

```python
from scripts.query_vector_db import query_vector_db

results = query_vector_db(
    queries=["What is covered by insurance?"],
    top_k=5
)
```

---

### 5.7 Detailed Evaluation with Metrics

See **[Evaluation Guide](docs/EVALUATION.md)** - How to run evaluation of retrieval quality with metrics (@K - Precision, Recall, F1, Hit Rate, nDCG).  
Ensure queries + ground truth JSON file are created.

---

## 6. Project Structure

```
SemantIQ/
├── config.py
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   └── storage.py
├── docker-compose-milvus-standalone.yml
├── chunking/
│   ├── base.py
│   └── pdf_chunker.py
├── embedding/
│   └── embedding_manager.py
├── vector_db/
│   ├── milvus_client.py
│   └── faiss_client.py
├── evaluation/
│   ├── metrics.py
│   └── ground_truth/
├── scripts/
│   ├── run_chunking.py
│   ├── ingest_to_milvus.py
│   ├── ingest_to_faiss.py
│   ├── query_vector_db.py
│   ├── evaluate_retrieval.py
│   ├── compare_experiments.py
│   └── create_ground_truth.py
└── data/
    ├── chunks/
    ├── faiss_indices/
    └── evaluation/
        ├── query_results/
        ├── ground_truth/
        └── eval_results/
```

---

## 7. Documentation

| Guide | Description |
|-------|-------------|
| **[Vector Database Setup](docs/VECTOR_DB_SETUP.md)** | Milvus Docker setup, FAISS configuration, comparison |
| **[Configuration Guide](docs/CONFIGURATION.md)** | Chunking, embeddings, vector DB settings |
| **[Evaluation Guide](docs/EVALUATION.md)** | Create ground truth, evaluate results, compare experiments |
| **[Extension Guide](docs/EXTENSION.md)** | Add new embeddings, vector DBs, document types |
| **[Troubleshooting](docs/TROUBLESHOOTING.md)** | Common issues and solutions |

---

## 8. Use Cases

### 8.1 Compare Embedding Models
```bash
uv run scripts/run_chunking.py
uv run scripts/ingest_to_faiss.py
```

### 8.2 Compare Vector Databases
```bash
uv run scripts/ingest_to_faiss.py
uv run scripts/ingest_to_milvus.py
```

### 8.3 Evaluate Retrieval Quality
```bash
uv run scripts/create_ground_truth.py --interactive
uv run scripts/query_vector_db.py --save-results results.json
uv run scripts/evaluate_retrieval.py --results results.json --ground-truth gt.json
```

### 8.4 Extensions
1. Experiment with New Embedding Models  
2. Process New Document Types  
3. Experiment with New Vector Databases  
→ See **[Extension Guide](docs/EXTENSION.md)**

---

## 9. Extending the Pipeline
See **[Extension Guide](docs/EXTENSION.md)** for information on adding new features (support for new Embedding models, Vector DBs, etc).

---

## 10. Troubleshooting
See **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)**

---

## 11. Performance Tips
1. **Chunk Reuse**: Save chunks to avoid re-processing documents  
2. **Batch Size**: Tune embedding batch size for your hardware  
3. **Index Selection**:  
   - HNSW: Best for accuracy, slower build  
   - IVF_FLAT: Balanced  
   - IVF_PQ: Fastest, uses quantization  
4. **Vision Processing**: Only use for truly image-heavy pages (adjust threshold)

---

## 12. Support
For issues or questions:  
- Open an issue on GitHub  
- Review logs in `logs/`
