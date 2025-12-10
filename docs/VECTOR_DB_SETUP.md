# Vector Database Setup Guide

This guide covers setup and configuration for supported vector databases.

## Supported Databases

| Database | Best For | Setup Required |
|----------|----------|----------------|
| **FAISS** | Development, experimentation, small-medium datasets | No |
| **Milvus** | Production, large-scale datasets, distributed deployments | Yes (Docker) |

---

## FAISS Setup

### Installation

```bash
# For CPU support
uv add faiss-cpu

# For GPU support (requires CUDA)
uv add faiss-gpu
```

### Configuration

Edit `config.py`:

```python
ACTIVE_VECTOR_DB = "faiss"

FAISS_CONFIG = {
    "index": {
        "index_dir": "data/faiss_indices",
        "name": "document_embeddings",
        "index_type": "Flat",      # Options: Flat, IVF, HNSW
        "metric_type": "IP",       # Options: IP (inner product), L2
        "normalize": True,         # True for cosine similarity with IP
        "use_gpu": False,          # Set True if using faiss-gpu
        "gpu_id": 0,              # GPU device ID
        "params": {}
    },
    "search": {
        "top_k": 5,
        "params": {}
    }
}
```

### Index Types

**Flat (Exact Search)**
- Most accurate but slower for large datasets
- Good for: < 1M vectors
```python
"index_type": "Flat",
"params": {}
```

**IVF (Inverted File Index)**
- Faster search with slight accuracy tradeoff
- Good for: 1M+ vectors
```python
"index_type": "IVF",
"params": {"nlist": 100}  # Number of clusters
```

Search params:
```python
"search": {
    "params": {"nprobe": 10}  # Number of clusters to search
}
```

**HNSW (Hierarchical Navigable Small World)**
- Best speed/accuracy tradeoff
- Good for: Production use
```python
"index_type": "HNSW",
"params": {"M": 32}  # Number of connections
```

### Usage

```bash
# Ingest
uv run scripts/ingest_to_faiss.py

# Query
uv run scripts/query_vector_db.py
```

---

## Milvus Setup

### Step 1: Start Milvus

```bash
# Start services
docker-compose -f docker-compose-milvus-standalone.yml up -d

# Verify health
curl http://localhost:9091/healthz

# Check container status
docker-compose -f docker-compose-milvus-standalone.yml ps
```

### Step 2: Configuration

Edit `config.py`:

```python
ACTIVE_VECTOR_DB = "milvus"

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
        "index_type": "IVF_FLAT",  # Options: HNSW, IVF_FLAT, IVF_PQ
        "metric_type": "IP",        # Options: IP, L2, COSINE
        "params": {"nlist": 1024}
    },
    "search": {
        "top_k": 5,
        "params": {}  # e.g., {"nprobe": 10} for IVF
    }
}
```

### Index Types

- **HNSW**: Best for accuracy, slower build time
- **IVF_FLAT**: Balanced performance
- **IVF_PQ**: Fastest, uses quantization (slight accuracy loss)

### Usage

```bash
# Ingest
uv run scripts/ingest_to_milvus.py

# Query
uv run scripts/query_vector_db.py
```

### Stop Milvus

```bash
docker-compose -f docker-compose-milvus-standalone.yml down
```

---

## Comparison

| Feature | FAISS | Milvus |
|---------|-------|--------|
| **Setup** | No setup (file-based) | Docker required |
| **Deployment** | Single machine | Distributed capable |
| **Best For** | Development, small-medium datasets | Production, large datasets |
| **Persistence** | File system | Database with backup |
| **Scalability** | Limited to single machine | Horizontal scaling |
| **Index Types** | Flat, IVF, HNSW | IVF_FLAT, HNSW, IVF_PQ, more |
| **GPU Support** | Yes (with faiss-gpu) | Yes |
| **Memory** | Entire index in RAM | Configurable |
| **Ease of Use** | Very easy | Moderate (requires Docker) |

---

## Migration Between Databases

### From Milvus to FAISS

```bash
# 1. Update config
# ACTIVE_VECTOR_DB = "faiss"

# 2. Re-run ingestion
uv run scripts/ingest_to_faiss.py
```

### From FAISS to Milvus

```bash
# 1. Start Milvus
docker-compose -f docker-compose-milvus-standalone.yml up -d

# 2. Update config
# ACTIVE_VECTOR_DB = "milvus"

# 3. Re-run ingestion
uv run scripts/ingest_to_milvus.py
```

**Note**: Chunks are cached, so re-ingestion only re-embeds and indexes.

---

## Performance Recommendations

### Small Datasets (<100K vectors)
**FAISS:**
```python
"index_type": "Flat",
"metric_type": "IP",
"normalize": True
```

### Medium Datasets (100K-1M vectors)
**FAISS:**
```python
"index_type": "IVF",
"params": {"nlist": 100}
```

**Milvus:**
```python
"index_type": "IVF_FLAT",
"params": {"nlist": 1024}
```

### Large Datasets (1M+ vectors)
**FAISS:**
```python
"index_type": "HNSW",
"params": {"M": 32},
"use_gpu": True  # If available
```

**Milvus:**
```python
"index_type": "HNSW",
"params": {"M": 16, "efConstruction": 200}
```

---

## Troubleshooting

### FAISS Issues

**Installation Failed:**
```bash
# Try specific version
uv add "faiss-cpu==1.7.4"

# Or use conda
conda install -c conda-forge faiss-cpu
```

**Memory Errors:**
- Use IVF or HNSW index instead of Flat
- Reduce batch size during ingestion
- For very large datasets, consider Milvus

**GPU Not Working:**
```bash
# Ensure CUDA is installed
nvidia-smi

# Install faiss-gpu
uv add faiss-gpu

# Enable in config
FAISS_CONFIG["index"]["use_gpu"] = True
```

### Milvus Issues

**Connection Failed:**
```bash
# Check containers
docker-compose -f docker-compose-milvus-standalone.yml ps

# View logs
docker-compose -f docker-compose-milvus-standalone.yml logs milvus-standalone

# Restart
docker-compose -f docker-compose-milvus-standalone.yml restart
```

**Port Already in Use:**
```bash
# Check what's using port 19530
lsof -i :19530  # Mac/Linux
netstat -ano | findstr :19530  # Windows

# Either stop the other service or change port in docker-compose
```

**Data Persistence Issues:**
```bash
# Reset Milvus (WARNING: deletes all data)
docker-compose -f docker-compose-milvus-standalone.yml down -v
docker-compose -f docker-compose-milvus-standalone.yml up -d
```

---