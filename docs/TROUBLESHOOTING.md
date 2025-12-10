# Troubleshooting Guide

Common issues and solutions for SemantIQ.

---

## Installation Issues

### uv Not Found

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### Dependency Conflicts

```bash
# Clear cache and reinstall
rm -rf .venv
uv sync
```

### Python Version Issues

```bash
# Check Python version
python --version  # Should be >= 3.12

# Use specific Python version
uv venv --python 3.12
uv sync
```

---

## Chunking Issues

### PDF Processing Fails

**Error**: `PDFProcessingError: Could not extract text`

**Solutions:**
1. Check if PDF is corrupted
2. Try with a different PDF
3. Update PyMuPDF: `uv add --upgrade pymupdf`

### Vision Processing Fails

**Error**: `OpenAI API error: Invalid API key`

**Solutions:**
1. Check `.env` file has correct `OPENAI_API_KEY`
2. Verify API key has access to GPT-4o
3. Check rate limits

**Error**: `Rate limit exceeded`

**Solutions:**
1. Increase `image_coverage_threshold` to process fewer pages
2. Add delays between requests
3. Upgrade API tier

### Out of Memory During Chunking

**Solutions:**
1. Process fewer files at once
2. Reduce batch sizes
3. Increase system memory

---

## Embedding Issues

### OpenAI API Errors

**Error**: `AuthenticationError: Incorrect API key`

**Solutions:**
```bash
# Check .env file
cat .env | grep OPENAI_API_KEY

# Verify key is loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('OPENAI_API_KEY'))"
```

**Error**: `Rate limit exceeded`

**Solutions:**
1. Reduce `batch_size` in config
2. Add retry logic
3. Upgrade API tier

### Sentence Transformers Issues

**Error**: `Model not found`

**Solutions:**
```bash
# Install model manually
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('model-name')"
```

**Error**: `CUDA out of memory`

**Solutions:**
1. Reduce `batch_size`
2. Use CPU: set device to 'cpu'
3. Use smaller model

### Out of Memory During Embedding

**Solutions:**
```python
# Reduce batch size in config.py
EMBEDDING_CONFIG = {
    "text": {
        "openai": {
            "batch_size": 16  # Reduce from 64
        }
    }
}
```

---

## Vector Database Issues

### FAISS Issues

**Installation Failed**

```bash
# Try specific version
uv add "faiss-cpu==1.7.4"

# Or use conda
conda install -c conda-forge faiss-cpu
```

**GPU Not Working**

```bash
# Check CUDA
nvidia-smi

# Install GPU version
uv add faiss-gpu

# Enable in config
FAISS_CONFIG["index"]["use_gpu"] = True
```

**Index File Not Found**

**Solutions:**
1. Run ingestion first: `uv run scripts/ingest_to_faiss.py`
2. Check `index_dir` path in config
3. Verify file permissions

**Memory Errors**

**Solutions:**
1. Use IVF or HNSW instead of Flat index
2. Reduce batch size during ingestion
3. For large datasets, use Milvus

### Milvus Issues

**Connection Failed**

```bash
# Check if Milvus is running
docker ps | grep milvus

# Check all containers are healthy
docker-compose -f docker-compose-milvus-standalone.yml ps

# View logs
docker-compose -f docker-compose-milvus-standalone.yml logs milvus-standalone

# Restart Milvus
docker-compose -f docker-compose-milvus-standalone.yml restart
```

**Port Already in Use**

```bash
# Find what's using the port
# Linux/Mac
lsof -i :19530

# Windows
netstat -ano | findstr :19530

# Kill the process or change Milvus port in docker-compose
```

**Container Won't Start**

**Solutions:**
1. Check Docker is running
2. Check disk space: `df -h`
3. Check Docker logs: `docker logs milvus-standalone`
4. Reset Milvus (WARNING: deletes data):
```bash
docker-compose -f docker-compose-milvus-standalone.yml down -v
docker-compose -f docker-compose-milvus-standalone.yml up -d
```

**Data Not Persisting**

**Solutions:**
1. Check volume mounts in docker-compose file
2. Ensure `DOCKER_VOLUME_DIRECTORY` is set or use default (.)
3. Check directory permissions

**Collection Not Found**

**Solutions:**
1. Run ingestion first
2. Check collection name matches in config
3. Verify Milvus is running

---

## Query Issues

### No Results Returned

**Causes:**
1. Index/collection is empty
2. Query embedding dimension mismatch
3. Wrong metric type

**Solutions:**
```bash
# Verify ingestion
uv run scripts/ingest_to_faiss.py --verify

# Check collection stats
# Add logging to see index size
```

### Wrong Results

**Possible causes:**
1. Embedding model mismatch between ingestion and query
2. Normalization inconsistency
3. Wrong metric type

**Solutions:**
1. Use same embedding model for ingestion and query
2. Check `normalize` setting is consistent
3. Verify `metric_type` matches use case (IP for cosine, L2 for euclidean)

### Slow Query Performance

**Solutions:**

**FAISS:**
```python
# Use faster index
"index_type": "IVF",  # or "HNSW"
"params": {"nlist": 100}

# Reduce search clusters
"search": {"params": {"nprobe": 5}}
```

**Milvus:**
```python
# Use faster index
"index_type": "IVF_FLAT",

# Reduce search scope
"search": {"params": {"nprobe": 10}}
```

---

## Evaluation Issues

### No Ground Truth Found

**Error**: `No ground truth for query q1, skipping`

**Solutions:**
1. Verify ground truth file exists
2. Check query IDs match exactly (case-sensitive)
3. Ensure query IDs are strings

### Document ID Mismatch

**Error**: `No relevant IDs for query q1`

**Solutions:**
1. Check document IDs in ground truth match query results
2. Verify ID format (string vs int)
3. Print IDs from both sources to compare:
```python
print("Ground truth IDs:", gt_manager.get_relevant_docs("q1"))
print("Query result IDs:", [r["id"] for r in results])
```

### All Metrics Are Zero

**Causes:**
1. No relevant documents retrieved
2. ID mismatch
3. Empty ground truth

**Solutions:**
1. Verify ground truth has correct IDs
2. Check vector database has documents
3. Test with simple query you know should work

### TypeError in Metrics

**Error**: `TypeError: the resolved dtypes are not compatible`

**Solutions:**
This is fixed in the latest version. Update `evaluation/metrics.py` to filter out non-numeric fields.

---

## Logging and Debugging

### Enable Debug Logging

```python
# In config.py
LOGGING_CONFIG = {
    "handlers": {
        "console": {"level": "DEBUG"},  # Change from INFO
        "file": {"level": "DEBUG"}
    }
}
```

### View Logs

```bash
# Tail log file
tail -f logs/rag_pipeline.log

# View last 100 lines
tail -n 100 logs/rag_pipeline.log

# Search logs
grep "ERROR" logs/rag_pipeline.log
```

### Python Debugging

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use ipdb
import ipdb; ipdb.set_trace()
```

---

## Performance Issues

### Slow Chunking

**Solutions:**
1. Reduce `image_coverage_threshold` to process fewer pages with vision
2. Process files in parallel (implement multiprocessing)
3. Use faster vision model or skip vision for non-image-heavy docs

### Slow Embedding

**Solutions:**
1. Increase `batch_size` (if not hitting rate limits)
2. Use smaller embedding model
3. Use local model (Sentence Transformers) instead of API
4. Implement caching for repeated texts

### Slow Indexing

**FAISS:**
1. Use Flat index for small datasets (< 100K)
2. Use IVF for medium datasets
3. Use GPU if available

**Milvus:**
1. Increase `nlist` for IVF indices
2. Use IVF_PQ for very large datasets
3. Tune batch size during ingestion

### Slow Search

**Solutions:**
1. Reduce `top_k`
2. Use approximate search (IVF, HNSW)
3. Reduce `nprobe` for IVF indices
4. Enable GPU if available

---

## Docker Issues

### Docker Not Running

```bash
# Start Docker
# Mac: Open Docker Desktop
# Linux: sudo systemctl start docker
# Windows: Start Docker Desktop
```

### Docker Out of Disk Space

```bash
# Clean up
docker system prune -a

# Remove unused volumes
docker volume prune
```

### Container Permissions

```bash
# Fix permissions (Linux)
sudo chown -R $USER:$USER volumes/
```

---

## Common Error Messages

### `ModuleNotFoundError: No module named 'X'`

**Solution:**
```bash
uv sync
```

### `FileNotFoundError: [Errno 2] No such file or directory`

**Solutions:**
1. Check path in error message
2. Create missing directories
3. Run from project root

### `AttributeError: 'NoneType' object has no attribute 'X'`

**Solutions:**
1. Check configuration is loaded
2. Verify objects are initialized
3. Add null checks

---

## Getting Help

If you're still stuck:

1. **Check logs**: `logs/rag_pipeline.log`
2. **Enable debug logging**: Set console level to DEBUG
3. **Search issues**: Check GitHub issues
4. **Create issue**: Include:
   - Error message
   - Full traceback
   - Configuration (sanitized)
   - Steps to reproduce
   - Python version
   - OS version

---

## Prevention

### Best Practices

1. **Version Control**: Commit working configurations
2. **Test Small**: Test with small datasets first
3. **Monitor Logs**: Watch logs during processing
4. **Resource Monitoring**: Monitor CPU, RAM, disk usage
5. **Backups**: Backup indices and ground truth
6. **Documentation**: Document configuration changes

### Health Checks

```bash
# Check Python version
python --version

# Check dependencies
uv pip list

# Check Docker (if using Milvus)
docker ps

# Check disk space
df -h

# Check memory
free -h  # Linux
vm_stat  # Mac
```