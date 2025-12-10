# Evaluation System Guide

Complete guide for evaluating retrieval quality in SemantIQ.

## Overview

The evaluation system allows you to:
- Create ground truth annotations for queries
- Evaluate retrieval results with standard IR metrics
- Compare different configurations (embeddings, vector DBs, etc.)
- Track experiment performance over time

## Quick Start

### 1. Create Ground Truth

Ground truth defines which documents are relevant for each query.

**Interactive Mode:**
```bash
uv run scripts/create_ground_truth.py --interactive --output data/evaluation/ground_truth/my_queries.json
```

**From Template:**
```bash
# Create a template with 10 example queries
uv run scripts/create_ground_truth.py --template --num-queries 10 --output data/evaluation/ground_truth/template.json

# Edit the template file, then use it for evaluation
```

**From Existing Queries File:**
```bash
uv run scripts/create_ground_truth.py --from-file queries.json --output data/evaluation/ground_truth/my_queries.json
```

### 2. Run Queries and Save Results

```bash
# Query and save results for evaluation
uv run scripts/query_vector_db.py --save-results data/evaluation/query_results/experiment1.json
```

### 3. Evaluate Results

```bash
uv run scripts/evaluate_retrieval.py \
  --results data/evaluation/query_results/experiment1.json \
  --ground-truth data/evaluation/ground_truth/my_queries.json \
  --output data/evaluation/eval_results/experiment1_eval.json
```

### 4. Compare Experiments

```bash
uv run scripts/compare_experiments.py \
  --experiments data/evaluation/eval_results/exp1_eval.json data/evaluation/eval_results/exp2_eval.json \
  --output data/evaluation/eval_results/comparison.csv
```

---

## Ground Truth Format

Ground truth is stored as JSON with **graded relevance scores**:

```json
{
  "queries": [
    {
      "query_id": "q1",
      "query_text": "What is covered by insurance?",
      "relevant_docs": ["5", "12", "23"],
      "metadata": {
        "category": "insurance",
        "difficulty": "easy"
      }
    }
  ],
  "metadata": {
    "num_queries": 1,
    "format_version": "2.0"
  }
}
```

**Fields:**
- `query_id`: Unique identifier for the query
- `query_text`: The actual query text
- `relevant_docs`: **Dict mapping document IDs to relevance scores**
  - **Higher scores = more relevant**
  - Recommended scale: 0-3 (but any positive scale works)
  - 0 = not relevant, 1 = somewhat relevant, 2 = relevant, 3 = highly relevant
- `metadata`: Optional metadata (category, difficulty, etc.)

**Important:** Document IDs must match the IDs returned by your vector database (the `id` field in query results).

**Backward Compatibility:** The system still supports old binary format with `relevant_doc_ids` list (automatically converted to score 1.0).

---

## Evaluation Metrics

The system computes the following metrics:

### Metrics at K (configurable K values)

**Precision@K**
- Fraction of retrieved documents that are relevant
- Formula: (# relevant in top-K) / K
- Range: 0.0 to 1.0
- Higher is better

**Recall@K**
- Fraction of relevant documents that were retrieved
- Formula: (# relevant in top-K) / (total # relevant)
- Range: 0.0 to 1.0
- Higher is better

**F1@K**
- Harmonic mean of precision and recall
- Formula: 2 * (P * R) / (P + R)
- Range: 0.0 to 1.0
- Higher is better

**Hit Rate@K** (Success@K)
- Whether at least one relevant document was retrieved
- Binary: 1.0 if hit, 0.0 otherwise
- Higher is better

**nDCG@K** (Normalized Discounted Cumulative Gain)
- Rewards relevant documents at higher ranks
- **Uses graded relevance scores** - more relevant docs contribute more
- Accounts for position in ranking
- Formula: DCG@K / IDCG@K where DCG = sum(relevance_score / log2(rank+1))
- Range: 0.0 to 1.0
- Higher is better
- **Best metric for graded relevance**

### Metrics Independent of K

**MRR** (Mean Reciprocal Rank)
- Reciprocal of the rank of the first relevant document
- Formula: 1 / (rank of first relevant item)
- Range: 0.0 to 1.0
- Higher is better

**MAP** (Mean Average Precision)
- Average of precision values at each relevant document position
- Comprehensive quality measure
- Range: 0.0 to 1.0
- Higher is better

---

## Configuration

Edit `config.py` to configure evaluation:

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
    "k_values": [1, 3, 5, 10],  # Compute metrics at these K values
    "default_ground_truth": "data/evaluation/ground_truth/default_queries.json"
}
```

---

## Complete Workflow Example

Let's say you want to compare FAISS vs Milvus with OpenAI embeddings:

### Step 1: Create Ground Truth (Once)

```bash
# Interactive creation
uv run scripts/create_ground_truth.py --interactive --output data/evaluation/ground_truth/insurance_queries.json
```

Example session:
```
--- Query 1 ---
Enter query text: What does insurance cover for surgeries?
Enter query ID (default: q1): q1
Enter relevant document/chunk IDs with relevance scores:
Format: doc_id:score,doc_id:score,...
Example: 5:3.0,12:2.0,23:1.0
Relevance scale: 0-3 (0=not relevant, 1=somewhat, 2=relevant, 3=highly relevant)
IDs with scores: 15:3.0,23:2.5,67:2.0
✓ Added query 'q1' with 3 relevant documents
    - 15: 3.0
    - 23: 2.5
    - 67: 2.0

--- Query 2 ---
Enter query text: Which hospitals are in network?
Enter query ID (default: q2): q2
IDs with scores: 8:3.0,42:3.0,91:2.0,103:1.0
✓ Added query 'q2' with 4 relevant documents
    - 8: 3.0
    - 42: 3.0
    - 91: 2.0
    - 103: 1.0

Enter query text: done
✓ Ground truth saved
```

### Step 2: Run Experiment 1 (FAISS)

```bash
# Set config
# In config.py: ACTIVE_VECTOR_DB = "faiss"

# Query and save results
uv run scripts/query_vector_db.py --save-results data/evaluation/query_results/faiss_openai.json
```

### Step 3: Run Experiment 2 (Milvus)

```bash
# Set config
# In config.py: ACTIVE_VECTOR_DB = "milvus"

# Query and save results
uv run scripts/query_vector_db.py --save-results data/evaluation/query_results/milvus_openai.json
```

### Step 4: Evaluate Both

```bash
# Evaluate FAISS
uv run scripts/evaluate_retrieval.py \
  --results data/evaluation/query_results/faiss_openai.json \
  --ground-truth data/evaluation/ground_truth/insurance_queries.json \
  --output data/evaluation/eval_results/faiss_openai_eval.json

# Evaluate Milvus
uv run scripts/evaluate_retrieval.py \
  --results data/evaluation/query_results/milvus_openai.json \
  --ground-truth data/evaluation/ground_truth/insurance_queries.json \
  --output data/evaluation/eval_results/milvus_openai_eval.json
```

### Step 5: Compare Results

```bash
uv run scripts/compare_experiments.py \
  --experiments \
    data/evaluation/eval_results/faiss_openai_eval.json \
    data/evaluation/eval_results/milvus_openai_eval.json \
  --primary-metric ndcg@5 \
  --output data/evaluation/eval_results/faiss_vs_milvus.csv
```

Output shows comparison table with best values highlighted:
```
EXPERIMENT COMPARISON
================================================================================
Metric                      faiss_openai    milvus_openai
--------------------------------------------------------------------------------
PRECISION:
  precision@1                      0.8500            0.9000*
  precision@3                      0.7333            0.8000*
  precision@5                      0.6400            0.7000*

nDCG:
  ndcg@1                           0.8500            0.9000*
  ndcg@3                           0.7891            0.8456*
  ndcg@5                           0.7234            0.7890*

MRR:                               0.8750            0.9250*
MAP:                               0.8234            0.8756*
================================================================================
* indicates best value for that metric

BEST EXPERIMENT
================================================================================
Name: milvus_openai_eval
Based on: ndcg@5 = 0.7890
```

---

## Advanced Usage

### Custom K Values

```bash
uv run scripts/evaluate_retrieval.py \
  --results results.json \
  --ground-truth gt.json \
  --k-values 1 5 10 20
```

### Show Per-Query Metrics

```bash
uv run scripts/evaluate_retrieval.py \
  --results results.json \
  --ground-truth gt.json \
  --show-per-query
```

### Query from Queries File

Create a queries file with IDs:
```json
{
  "queries": [
    {"query_id": "q1", "query_text": "What is covered?"},
    {"query_id": "q2", "query_text": "Which hospitals?"}
  ]
}
```

Then query:
```bash
uv run scripts/query_vector_db.py --queries-file queries.json --save-results results.json
```

### Batch Ground Truth Creation

Create queries file with graded relevance, then:
```bash
uv run scripts/create_ground_truth.py --from-file queries_with_relevance.json --output gt.json
```

Example input file format:
```json
{
  "queries": [
    {
      "query_id": "q1",
      "query_text": "What is covered?",
      "relevant_docs": ["5", "12", "23"]
    }
  ]
}
```

---

## Best Practices

### Creating Good Ground Truth

1. **Start Small**: Begin with 10-20 high-quality annotations
2. **Diverse Queries**: Cover different query types and difficulties
3. **Be Consistent**: Use the same criteria for relevance scores across queries
4. **Document Criteria**: Note your relevance criteria in metadata
5. **Multiple Levels**: Include documents at different relevance levels for better nDCG evaluation

### Interpreting Results

**High Precision, Low Recall:**
- System is conservative, returns only confident matches
- Good for precision-critical applications
- May need to increase K or adjust retrieval parameters

**Low Precision, High Recall:**
- System returns many results, some irrelevant
- Good for recall-critical applications
- May need better ranking or filtering

**Good nDCG, Variable P/R:**
- Ranking is good (relevant docs appear early)
- May have inconsistent total relevant docs retrieved

**Low MRR:**
- First relevant result appears late in ranking
- Focus on improving top-1 accuracy

### Comparing Experiments

1. **Control Variables**: Change one thing at a time (e.g., just the vector DB)
2. **Use Same Ground Truth**: Compare apples to apples
3. **Multiple Runs**: Run multiple times to check consistency
4. **Consider Trade-offs**: Speed vs accuracy, cost vs quality
5. **Domain-Specific**: What matters for YOUR use case?

---

## Troubleshooting

### No Ground Truth Found

```
Error: No ground truth for query q1, skipping
```

**Solution:** Ensure query IDs in results match ground truth. Check:
- Query IDs are strings
- IDs match exactly (case-sensitive)

### Document ID Mismatch

```
Warning: No relevant IDs for query q1
```

**Solution:** Verify document IDs in ground truth match the `id` field from vector DB results. Use the same ID format (string vs int).

### Empty Evaluation Results

```
Warning: No queries could be evaluated!
```

**Causes:**
- Query IDs don't match between results and ground truth
- No results returned for queries
- Document IDs don't match

**Solution:** 
1. Check query_results file structure
2. Verify ground truth file
3. Ensure IDs are consistent

### Metrics All Zero

Usually means no relevant documents were retrieved. Check:
- Ground truth has correct document IDs
- Vector database contains the expected documents
- Queries are appropriate for your document set

---

## File Structure

```
data/
├── evaluation/
│   ├── ground_truth/
│   │   ├── insurance_queries.json
│   │   └── template.json
│   ├── query_results/
│   │   ├── faiss_openai_20241028.json
│   │   └── milvus_openai_20241028.json
│   └── eval_results/
│       ├── faiss_openai_eval_20241028.json
│       ├── milvus_openai_eval_20241028.json
│       └── comparison_20241028.csv
```

---

## API Usage (Programmatic)

You can also use the evaluation system programmatically:

```python
from evaluation import RetrievalEvaluator, GroundTruthManager
from pathlib import Path

# Create ground truth with graded relevance
gt_manager = GroundTruthManager()
gt_manager.add_query(
    query_id="q1",
    query_text="What is covered?",
    relevant_docs={"5": 3.0, "12": 2.0, "23": 1.0}  # Dict with scores
)
gt_manager.save(Path("data/evaluation/ground_truth/my_gt.json"))

# Evaluate
evaluator = RetrievalEvaluator(
    ground_truth_path=Path("data/evaluation/ground_truth/my_gt.json"),
    k_values=[1, 3, 5]
)
evaluator.load_results(Path("data/evaluation/query_results/results.json"))
results = evaluator.evaluate()

# Get metrics
aggregate = evaluator.get_aggregate_metrics()
print(f"nDCG@5: {aggregate['ndcg@5']:.4f}")
```

---

## Next Steps

1. **Create your first ground truth** with 10-20 queries
2. **Run baseline evaluation** with current configuration
3. **Experiment with different settings** (embeddings, vector DBs, chunking)
4. **Compare results** to find the best configuration
5. **Iterate** based on findings

For more details on specific metrics, see evaluation literature or IR textbooks.