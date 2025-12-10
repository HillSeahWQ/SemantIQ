"""
Evaluation metrics for retrieval system.
Implements precision@k, recall@k, MRR, MAP, nDCG@k, etc.
Supports both binary and graded relevance.
"""
import numpy as np
from typing import List, Dict, Set, Any, Union
import logging

logger = logging.getLogger(__name__)


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Precision@K.
    
    Precision@K = (# of relevant items in top-K) / K
    
    Args:
        retrieved: List of retrieved item IDs (in ranked order)
        relevant: Set of relevant item IDs
        k: Cutoff rank
        
    Returns:
        Precision score (0.0 to 1.0)
    """
    if k <= 0 or len(retrieved) == 0:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = sum(1 for item in retrieved_at_k if item in relevant)
    
    return relevant_retrieved / k


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Recall@K.
    
    Recall@K = (# of relevant items in top-K) / (total # of relevant items)
    
    Args:
        retrieved: List of retrieved item IDs (in ranked order)
        relevant: Set of relevant item IDs
        k: Cutoff rank
        
    Returns:
        Recall score (0.0 to 1.0)
    """
    if len(relevant) == 0:
        return 0.0
    
    if k <= 0 or len(retrieved) == 0:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    relevant_retrieved = sum(1 for item in retrieved_at_k if item in relevant)
    
    return relevant_retrieved / len(relevant)


def f1_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate F1@K score.
    
    F1@K = 2 * (Precision@K * Recall@K) / (Precision@K + Recall@K)
    
    Args:
        retrieved: List of retrieved item IDs (in ranked order)
        relevant: Set of relevant item IDs
        k: Cutoff rank
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    precision = precision_at_k(retrieved, relevant, k)
    recall = recall_at_k(retrieved, relevant, k)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def hit_rate_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """
    Calculate Hit Rate@K (also called Success@K).
    
    Hit Rate@K = 1 if at least one relevant item in top-K, else 0
    
    Args:
        retrieved: List of retrieved item IDs (in ranked order)
        relevant: Set of relevant item IDs
        k: Cutoff rank
        
    Returns:
        1.0 if hit, 0.0 otherwise
    """
    if k <= 0 or len(retrieved) == 0 or len(relevant) == 0:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    
    for item in retrieved_at_k:
        if item in relevant:
            return 1.0
    
    return 0.0


def reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Calculate Reciprocal Rank (RR).
    
    RR = 1 / (rank of first relevant item)
    
    Args:
        retrieved: List of retrieved item IDs (in ranked order)
        relevant: Set of relevant item IDs
        
    Returns:
        Reciprocal rank score (0.0 to 1.0)
    """
    if len(retrieved) == 0 or len(relevant) == 0:
        return 0.0
    
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    
    return 0.0


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """
    Calculate Average Precision (AP).
    
    AP = (sum of P@k for each relevant item) / (total # of relevant items)
    
    Args:
        retrieved: List of retrieved item IDs (in ranked order)
        relevant: Set of relevant item IDs
        
    Returns:
        Average precision score (0.0 to 1.0)
    """
    if len(relevant) == 0:
        return 0.0
    
    if len(retrieved) == 0:
        return 0.0
    
    precision_sum = 0.0
    num_relevant_seen = 0
    
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            num_relevant_seen += 1
            precision_at_rank = num_relevant_seen / rank
            precision_sum += precision_at_rank
    
    return precision_sum / len(relevant)


def dcg_at_k(retrieved: List[str], relevant: Union[Set[str], Dict[str, float]], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at K (DCG@K).
    
    For binary relevance:
    DCG@K = sum_{i=1}^{K} (rel_i / log2(i + 1))
    where rel_i = 1 if item is relevant, 0 otherwise
    
    For graded relevance:
    DCG@K = sum_{i=1}^{K} (relevance_score_i / log2(i + 1))
    
    Args:
        retrieved: List of retrieved item IDs (in ranked order)
        relevant: Set of relevant IDs (binary) or Dict mapping IDs to relevance scores (graded)
        k: Cutoff rank
        
    Returns:
        DCG score
    """
    if k <= 0 or len(retrieved) == 0:
        return 0.0
    
    dcg = 0.0
    retrieved_at_k = retrieved[:k]
    
    # Check if graded relevance (dict) or binary (set)
    is_graded = isinstance(relevant, dict)
    
    for rank, item in enumerate(retrieved_at_k, start=1):
        if is_graded:
            relevance = relevant.get(item, 0.0)
        else:
            relevance = 1.0 if item in relevant else 0.0
        
        dcg += relevance / np.log2(rank + 1)
    
    return dcg


def idcg_at_k(relevant: Union[Set[str], Dict[str, float]], k: int) -> float:
    """
    Calculate Ideal Discounted Cumulative Gain at K (IDCG@K).
    
    IDCG@K = DCG of the ideal ranking (all relevant items at top, sorted by relevance)
    
    Args:
        relevant: Set of relevant IDs (binary) or Dict mapping IDs to relevance scores (graded)
        k: Cutoff rank
        
    Returns:
        IDCG score
    """
    is_graded = isinstance(relevant, dict)
    
    if is_graded:
        if not relevant:
            return 0.0
        
        # Sort relevance scores in descending order
        sorted_scores = sorted(relevant.values(), reverse=True)
        num_relevant_at_k = min(len(sorted_scores), k)
        
        idcg = 0.0
        for rank in range(1, num_relevant_at_k + 1):
            idcg += sorted_scores[rank - 1] / np.log2(rank + 1)
        
        return idcg
    else:
        # Binary relevance
        if len(relevant) == 0:
            return 0.0
        
        num_relevant_at_k = min(len(relevant), k)
        
        idcg = 0.0
        for rank in range(1, num_relevant_at_k + 1):
            idcg += 1.0 / np.log2(rank + 1)
        
        return idcg


def ndcg_at_k(retrieved: List[str], relevant: Union[Set[str], Dict[str, float]], k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K (nDCG@K).
    
    nDCG@K = DCG@K / IDCG@K
    
    Supports both binary and graded relevance.
    
    Args:
        retrieved: List of retrieved item IDs (in ranked order)
        relevant: Set of relevant IDs (binary) or Dict mapping IDs to relevance scores (graded)
        k: Cutoff rank
        
    Returns:
        nDCG score (0.0 to 1.0)
    """
    dcg = dcg_at_k(retrieved, relevant, k)
    idcg = idcg_at_k(relevant, k)
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


def calculate_all_metrics(
    retrieved: List[str],
    relevant: Union[Set[str], Dict[str, float]],
    k_values: List[int]
) -> Dict[str, float]:
    """
    Calculate all retrieval metrics for a single query.
    
    Supports both binary relevance (Set) and graded relevance (Dict).
    
    Args:
        retrieved: List of retrieved item IDs (in ranked order)
        relevant: Set of relevant IDs (binary) or Dict mapping IDs to relevance scores (graded)
        k_values: List of K values to compute metrics for
        
    Returns:
        Dictionary of metric names to scores
    """
    metrics = {}
    
    # Convert to set for binary metrics if needed
    if isinstance(relevant, dict):
        relevant_set = set(relevant.keys())
    else:
        relevant_set = relevant
    
    # Metrics at different K values
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(retrieved, relevant_set, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved, relevant_set, k)
        metrics[f"f1@{k}"] = f1_at_k(retrieved, relevant_set, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, relevant_set, k)
        # nDCG supports both binary and graded
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    
    # Metrics not dependent on K
    metrics["mrr"] = reciprocal_rank(retrieved, relevant_set)
    metrics["map"] = average_precision(retrieved, relevant_set)
    
    return metrics


def aggregate_metrics(
    per_query_metrics: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple queries using mean.
    
    Args:
        per_query_metrics: List of metric dictionaries, one per query
        
    Returns:
        Dictionary of aggregated metrics
    """
    if not per_query_metrics:
        return {}
    
    # Get all metric names, excluding non-numeric fields
    excluded_fields = {'query_id', 'query_text', 'num_retrieved', 'num_relevant'}
    metric_names = [k for k in per_query_metrics[0].keys() if k not in excluded_fields]
    
    aggregated = {}
    for metric_name in metric_names:
        values = [m[metric_name] for m in per_query_metrics if isinstance(m.get(metric_name), (int, float))]
        if values:
            aggregated[metric_name] = float(np.mean(values))
    
    return aggregated


def print_metrics_table(
    metrics: Dict[str, float],
    title: str = "Evaluation Metrics"
) -> None:
    """
    Print metrics in a formatted table.
    
    Args:
        metrics: Dictionary of metric names to scores
        title: Title for the table
    """
    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    
    # Group metrics by type
    metric_groups = {
        "Precision": [],
        "Recall": [],
        "F1": [],
        "Hit Rate": [],
        "nDCG": [],
        "Other": []
    }
    
    for metric_name, score in sorted(metrics.items()):
        if metric_name.startswith("precision"):
            metric_groups["Precision"].append((metric_name, score))
        elif metric_name.startswith("recall"):
            metric_groups["Recall"].append((metric_name, score))
        elif metric_name.startswith("f1"):
            metric_groups["F1"].append((metric_name, score))
        elif metric_name.startswith("hit_rate"):
            metric_groups["Hit Rate"].append((metric_name, score))
        elif metric_name.startswith("ndcg"):
            metric_groups["nDCG"].append((metric_name, score))
        else:
            metric_groups["Other"].append((metric_name, score))
    
    for group_name, group_metrics in metric_groups.items():
        if group_metrics:
            print(f"\n{group_name}:")
            print("-" * 60)
            for metric_name, score in group_metrics:
                print(f"  {metric_name:<30} {score:>10.4f}")
    
    print("=" * 60 + "\n")