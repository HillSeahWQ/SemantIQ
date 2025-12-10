"""
Evaluation module for retrieval system.
Provides metrics, ground truth management, and evaluation orchestration.
"""

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    average_precision,
    ndcg_at_k,
    calculate_all_metrics,
    aggregate_metrics
)

from evaluation.ground_truth import (
    QueryGroundTruth,
    GroundTruthManager,
    create_ground_truth_template
)

from evaluation.evaluator import RetrievalEvaluator

__all__ = [
    # Metrics
    'precision_at_k',
    'recall_at_k',
    'f1_at_k',
    'hit_rate_at_k',
    'reciprocal_rank',
    'average_precision',
    'ndcg_at_k',
    'calculate_all_metrics',
    'aggregate_metrics',
    
    # Ground Truth
    'QueryGroundTruth',
    'GroundTruthManager',
    'create_ground_truth_template',
    
    # Evaluator
    'RetrievalEvaluator'
]