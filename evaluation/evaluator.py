"""
Main evaluator for retrieval results.
Orchestrates evaluation by loading results, ground truth, and computing metrics.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from evaluation.metrics import calculate_all_metrics, aggregate_metrics, print_metrics_table
from evaluation.ground_truth import GroundTruthManager

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Evaluator for retrieval system results."""
    
    def __init__(
        self,
        ground_truth_path: Path,
        k_values: List[int] = [1, 3, 5, 10]
    ):
        """
        Initialize evaluator.
        
        Args:
            ground_truth_path: Path to ground truth JSON file
            k_values: List of K values for metrics computation
        """
        self.ground_truth_manager = GroundTruthManager(ground_truth_path)
        self.k_values = k_values
        self.results_data = None
        self.per_query_metrics = []
        self.aggregate_metrics = {}
    
    def load_results(self, results_path: Path) -> None:
        """
        Load query results from JSON file.
        
        Expected format:
        {
            "metadata": {...},
            "queries": [
                {
                    "query_id": "q1",
                    "query_text": "...",
                    "results": [
                        {"id": "doc1", "score": 0.95, ...},
                        {"id": "doc2", "score": 0.87, ...}
                    ]
                }
            ]
        }
        
        Args:
            results_path: Path to results JSON file
        """
        if not results_path.exists():
            raise FileNotFoundError(f"Results file not found: {results_path}")
        
        logger.info(f"Loading results from: {results_path}")
        
        with open(results_path, 'r', encoding='utf-8') as f:
            self.results_data = json.load(f)
        
        # Validate format
        if "queries" not in self.results_data:
            raise ValueError("Invalid results format: missing 'queries' key")
        
        logger.info(f"Loaded results for {len(self.results_data['queries'])} queries")
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate retrieval results against ground truth.
        
        Returns:
            Dictionary containing evaluation results
        """
        if self.results_data is None:
            raise ValueError("No results loaded. Call load_results() first.")
        
        logger.info("Starting evaluation...")
        logger.info(f"Computing metrics at K = {self.k_values}")
        
        self.per_query_metrics = []
        queries_evaluated = 0
        queries_skipped = 0
        
        for query_result in self.results_data["queries"]:
            query_id = query_result["query_id"]
            query_text = query_result.get("query_text", "")
            
            # Get ground truth
            relevant_docs = self.ground_truth_manager.get_relevant_docs(query_id)
            
            if not relevant_docs:
                logger.warning(f"No ground truth for query {query_id}, skipping")
                queries_skipped += 1
                continue
            
            # Extract retrieved document IDs
            retrieved_docs = []
            for result in query_result.get("results", []):
                # Support different ID field names
                doc_id = result.get("id") or result.get("doc_id") or result.get("chunk_id")
                if doc_id is not None:
                    # Convert to string for consistent comparison
                    retrieved_docs.append(str(doc_id))
            
            if not retrieved_docs:
                logger.warning(f"No results for query {query_id}")
                queries_skipped += 1
                continue
            
            # Calculate metrics for this query
            query_metrics = calculate_all_metrics(
                retrieved=retrieved_docs,
                relevant=relevant_docs,
                k_values=self.k_values
            )
            
            # Add query information
            query_metrics["query_id"] = query_id
            query_metrics["query_text"] = query_text
            query_metrics["num_retrieved"] = len(retrieved_docs)
            query_metrics["num_relevant"] = len(relevant_docs)
            
            self.per_query_metrics.append(query_metrics)
            queries_evaluated += 1
        
        # Compute aggregate metrics
        if self.per_query_metrics:
            self.aggregate_metrics = aggregate_metrics(self.per_query_metrics)
        else:
            logger.warning("No queries could be evaluated!")
            self.aggregate_metrics = {}
        
        logger.info(f"Evaluation complete: {queries_evaluated} queries evaluated, {queries_skipped} skipped")
        
        return {
            "aggregate_metrics": self.aggregate_metrics,
            "per_query_metrics": self.per_query_metrics,
            "summary": {
                "queries_evaluated": queries_evaluated,
                "queries_skipped": queries_skipped,
                "total_queries": len(self.results_data["queries"])
            }
        }
    
    def save_results(self, output_path: Path, include_metadata: bool = True) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            output_path: Path to save results
            include_metadata: Whether to include metadata from original results
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "k_values": self.k_values,
            "aggregate_metrics": self.aggregate_metrics,
            "per_query_metrics": self.per_query_metrics,
            "summary": {
                "queries_evaluated": len(self.per_query_metrics),
                "total_queries": len(self.results_data["queries"]) if self.results_data else 0
            }
        }
        
        # Optionally include original metadata
        if include_metadata and self.results_data and "metadata" in self.results_data:
            output_data["original_metadata"] = self.results_data["metadata"]
        
        logger.info(f"Saving evaluation results to: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info("Evaluation results saved")
    
    def print_results(self, show_per_query: bool = False) -> None:
        """
        Print evaluation results in a formatted way.
        
        Args:
            show_per_query: Whether to show per-query metrics
        """
        if not self.aggregate_metrics:
            logger.warning("No evaluation results to display")
            return
        
        # Print aggregate metrics
        print_metrics_table(self.aggregate_metrics, "Aggregate Metrics")
        
        # Optionally print per-query metrics
        if show_per_query and self.per_query_metrics:
            print("\n" + "=" * 80)
            print("Per-Query Metrics")
            print("=" * 80)
            
            for i, query_metrics in enumerate(self.per_query_metrics, 1):
                print(f"\nQuery {i}: {query_metrics.get('query_text', 'N/A')[:60]}...")
                print(f"  Query ID: {query_metrics['query_id']}")
                print(f"  Retrieved: {query_metrics['num_retrieved']}, Relevant: {query_metrics['num_relevant']}")
                print("-" * 80)
                
                # Print key metrics
                for k in self.k_values:
                    print(f"  K={k}:")
                    print(f"    Precision: {query_metrics.get(f'precision@{k}', 0):.4f}")
                    print(f"    Recall:    {query_metrics.get(f'recall@{k}', 0):.4f}")
                    print(f"    nDCG:      {query_metrics.get(f'ndcg@{k}', 0):.4f}")
                
                print(f"  MRR: {query_metrics.get('mrr', 0):.4f}")
                print(f"  MAP: {query_metrics.get('map', 0):.4f}")
    
    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Get aggregate metrics."""
        return self.aggregate_metrics
    
    def get_per_query_metrics(self) -> List[Dict[str, Any]]:
        """Get per-query metrics."""
        return self.per_query_metrics