"""
Evaluation script for retrieval results.
Computes precision, recall, MRR, MAP, nDCG and other metrics.
"""
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    EVALUATION_CONFIG,
    GROUND_TRUTH_DIR,
    QUERY_RESULTS_DIR,
    EVAL_RESULTS_DIR
)
from evaluation.evaluator import RetrievalEvaluator
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval results against ground truth"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to query results JSON file"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        help=f"Path to ground truth JSON file (default: {EVALUATION_CONFIG['default_ground_truth']})"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save evaluation results (default: auto-generated in eval_results/)"
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=None,
        help=f"K values for metrics (default: {EVALUATION_CONFIG['k_values']})"
    )
    parser.add_argument(
        "--show-per-query",
        action="store_true",
        help="Show per-query metrics in output"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save evaluation results to file"
    )
    
    args = parser.parse_args()
    
    # Parse paths
    results_path = Path(args.results)
    
    if args.ground_truth:
        ground_truth_path = Path(args.ground_truth)
    else:
        ground_truth_path = Path(EVALUATION_CONFIG["default_ground_truth"])
    
    # Validate paths
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return 1
    
    if not ground_truth_path.exists():
        logger.error(f"Ground truth file not found: {ground_truth_path}")
        logger.info("\nTo create ground truth, run:")
        logger.info("  python scripts/create_ground_truth.py --interactive")
        return 1
    
    # K values
    k_values = args.k_values or EVALUATION_CONFIG["k_values"]
    
    # Run evaluation
    logger.info("="*80)
    logger.info("RETRIEVAL EVALUATION")
    logger.info("="*80)
    logger.info(f"Results file: {results_path}")
    logger.info(f"Ground truth: {ground_truth_path}")
    logger.info(f"K values: {k_values}")
    logger.info("")
    
    try:
        # Create evaluator
        evaluator = RetrievalEvaluator(
            ground_truth_path=ground_truth_path,
            k_values=k_values
        )
        
        # Load results
        evaluator.load_results(results_path)
        
        # Evaluate
        eval_results = evaluator.evaluate()
        
        # Print results
        evaluator.print_results(show_per_query=args.show_per_query)
        
        # Save results
        if not args.no_save:
            if args.output:
                output_path = Path(args.output)
            else:
                # Auto-generate output filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_basename = results_path.stem
                output_filename = f"{results_basename}_eval_{timestamp}.json"
                output_path = EVAL_RESULTS_DIR / output_filename
            
            evaluator.save_results(output_path)
            logger.info(f"\n✓ Evaluation results saved to: {output_path}")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)
        logger.info(f"Queries evaluated: {eval_results['summary']['queries_evaluated']}")
        logger.info(f"Queries skipped: {eval_results['summary']['queries_skipped']}")
        logger.info("")
        
        # Print top metrics
        aggregate = eval_results['aggregate_metrics']
        if aggregate:
            logger.info("Key Metrics:")
            for k in k_values[:3]:  # Show first 3 K values
                logger.info(f"  Precision@{k}: {aggregate.get(f'precision@{k}', 0):.4f}")
                logger.info(f"  Recall@{k}: {aggregate.get(f'recall@{k}', 0):.4f}")
                logger.info(f"  nDCG@{k}: {aggregate.get(f'ndcg@{k}', 0):.4f}")
                logger.info("")
            logger.info(f"  MRR: {aggregate.get('mrr', 0):.4f}")
            logger.info(f"  MAP: {aggregate.get('map', 0):.4f}")
        
        logger.info("="*80)
        logger.info("\n[SUCCESS] - Evaluation complete")
        
        return 0
        
    except Exception as e:
        logger.error(f"[ERROR] - Evaluation failed: {e}")
        logger.exception("Full error traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())