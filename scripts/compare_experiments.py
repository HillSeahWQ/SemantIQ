"""
Compare evaluation results from multiple experiments.
Generates comparison tables and identifies best configurations.
"""
import sys
import json
import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import EVAL_RESULTS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


def load_evaluation_result(path: Path) -> Dict[str, Any]:
    """Load an evaluation result file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_experiments(eval_files: List[Path]) -> Dict[str, Any]:
    """
    Compare multiple evaluation results.
    
    Args:
        eval_files: List of paths to evaluation result files
        
    Returns:
        Dictionary with comparison data
    """
    logger.info(f"Comparing {len(eval_files)} experiments...")
    
    experiments = []
    
    for eval_file in eval_files:
        try:
            data = load_evaluation_result(eval_file)
            
            # Extract experiment info
            experiment = {
                "name": eval_file.stem,
                "path": str(eval_file),
                "timestamp": data.get("evaluation_timestamp", "unknown"),
                "metrics": data.get("aggregate_metrics", {}),
                "k_values": data.get("k_values", []),
                "queries_evaluated": data.get("summary", {}).get("queries_evaluated", 0)
            }
            
            # Add original metadata if available
            if "original_metadata" in data:
                experiment["original_metadata"] = data["original_metadata"]
            
            experiments.append(experiment)
            
        except Exception as e:
            logger.error(f"Error loading {eval_file}: {e}")
            continue
    
    if not experiments:
        raise ValueError("No valid evaluation files loaded")
    
    # Get all metric names
    all_metrics = set()
    for exp in experiments:
        all_metrics.update(exp["metrics"].keys())
    
    return {
        "experiments": experiments,
        "all_metrics": sorted(all_metrics),
        "num_experiments": len(experiments)
    }


def print_comparison_table(comparison: Dict[str, Any]) -> None:
    """Print comparison in a formatted table."""
    experiments = comparison["experiments"]
    all_metrics = comparison["all_metrics"]
    
    print("\n" + "="*100)
    print("EXPERIMENT COMPARISON")
    print("="*100)
    
    # Print header
    header = ["Metric"] + [exp["name"][:30] for exp in experiments]
    
    print(f"\n{'Metric':<30}", end="")
    for exp in experiments:
        print(f"{exp['name'][:20]:>20}", end="")
    print()
    print("-"*100)
    
    # Group metrics by type
    metric_groups = defaultdict(list)
    for metric in all_metrics:
        if "@" in metric:
            base_metric = metric.split("@")[0]
            metric_groups[base_metric].append(metric)
        else:
            metric_groups[metric].append(metric)
    
    # Print metrics by group
    for group_name in sorted(metric_groups.keys()):
        group_metrics = sorted(metric_groups[group_name])
        
        print(f"\n{group_name.upper()}:")
        for metric in group_metrics:
            print(f"  {metric:<28}", end="")
            
            # Find best value for highlighting
            values = [exp["metrics"].get(metric, 0) for exp in experiments]
            best_value = max(values) if values else 0
            
            for exp in experiments:
                value = exp["metrics"].get(metric, 0)
                
                # Highlight best value
                if value == best_value and value > 0:
                    print(f"  {value:>18.4f}*", end="")
                else:
                    print(f"  {value:>19.4f}", end="")
            print()
    
    print("\n" + "-"*100)
    print("* indicates best value for that metric")
    print("="*100 + "\n")


def save_comparison_csv(comparison: Dict[str, Any], output_path: Path) -> None:
    """Save comparison to CSV file."""
    experiments = comparison["experiments"]
    all_metrics = comparison["all_metrics"]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ["Metric"] + [exp["name"] for exp in experiments]
        writer.writerow(header)
        
        # Write metrics
        for metric in all_metrics:
            row = [metric]
            for exp in experiments:
                value = exp["metrics"].get(metric, 0)
                row.append(f"{value:.4f}")
            writer.writerow(row)
        
        # Write metadata
        writer.writerow([])
        writer.writerow(["Metadata"])
        writer.writerow(["Experiment", "Queries Evaluated", "Timestamp"])
        
        for exp in experiments:
            writer.writerow([
                exp["name"],
                exp["queries_evaluated"],
                exp["timestamp"]
            ])
    
    logger.info(f"Comparison saved to CSV: {output_path}")


def find_best_experiment(comparison: Dict[str, Any], primary_metric: str = "ndcg@5") -> Dict[str, Any]:
    """
    Find the best experiment based on a primary metric.
    
    Args:
        comparison: Comparison data
        primary_metric: Metric to use for ranking
        
    Returns:
        Best experiment data
    """
    experiments = comparison["experiments"]
    
    if primary_metric not in comparison["all_metrics"]:
        logger.warning(f"Metric {primary_metric} not found. Using first available metric.")
        primary_metric = comparison["all_metrics"][0] if comparison["all_metrics"] else None
    
    if not primary_metric:
        return None
    
    best_exp = max(
        experiments,
        key=lambda exp: exp["metrics"].get(primary_metric, 0)
    )
    
    return {
        "experiment": best_exp,
        "metric": primary_metric,
        "value": best_exp["metrics"].get(primary_metric, 0)
    }


def print_best_experiment(best: Dict[str, Any]) -> None:
    """Print information about the best experiment."""
    if not best:
        logger.warning("No best experiment found")
        return
    
    exp = best["experiment"]
    
    print("\n" + "="*80)
    print("BEST EXPERIMENT")
    print("="*80)
    print(f"Name: {exp['name']}")
    print(f"Based on: {best['metric']} = {best['value']:.4f}")
    print(f"Queries evaluated: {exp['queries_evaluated']}")
    print(f"Timestamp: {exp['timestamp']}")
    
    if "original_metadata" in exp:
        metadata = exp["original_metadata"]
        print("\nConfiguration:")
        print(f"  Vector DB: {metadata.get('vector_db', 'unknown')}")
        print(f"  Embedding Model: {metadata.get('embedding_model', 'unknown')}")
        print(f"  Top K: {metadata.get('top_k', 'unknown')}")
    
    print("="*80 + "\n")


def main():
    """Main comparison entry point."""
    parser = argparse.ArgumentParser(
        description="Compare evaluation results from multiple experiments"
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        required=True,
        help="Paths to evaluation result files"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save comparison CSV"
    )
    parser.add_argument(
        "--primary-metric",
        type=str,
        default="ndcg@5",
        help="Primary metric for ranking (default: ndcg@5)"
    )
    
    args = parser.parse_args()
    
    # Parse paths
    eval_files = [Path(p) for p in args.experiments]
    
    # Validate paths
    for eval_file in eval_files:
        if not eval_file.exists():
            logger.error(f"File not found: {eval_file}")
            return 1
    
    logger.info("="*80)
    logger.info("COMPARING EXPERIMENTS")
    logger.info("="*80)
    
    try:
        # Compare experiments
        comparison = compare_experiments(eval_files)
        
        # Print comparison table
        print_comparison_table(comparison)
        
        # Find and print best experiment
        best = find_best_experiment(comparison, args.primary_metric)
        print_best_experiment(best)
        
        # Save to CSV if requested
        if args.output:
            output_path = Path(args.output)
            save_comparison_csv(comparison, output_path)
        else:
            # Auto-generate filename
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = EVAL_RESULTS_DIR / f"comparison_{timestamp}.csv"
            save_comparison_csv(comparison, output_path)
        
        logger.info(f"\n[SUCCESS] - Comparison complete")
        logger.info(f"Results saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"[ERROR] - Comparison failed: {e}")
        logger.exception("Full error traceback:")
        return 1


if __name__ == "__main__":
    sys.exit(main())