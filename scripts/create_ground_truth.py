"""
Script to create ground truth data for evaluation.
Supports interactive mode and batch creation from file.
"""
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import GROUND_TRUTH_DIR
from evaluation.ground_truth import GroundTruthManager, create_ground_truth_template
from utils.logger import get_logger

logger = get_logger(__name__)


def interactive_mode(output_path: Path) -> None:
    """
    Interactive mode for creating ground truth.
    Allows user to input queries and relevant document IDs.
    
    Args:
        output_path: Path to save ground truth file
    """
    print("\n" + "="*80)
    print("INTERACTIVE GROUND TRUTH CREATION")
    print("="*80)
    print("\nThis tool will help you create ground truth annotations for evaluation.")
    print("You'll need to provide:")
    print("  1. Query text")
    print("  2. Relevant document/chunk IDs (comma-separated)")
    print("\nPress Ctrl+C at any time to save and exit.\n")
    
    manager = GroundTruthManager()
    query_num = 1
    
    try:
        while True:
            print(f"\n--- Query {query_num} ---")
            
            # Get query text
            query_text = input("Enter query text (or 'done' to finish): ").strip()
            
            if query_text.lower() == 'done':
                break
            
            if not query_text:
                print("Query text cannot be empty. Try again.")
                continue
            
            # Generate query ID
            query_id = input(f"Enter query ID (default: q{query_num}): ").strip()
            if not query_id:
                query_id = f"q{query_num}"
            
            # Get relevant doc IDs
            print("Enter relevant document/chunk IDs (comma-separated):")
            print("Example: 5,12,23,45")
            relevant_ids_str = input("IDs: ").strip()
            
            if not relevant_ids_str:
                print("No relevant IDs provided. Skipping this query.")
                continue
            
            # Parse IDs
            try:
                relevant_ids = [id.strip() for id in relevant_ids_str.split(',')]
                relevant_ids = [id for id in relevant_ids if id]  # Remove empty strings
            except Exception as e:
                print(f"Error parsing IDs: {e}. Skipping this query.")
                continue
            
            if not relevant_ids:
                print("No valid IDs provided. Skipping this query.")
                continue
            
            # Optional metadata
            add_metadata = input("Add metadata? (y/n, default: n): ").strip().lower()
            metadata = None
            
            if add_metadata == 'y':
                category = input("  Category (optional): ").strip()
                difficulty = input("  Difficulty (easy/medium/hard, optional): ").strip()
                
                metadata = {}
                if category:
                    metadata["category"] = category
                if difficulty:
                    metadata["difficulty"] = difficulty
            
            # Add to manager
            manager.add_query(
                query_id=query_id,
                query_text=query_text,
                relevant_doc_ids=relevant_ids,
                metadata=metadata
            )
            
            print(f"✓ Added query '{query_id}' with {len(relevant_ids)} relevant documents")
            query_num += 1
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving current progress...")
    
    # Save ground truth
    if len(manager.ground_truth_data) > 0:
        manager.save(output_path)
        manager.print_summary()
        print(f"\n✓ Ground truth saved to: {output_path}")
        print(f"  Total queries: {len(manager.ground_truth_data)}")
    else:
        print("\nNo queries added. Nothing to save.")


def batch_mode(queries_file: Path, output_path: Path) -> None:
    """
    Batch mode for creating ground truth from a queries file.
    
    Expected input format (JSON):
    {
        "queries": [
            {
                "query_id": "q1",
                "query_text": "What is covered?",
                "relevant_doc_ids": ["5", "12", "23"]
            }
        ]
    }
    
    Args:
        queries_file: Path to queries JSON file
        output_path: Path to save ground truth file
    """
    print("\n" + "="*80)
    print("BATCH GROUND TRUTH CREATION")
    print("="*80)
    
    if not queries_file.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_file}")
    
    logger.info(f"Loading queries from: {queries_file}")
    
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)
    
    if "queries" not in queries_data:
        raise ValueError("Invalid format: missing 'queries' key")
    
    manager = GroundTruthManager()
    
    for query_data in queries_data["queries"]:
        query_id = query_data.get("query_id")
        query_text = query_data.get("query_text")
        relevant_ids = query_data.get("relevant_doc_ids", [])
        metadata = query_data.get("metadata")
        
        if not query_id or not query_text:
            logger.warning(f"Skipping invalid query: {query_data}")
            continue
        
        if not relevant_ids:
            logger.warning(f"No relevant IDs for query {query_id}")
        
        manager.add_query(
            query_id=query_id,
            query_text=query_text,
            relevant_doc_ids=relevant_ids,
            metadata=metadata
        )
    
    # Save
    manager.save(output_path)
    manager.print_summary()
    
    print(f"\n✓ Ground truth created from {len(queries_data['queries'])} queries")
    print(f"  Saved to: {output_path}")


def create_template_mode(output_path: Path, num_queries: int) -> None:
    """
    Create a template ground truth file for manual annotation.
    
    Args:
        output_path: Path to save template
        num_queries: Number of template queries to create
    """
    print("\n" + "="*80)
    print("CREATING GROUND TRUTH TEMPLATE")
    print("="*80)
    
    create_ground_truth_template(output_path, num_queries)
    
    print(f"\n✓ Template created with {num_queries} example queries")
    print(f"  Location: {output_path}")
    print("\nNext steps:")
    print("  1. Edit the template file")
    print("  2. Replace example queries with your actual queries")
    print("  3. Add relevant document IDs for each query")
    print("  4. Use this file for evaluation")


def main():
    """Main entry point for ground truth creation."""
    parser = argparse.ArgumentParser(
        description="Create ground truth data for retrieval evaluation"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode: enter queries manually"
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="Batch mode: create from queries file (JSON)"
    )
    parser.add_argument(
        "--template",
        action="store_true",
        help="Create a template file for manual annotation"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=10,
        help="Number of template queries (for --template mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for ground truth file"
    )
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = GROUND_TRUTH_DIR / "ground_truth.json"
    
    # Execute appropriate mode
    if args.interactive:
        interactive_mode(output_path)
    elif args.from_file:
        queries_file = Path(args.from_file)
        batch_mode(queries_file, output_path)
    elif args.template:
        create_template_mode(output_path, args.num_queries)
    else:
        parser.print_help()
        print("\nPlease specify a mode:")
        print("  --interactive       Interactive mode")
        print("  --from-file FILE    Batch mode from file")
        print("  --template          Create template file")


if __name__ == "__main__":
    main()