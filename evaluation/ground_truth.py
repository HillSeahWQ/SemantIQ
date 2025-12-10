"""
Ground truth data management for retrieval evaluation.
Handles loading, validation, and creation of query-document relevance annotations.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class QueryGroundTruth:
    """Ground truth for a single query."""
    query_id: str
    query_text: str
    relevant_doc_ids: List[str]  # List of relevant chunk IDs or document IDs
    metadata: Optional[Dict] = None  # Optional metadata (e.g., query category, difficulty)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'QueryGroundTruth':
        """Create from dictionary."""
        return cls(
            query_id=data["query_id"],
            query_text=data["query_text"],
            relevant_doc_ids=data["relevant_doc_ids"],
            metadata=data.get("metadata")
        )


class GroundTruthManager:
    """Manager for ground truth data."""
    
    def __init__(self, ground_truth_path: Optional[Path] = None):
        """
        Initialize ground truth manager.
        
        Args:
            ground_truth_path: Path to ground truth JSON file
        """
        self.ground_truth_path = ground_truth_path
        self.ground_truth_data: Dict[str, QueryGroundTruth] = {}
        
        if ground_truth_path and ground_truth_path.exists():
            self.load()
    
    def load(self, path: Optional[Path] = None) -> None:
        """
        Load ground truth from JSON file.
        
        Args:
            path: Path to ground truth file (uses self.ground_truth_path if None)
        """
        load_path = path or self.ground_truth_path
        
        if not load_path or not load_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {load_path}")
        
        logger.info(f"Loading ground truth from: {load_path}")
        
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Support two formats:
        # 1. List of query objects
        # 2. Dict with "queries" key
        if isinstance(data, list):
            queries = data
        elif isinstance(data, dict) and "queries" in data:
            queries = data["queries"]
        else:
            raise ValueError("Invalid ground truth format. Expected list or dict with 'queries' key")
        
        self.ground_truth_data = {}
        for query_data in queries:
            gt = QueryGroundTruth.from_dict(query_data)
            self.ground_truth_data[gt.query_id] = gt
        
        logger.info(f"Loaded {len(self.ground_truth_data)} queries with ground truth")
    
    def save(self, path: Optional[Path] = None) -> None:
        """
        Save ground truth to JSON file.
        
        Args:
            path: Path to save file (uses self.ground_truth_path if None)
        """
        save_path = path or self.ground_truth_path
        
        if not save_path:
            raise ValueError("No save path specified")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving ground truth to: {save_path}")
        
        data = {
            "queries": [gt.to_dict() for gt in self.ground_truth_data.values()],
            "metadata": {
                "num_queries": len(self.ground_truth_data),
                "format_version": "1.0"
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(self.ground_truth_data)} queries")
    
    def add_query(
        self,
        query_id: str,
        query_text: str,
        relevant_doc_ids: List[str],
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a query with its ground truth.
        
        Args:
            query_id: Unique query identifier
            query_text: The query text
            relevant_doc_ids: List of relevant document/chunk IDs
            metadata: Optional metadata
        """
        gt = QueryGroundTruth(
            query_id=query_id,
            query_text=query_text,
            relevant_doc_ids=relevant_doc_ids,
            metadata=metadata
        )
        self.ground_truth_data[query_id] = gt
        logger.debug(f"Added ground truth for query: {query_id}")
    
    def get_relevant_docs(self, query_id: str) -> Set[str]:
        """
        Get set of relevant document IDs for a query.
        
        Args:
            query_id: Query identifier
            
        Returns:
            Set of relevant document IDs
        """
        if query_id not in self.ground_truth_data:
            logger.warning(f"No ground truth found for query: {query_id}")
            return set()
        
        return set(self.ground_truth_data[query_id].relevant_doc_ids)
    
    def get_query_text(self, query_id: str) -> Optional[str]:
        """
        Get query text for a query ID.
        
        Args:
            query_id: Query identifier
            
        Returns:
            Query text or None if not found
        """
        if query_id not in self.ground_truth_data:
            return None
        return self.ground_truth_data[query_id].query_text
    
    def get_all_query_ids(self) -> List[str]:
        """Get all query IDs."""
        return list(self.ground_truth_data.keys())
    
    def get_all_queries(self) -> List[QueryGroundTruth]:
        """Get all ground truth queries."""
        return list(self.ground_truth_data.values())
    
    def validate(self) -> bool:
        """
        Validate ground truth data.
        
        Returns:
            True if valid, False otherwise
        """
        if not self.ground_truth_data:
            logger.warning("Ground truth is empty")
            return False
        
        valid = True
        for query_id, gt in self.ground_truth_data.items():
            # Check query ID consistency
            if gt.query_id != query_id:
                logger.error(f"Query ID mismatch: {query_id} vs {gt.query_id}")
                valid = False
            
            # Check query text
            if not gt.query_text or not gt.query_text.strip():
                logger.error(f"Empty query text for: {query_id}")
                valid = False
            
            # Check relevant docs
            if not gt.relevant_doc_ids:
                logger.warning(f"No relevant docs for query: {query_id}")
            
            # Check for duplicate relevant docs
            if len(gt.relevant_doc_ids) != len(set(gt.relevant_doc_ids)):
                logger.warning(f"Duplicate relevant docs for query: {query_id}")
        
        if valid:
            logger.info("Ground truth validation passed")
        else:
            logger.error("Ground truth validation failed")
        
        return valid
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the ground truth data.
        
        Returns:
            Dictionary with statistics
        """
        if not self.ground_truth_data:
            return {"num_queries": 0}
        
        relevant_counts = [len(gt.relevant_doc_ids) for gt in self.ground_truth_data.values()]
        
        return {
            "num_queries": len(self.ground_truth_data),
            "total_relevant_docs": sum(relevant_counts),
            "avg_relevant_per_query": sum(relevant_counts) / len(relevant_counts),
            "min_relevant_per_query": min(relevant_counts),
            "max_relevant_per_query": max(relevant_counts),
            "queries_with_metadata": sum(1 for gt in self.ground_truth_data.values() if gt.metadata)
        }
    
    def print_summary(self) -> None:
        """Print a summary of the ground truth data."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("Ground Truth Summary")
        print("=" * 60)
        print(f"Total Queries: {stats['num_queries']}")
        print(f"Total Relevant Documents: {stats['total_relevant_docs']}")
        print(f"Avg Relevant per Query: {stats['avg_relevant_per_query']:.2f}")
        print(f"Min Relevant per Query: {stats['min_relevant_per_query']}")
        print(f"Max Relevant per Query: {stats['max_relevant_per_query']}")
        print("=" * 60 + "\n")


def create_ground_truth_template(output_path: Path, num_queries: int = 5) -> None:
    """
    Create a template ground truth file for annotation.
    
    Args:
        output_path: Path to save the template
        num_queries: Number of template queries to create
    """
    template_queries = []
    
    for i in range(1, num_queries + 1):
        template_queries.append({
            "query_id": f"q{i}",
            "query_text": f"Example query {i} - Replace with actual query",
            "relevant_doc_ids": [],  # Add relevant chunk IDs here
            "metadata": {
                "category": "example",
                "difficulty": "easy"
            }
        })
    
    data = {
        "queries": template_queries,
        "metadata": {
            "num_queries": num_queries,
            "format_version": "1.0",
            "description": "Template for ground truth annotation"
        }
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Created ground truth template: {output_path}")
    print(f"\nTemplate created at: {output_path}")
    print("Please edit the file to add your queries and relevant document IDs.\n")