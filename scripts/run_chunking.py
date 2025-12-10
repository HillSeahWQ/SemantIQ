"""
Document chunking script with command-line interface.
Supports multiple document types: PDF, Word (DOC/DOCX), and more.
Supports both fast runs (file paths only) and advanced runs (with hyperparameters).
"""
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import (
    INPUT_DIR,
    get_chunk_output_path,
    CHUNKING_CONFIG,
    EXPERIMENT_CONFIG,
    DATA_DIR
)
from chunking.pdf_chunker import MultimodalPDFChunker
from chunking.word_doc_chunker import WordDocumentChunker
from chunking.code_chunker import CodeChunker
from chunking.base import Chunk
from utils.storage import save_chunks
from utils.logger import get_logger

logger = get_logger(__name__)


# Supported file types and their extensions
SUPPORTED_FILE_TYPES = {
    'pdf': ['.pdf'],
    'word': ['.doc', '.docx'],
    'code': ['.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.h', '.hpp', '.go', '.rb', '.php', '.rs'],
    # Add more file types here as you implement them
    # 'markdown': ['.md'],
    # 'text': ['.txt'],
    # 'html': ['.html', '.htm'],
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Chunk documents for RAG pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast run - minimal arguments (processes all supported file types)
  python run_chunking.py --input data/raw-documents --output data/chunks/output.json
  
  # Process specific file types only
  python run_chunking.py --input data/raw-documents --output data/chunks/output.json --file-types pdf word
  
  # Advanced run - with hyperparameters for PDF
  python run_chunking.py \\
      --input data/raw-documents \\
      --output data/chunks/output.json \\
      --pdf-image-threshold 0.2 \\
      --pdf-vision-model gpt-4o
  
  # Advanced run - with hyperparameters for Word
  python run_chunking.py \\
      --input data/raw-documents \\
      --output data/chunks/output.json \\
      --word-max-chunk-size 1500 \\
      --word-process-images
        """
    )
    
    # === FAST RUN ARGUMENTS (File Paths) ===
    parser.add_argument(
        "--input",
        type=str,
        help=f"Input directory with documents (default: data/raw-documents, or from config)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help=f"Output JSON file for chunks (default: data/chunks/<experiment>_chunks.json)"
    )
    parser.add_argument(
        "--file-types",
        type=str,
        nargs='+',
        choices=['pdf', 'word', 'code', 'all'],
        default=['all'],
        help="File types to process (default: all)"
    )
    
    # === PDF CHUNKER ARGUMENTS ===
    parser.add_argument(
        "--pdf-image-threshold",
        type=float,
        help=f"PDF: Image coverage threshold for vision processing (default: from config, currently {CHUNKING_CONFIG.get('pdf', {}).get('image_coverage_threshold', 0.15)})"
    )
    parser.add_argument(
        "--pdf-vision-model",
        type=str,
        choices=["gpt-4o", "gpt-4o-mini", "gpt-4-vision-preview"],
        help=f"PDF: Vision model for image processing (default: from config, currently {CHUNKING_CONFIG.get('pdf', {}).get('vision_model', 'gpt-4o')})"
    )
    
    # === WORD CHUNKER ARGUMENTS ===
    parser.add_argument(
        "--word-max-chunk-size",
        type=int,
        help=f"Word: Maximum chunk size in characters (default: from config, currently {CHUNKING_CONFIG.get('word', {}).get('max_chunk_size', 1000)})"
    )
    parser.add_argument(
        "--word-vision-model",
        type=str,
        choices=["gpt-4o", "gpt-4o-mini", "gpt-4-vision-preview"],
        help=f"Word: Vision model for image processing (default: from config, currently {CHUNKING_CONFIG.get('word', {}).get('vision_model', 'gpt-4o')})"
    )
    parser.add_argument(
        "--word-process-images",
        action="store_true",
        help="Word: Enable image processing with vision model (default: from config)"
    )
    
    # === CODE CHUNKER ARGUMENTS ===
    parser.add_argument(
        "--code-max-chunk-size",
        type=int,
        help=f"Code: Maximum chunk size in characters (default: from config, currently {CHUNKING_CONFIG.get('code', {}).get('max_chunk_size', 1500)})"
    )
    parser.add_argument(
        "--code-min-chunk-size",
        type=int,
        help=f"Code: Minimum chunk size in characters (default: from config, currently {CHUNKING_CONFIG.get('code', {}).get('min_chunk_size', 100)})"
    )
    parser.add_argument(
        "--code-include-imports",
        action="store_true",
        help="Code: Include imports in context (default: from config)"
    )
    parser.add_argument(
        "--code-no-include-imports",
        action="store_true",
        help="Code: Don't include imports (overrides config)"
    )
    
    parser.add_argument(
        "--word-no-process-images",
        action="store_true",
        help="Word: Disable image processing (overrides config)"
    )
    
    # === GLOBAL ARGUMENTS ===
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: from config)"
    )
    
    return parser.parse_args()


def get_files_by_type(input_dir: Path, file_types: List[str]) -> dict:
    """
    Get all files organized by type.
    
    Args:
        input_dir: Directory to search
        file_types: List of file types to include (e.g., ['pdf', 'word', 'all'])
    
    Returns:
        Dictionary mapping file type to list of file paths
    """
    files_by_type = {}
    
    # Determine which types to process
    types_to_process = []
    if 'all' in file_types:
        types_to_process = list(SUPPORTED_FILE_TYPES.keys())
    else:
        types_to_process = file_types
    
    # Collect files for each type
    for file_type in types_to_process:
        if file_type not in SUPPORTED_FILE_TYPES:
            logger.warning(f"Unknown file type: {file_type}")
            continue
        
        extensions = SUPPORTED_FILE_TYPES[file_type]
        files = []
        
        for ext in extensions:
            files.extend(input_dir.glob(f"**/*{ext}"))
        
        if files:
            files_by_type[file_type] = files
            logger.info(f"Found {len(files)} {file_type.upper()} files")
    
    return files_by_type


def process_pdf_files(
    pdf_files: List[Path],
    chunking_config: dict
) -> Tuple[List[Chunk], int, int]:
    """
    Process PDF files with PDF chunker.
    
    Args:
        pdf_files: List of PDF file paths
        chunking_config: Configuration for PDF chunker
    
    Returns:
        Tuple of (all_chunks, successful_count, failed_count)
    """
    logger.info("")
    logger.info("="*80)
    logger.info("PROCESSING PDF FILES")
    logger.info("="*80)
    
    chunker = MultimodalPDFChunker(**chunking_config)
    all_chunks = []
    successful = 0
    failed = 0
    
    for pdf_file in pdf_files:
        logger.info(f"Processing: {pdf_file.name}")
        try:
            chunks = chunker.chunk(pdf_file)
            all_chunks.extend(chunks)
            successful += 1
            logger.info(f"[SUCCESS] - Generated {len(chunks)} chunks")
        except Exception as e:
            failed += 1
            logger.error(f"[FAILED] - {e}")
            logger.exception("Error details:")
    
    logger.info("")
    logger.info(f"PDF Processing Summary:")
    logger.info(f"  Successful: {successful}/{len(pdf_files)}")
    logger.info(f"  Failed: {failed}/{len(pdf_files)}")
    logger.info(f"  Total chunks: {len(all_chunks)}")
    
    return all_chunks, successful, failed


def process_word_files(
    word_files: List[Path],
    chunking_config: dict
) -> Tuple[List[Chunk], int, int]:
    """
    Process Word files with Word chunker.
    
    Args:
        word_files: List of Word file paths
        chunking_config: Configuration for Word chunker
    
    Returns:
        Tuple of (all_chunks, successful_count, failed_count)
    """
    logger.info("")
    logger.info("="*80)
    logger.info("PROCESSING WORD FILES")
    logger.info("="*80)
    
    chunker = WordDocumentChunker(**chunking_config)
    all_chunks = []
    successful = 0
    failed = 0
    
    for word_file in word_files:
        logger.info(f"Processing: {word_file.name}")
        try:
            chunks = chunker.chunk(word_file)
            all_chunks.extend(chunks)
            successful += 1
            logger.info(f"[SUCCESS] - Generated {len(chunks)} chunks")
        except Exception as e:
            failed += 1
            logger.error(f"[FAILED] - {e}")
            logger.exception("Error details:")
    
    logger.info("")
    logger.info(f"Word Processing Summary:")
    logger.info(f"  Successful: {successful}/{len(word_files)}")
    logger.info(f"  Failed: {failed}/{len(word_files)}")
    logger.info(f"  Total chunks: {len(all_chunks)}")
    
    return all_chunks, successful, failed


def process_code_files(
    code_files: List[Path],
    chunking_config: dict
) -> Tuple[List[Chunk], int, int]:
    """
    Process code files with Code chunker.
    
    Args:
        code_files: List of code file paths
        chunking_config: Configuration for Code chunker
    
    Returns:
        Tuple of (all_chunks, successful_count, failed_count)
    """
    logger.info("")
    logger.info("="*80)
    logger.info("PROCESSING CODE FILES")
    logger.info("="*80)
    
    chunker = CodeChunker(**chunking_config)
    all_chunks = []
    successful = 0
    failed = 0
    
    for code_file in code_files:
        logger.info(f"Processing: {code_file.name}")
        try:
            chunks = chunker.chunk(code_file)
            all_chunks.extend(chunks)
            successful += 1
            logger.info(f"[SUCCESS] - Generated {len(chunks)} chunks")
        except Exception as e:
            failed += 1
            logger.error(f"[FAILED] - {e}")
            logger.exception("Error details:")
    
    logger.info("")
    logger.info(f"Code Processing Summary:")
    logger.info(f"  Successful: {successful}/{len(code_files)}")
    logger.info(f"  Failed: {failed}/{len(code_files)}")
    logger.info(f"  Total chunks: {len(all_chunks)}")
    
    return all_chunks, successful, failed


# ============================================================================
# SECTION: Add more file type processors here
# ============================================================================
# 
# def process_markdown_files(
#     markdown_files: List[Path],
#     chunking_config: dict
# ) -> Tuple[List[Chunk], int, int]:
#     """Process Markdown files."""
#     logger.info("")
#     logger.info("="*80)
#     logger.info("PROCESSING MARKDOWN FILES")
#     logger.info("="*80)
#     
#     # TODO: Implement MarkdownChunker
#     # chunker = MarkdownChunker(**chunking_config)
#     # ... process files ...
#     
#     logger.warning("Markdown processing not yet implemented")
#     return [], 0, len(markdown_files)
#
# def process_text_files(
#     text_files: List[Path],
#     chunking_config: dict
# ) -> Tuple[List[Chunk], int, int]:
#     """Process plain text files."""
#     logger.info("")
#     logger.info("="*80)
#     logger.info("PROCESSING TEXT FILES")
#     logger.info("="*80)
#     
#     # TODO: Implement TextChunker
#     # chunker = TextChunker(**chunking_config)
#     # ... process files ...
#     
#     logger.warning("Text processing not yet implemented")
#     return [], 0, len(text_files)
#
# ============================================================================


def main():
    """Run chunking pipeline."""
    args = parse_args()
    
    # === RESOLVE PATHS ===
    # Input directory (fast run)
    if args.input:
        input_dir = Path(args.input)
        # Support relative paths from data/
        if not input_dir.is_absolute():
            if not input_dir.exists():
                # Try relative to DATA_DIR
                input_dir = DATA_DIR / args.input
    else:
        input_dir = INPUT_DIR
    
    # Output file (fast run)
    if args.output:
        output_file = Path(args.output)
        # Support relative paths from data/
        if not output_file.is_absolute():
            output_file = DATA_DIR / args.output
    else:
        output_file = get_chunk_output_path()
    
    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # === BUILD CONFIGS FOR EACH FILE TYPE ===
    
    # PDF Config
    pdf_config = CHUNKING_CONFIG.get("pdf", {}).copy()
    if args.pdf_image_threshold is not None:
        pdf_config["image_coverage_threshold"] = args.pdf_image_threshold
    if args.pdf_vision_model is not None:
        pdf_config["vision_model"] = args.pdf_vision_model
    if args.log_level is not None:
        pdf_config["log_level"] = args.log_level
    
    # Word Config
    word_config = CHUNKING_CONFIG.get("word", {}).copy()
    if args.word_max_chunk_size is not None:
        word_config["max_chunk_size"] = args.word_max_chunk_size
    if args.word_vision_model is not None:
        word_config["vision_model"] = args.word_vision_model
    if args.word_process_images:
        word_config["process_images"] = True
    if args.word_no_process_images:
        word_config["process_images"] = False
    if args.log_level is not None:
        word_config["log_level"] = args.log_level
    
    # Code Config
    code_config = CHUNKING_CONFIG.get("code", {}).copy()
    if args.code_max_chunk_size is not None:
        code_config["max_chunk_size"] = args.code_max_chunk_size
    if args.code_min_chunk_size is not None:
        code_config["min_chunk_size"] = args.code_min_chunk_size
    if args.code_include_imports:
        code_config["include_imports"] = True
    if args.code_no_include_imports:
        code_config["include_imports"] = False
    if args.log_level is not None:
        code_config["log_level"] = args.log_level
    
    # === LOG CONFIGURATION ===
    logger.info("="*80)
    logger.info("DOCUMENT CHUNKING PIPELINE")
    logger.info("="*80)
    logger.info(f"Experiment: {EXPERIMENT_CONFIG['name']}")
    logger.info(f"Description: {EXPERIMENT_CONFIG['description']}")
    logger.info("")
    logger.info("Configuration:")
    logger.info(f"  Input directory: {input_dir}")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  File types: {', '.join(args.file_types)}")
    logger.info("")
    logger.info("PDF Configuration:")
    logger.info(f"  Image threshold: {pdf_config.get('image_coverage_threshold', 'N/A')}")
    logger.info(f"  Vision model: {pdf_config.get('vision_model', 'N/A')}")
    logger.info("")
    logger.info("Word Configuration:")
    logger.info(f"  Max chunk size: {word_config.get('max_chunk_size', 'N/A')}")
    logger.info(f"  Vision model: {word_config.get('vision_model', 'N/A')}")
    logger.info(f"  Process images: {word_config.get('process_images', 'N/A')}")
    logger.info("")
    logger.info("Code Configuration:")
    logger.info(f"  Max chunk size: {code_config.get('max_chunk_size', 'N/A')}")
    logger.info(f"  Min chunk size: {code_config.get('min_chunk_size', 'N/A')}")
    logger.info(f"  Include imports: {code_config.get('include_imports', 'N/A')}")
    logger.info("")
    
    # === VALIDATION ===
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1
    
    # === DISCOVER FILES ===
    logger.info("="*80)
    logger.info("DISCOVERING FILES")
    logger.info("="*80)
    
    files_by_type = get_files_by_type(input_dir, args.file_types)
    
    if not files_by_type:
        logger.warning(f"No supported files found in {input_dir}")
        logger.info(f"Supported file types: {', '.join(SUPPORTED_FILE_TYPES.keys())}")
        return 1
    
    # === PROCESS FILES BY TYPE ===
    all_chunks = []
    total_successful = 0
    total_failed = 0
    
    # Process PDF files
    if 'pdf' in files_by_type:
        chunks, successful, failed = process_pdf_files(
            files_by_type['pdf'],
            pdf_config
        )
        all_chunks.extend(chunks)
        total_successful += successful
        total_failed += failed
    
    # Process Word files
    if 'word' in files_by_type:
        chunks, successful, failed = process_word_files(
            files_by_type['word'],
            word_config
        )
        all_chunks.extend(chunks)
        total_successful += successful
        total_failed += failed
    
    # Process Code files
    if 'code' in files_by_type:
        chunks, successful, failed = process_code_files(
            files_by_type['code'],
            code_config
        )
        all_chunks.extend(chunks)
        total_successful += successful
        total_failed += failed
    
    # ========================================================================
    # SECTION: Add processors for other file types here
    # ========================================================================
    # 
    # if 'markdown' in files_by_type:
    #     chunks, successful, failed = process_markdown_files(
    #         files_by_type['markdown'],
    #         markdown_config
    #     )
    #     all_chunks.extend(chunks)
    #     total_successful += successful
    #     total_failed += failed
    #
    # if 'text' in files_by_type:
    #     chunks, successful, failed = process_text_files(
    #         files_by_type['text'],
    #         text_config
    #     )
    #     all_chunks.extend(chunks)
    #     total_successful += successful
    #     total_failed += failed
    #
    # ========================================================================
    
    # === SAVE CHUNKS ===
    if all_chunks:
        logger.info("")
        logger.info("="*80)
        logger.info("SAVING CHUNKS")
        logger.info("="*80)
        logger.info(f"Total chunks generated: {len(all_chunks)}")
        logger.info(f"Output file: {output_file}")
        
        try:
            # save_chunks expects List[Chunk] objects, not separate lists
            save_chunks(all_chunks, output_file)
            logger.info("[SUCCESS] - Chunks saved successfully")
        except Exception as e:
            logger.error(f"[ERROR] - Failed to save chunks: {e}")
            logger.exception("Full error traceback:")
            return 1
    else:
        logger.warning("No chunks were generated")
    
    # === FINAL SUMMARY ===
    logger.info("")
    logger.info("="*80)
    logger.info("CHUNKING PIPELINE COMPLETE")
    logger.info("="*80)
    logger.info(f"Files processed successfully: {total_successful}")
    logger.info(f"Files failed: {total_failed}")
    logger.info(f"Total chunks generated: {len(all_chunks)}")
    logger.info(f"Output saved to: {output_file}")
    logger.info("="*80)
    
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    load_dotenv()
    sys.exit(main())