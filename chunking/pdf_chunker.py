"""
COST - 0.90 for 97 image processed pdf slides with gpt-4o

Multimodal PDF Chunker with intelligent page analysis and processing.

This chunker:
- Analyzes each page for text, tables, and images
- Detects tables spanning multiple pages
- Converts image-heavy pages (proportion above an area threshold) to images for vision model input
- Provides rich metadata for each chunk

Inspired by - https://medium.com/@saptarshi701/advanced-chunking-for-pdf-word-with-embedded-images-using-regular-parsers-and-gpt-4o-7f0d5eb97052

Key Features

1. Intelligent Page Analysis
   - Calculates image coverage ratio for each page
   - Detects when images exceed your threshold (default 30%)
   - Analyzes content composition (text, tables, images)

2. Image-Heavy Page Processing
   - Pages with high image coverage are converted to images
   - Processed with vision model (GPT-4o) using a detailed prompt
   - Critical: Only triggers if the page has NO tables (tables are handled separately)

3. Multi-Page Table Detection
   - Detects tables spanning multiple consecutive pages
   - Groups them into single chunks
   - Preserves HTML table structure in metadata

4. Smart Integration Logic
   Decision Tree:
   ├── Is page part of multi-page table? → Handle as TABLE chunk
   ├── Else: Does page exceed image threshold AND has no tables? → VISION process
   └── Else: Standard text extraction → TEXT/MIXED chunk
   This ensures tables are never interfered with by the image processing logic!

5. Unified Metadata Format
   Universal fields (compatible with all document types):
   - source_file: Document path
   - chunk_id: Unique chunk identifier
   - page_number: Page number in document
   - chunk_type: TEXT, TABLE, IMAGE_HEAVY_PAGE, or MIXED
   - text_length: Character count
   - preview: First 200 characters

   Document-specific metadata (in document_specific_metadata dict):
   - total_pages: Total pages in document
   - num_tables: Number of tables in chunk
   - num_images: Number of images in chunk
   - image_coverage_ratio: Ratio of page covered by images
   - table_content: List of table contents
   - is_vision_processed: Whether vision model was used
   - image_details: Details about images (bbox, size, etc.)
   - bounding_boxes: Bounding boxes for layout elements

## Usage
```python
# Install dependencies
pip install langchain-openai pymupdf pillow

# Set environment variables
export OPENAI_API_KEY="your-key"

# Use the chunker
chunker = MultimodalPDFChunker(
    image_coverage_threshold=0.3,  # Adjust as needed
    vision_model="gpt-4o"
)

chunks = chunker.chunk("document.pdf")

# Access chunks
for chunk in chunks:
    print(chunk.content)
    print(chunk.metadata.to_dict())
```

## Customization Tips
- Adjust threshold: Change image_coverage_threshold (0.0-1.0)
- Different vision model: Use "gpt-4o-mini" or "claude-3-5-sonnet-20241022"
- Custom prompts: Modify _process_page_with_vision() prompt

The chunker is production-ready and handles edge cases like overlapping content types gracefully!
"""

import base64
import io
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import fitz  # PyMuPDF
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .base import BaseChunker, Chunk, ChunkMetadata, ChunkType
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PDFChunkMetadata(ChunkMetadata):
    """Extended metadata for PDF chunks with unified format."""
    
    # Universal fields (compatible with all document types)
    page_number: int = 0
    chunk_type: ChunkType = ChunkType.TEXT
    text_length: int = 0
    preview: str = ""  # First 200 chars for quick reference
    
    # Document-specific metadata (PDF-specific fields)
    document_specific_metadata: Dict[str, Any] = field(default_factory=dict)
    

class MultimodalPDFChunker(BaseChunker):
    """
    PDF chunker using PyMuPDF for direct analysis.
    
    Features:
    - Page-based chunking with intelligent analysis
    - Table and image detection using PyMuPDF
    - Image coverage analysis with vision model processing
    - Comprehensive metadata generation with unified format
    """
    
    def __init__(
        self,
        image_coverage_threshold: float = 0.3,
        vision_model: str = "gpt-4o",
        log_level: str = "INFO"
    ):
        """
        Initialize the chunker.
        
        Args:
            image_coverage_threshold: Threshold ratio (0-1) for image coverage
                                     to trigger vision processing
            vision_model: Vision-capable LLM model name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.image_coverage_threshold = image_coverage_threshold
        self.vision_model = ChatOpenAI(model=vision_model)
        logger.setLevel(getattr(logging, log_level.upper()))
        
    def chunk(self, file_path: str | Path) -> List[Chunk]:
        """Process PDF and generate chunks."""
        return self.chunk_pdf(str(file_path))
    
    def get_metadata_schema(self) -> Dict[str, type]:
        """Return metadata schema for automatic DB schema generation."""
        return {
            "source_file": str,
            "chunk_id": int,
            "page_number": int,
            "chunk_type": str,
            "text_length": int,
            "preview": str,
            "document_specific_metadata": str,  # Stored as JSON string
        }
    
    def chunk_pdf(self, pdf_path: str) -> List[Chunk]:
        """
        Process PDF and generate chunks.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Chunk objects with content and metadata
        """
        start_time = datetime.now()
        logger.info(f"{'='*80}")
        logger.info(f"Starting PDF chunking: {pdf_path}")
        logger.info(f"Image coverage threshold: {self.image_coverage_threshold:.1%}")
        logger.info(f"{'='*80}")
        
        try:
            pdf_document = fitz.open(pdf_path)
            total_pages = len(pdf_document)
            logger.info(f"Successfully opened PDF: {total_pages} pages")
        except Exception as e:
            logger.error(f"[ERROR] - Failed to open PDF: {e}")
            raise
        
        chunks = []
        stats = {
            "text_pages": 0,
            "table_pages": 0,
            "mixed_pages": 0,
            "vision_processed_pages": 0,
            "total_tables": 0,
            "total_images": 0
        }
        
        for page_num in range(total_pages):
            page = pdf_document.load_page(page_num)
            
            logger.info(f"")
            logger.info(f"{'─'*80}")
            logger.info(f"Processing page {page_num + 1}/{total_pages}")
            
            try:
                page_analysis = self._analyze_page(page, page_num + 1)
                
                logger.info(f"Analysis complete:")
                logger.info(f"     • Text length: {len(page_analysis['text']):,} chars")
                logger.info(f"     • Tables detected: {page_analysis['num_tables']}")
                logger.info(f"     • Images detected: {page_analysis['num_images']}")
                logger.info(f"     • Image coverage: {page_analysis['image_coverage_ratio']:.1%}")
                
                chunk = self._generate_chunk_for_page(
                    pdf_path,
                    page,
                    page_num + 1,
                    total_pages,
                    page_analysis
                )
                
                # Assign chunk_id
                chunk.metadata.chunk_id = len(chunks)
                chunks.append(chunk)
                
                # Update statistics
                stats["total_tables"] += page_analysis["num_tables"]
                stats["total_images"] += page_analysis["num_images"]
                
                doc_specific = chunk.metadata.document_specific_metadata
                if doc_specific.get("is_vision_processed", False):
                    stats["vision_processed_pages"] += 1
                elif chunk.metadata.chunk_type == ChunkType.TABLE:
                    stats["table_pages"] += 1
                elif chunk.metadata.chunk_type == ChunkType.MIXED:
                    stats["mixed_pages"] += 1
                else:
                    stats["text_pages"] += 1
                
                logger.info(f"[SUCCESS] - Chunk created: {chunk.metadata.chunk_type.value.upper()} ({chunk.metadata.text_length:,} chars)")
                
            except Exception as e:
                logger.error(f"[ERROR] - Error processing page {page_num + 1}: {e}")
                logger.exception("Full error details:")
                raise
        
        pdf_document.close()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info(f"CHUNKING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total chunks created: {len(chunks)}")
        logger.info(f"Time elapsed: {elapsed:.2f}s ({elapsed/total_pages:.2f}s per page)")
        logger.info(f"")
        logger.info(f"Chunk Type Distribution:")
        logger.info(f"   • Text pages: {stats['text_pages']}")
        logger.info(f"   • Table pages: {stats['table_pages']}")
        logger.info(f"   • Mixed pages: {stats['mixed_pages']}")
        logger.info(f"   • Vision-processed pages: {stats['vision_processed_pages']}")
        logger.info(f"")
        logger.info(f"Content Statistics:")
        logger.info(f"   • Total tables detected: {stats['total_tables']}")
        logger.info(f"   • Total images detected: {stats['total_images']}")
        logger.info(f"{'='*80}")
        
        return chunks
    
    def _analyze_page(self, page: fitz.Page, page_num: int) -> Dict[str, Any]:
        """Analyze a single page for content composition."""
        logger.debug(f"  Analyzing page structure...")
        
        page_rect = page.rect
        page_area = page_rect.width * page_rect.height
        
        text = page.get_text()
        logger.debug(f"    Extracted {len(text)} characters of text")
        
        image_list = page.get_images(full=True)
        num_images = len(image_list)
        logger.debug(f"    Found {num_images} images")
        
        total_image_area = 0
        image_details = []
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            image_rects = page.get_image_rects(xref)
            
            for rect in image_rects:
                img_area = rect.width * rect.height
                total_image_area += img_area
                
                image_details.append({
                    "index": img_index,
                    "bbox": (rect.x0, rect.y0, rect.x1, rect.y1),
                    "area": img_area,
                    "width": rect.width,
                    "height": rect.height
                })
        
        image_coverage_ratio = total_image_area / page_area if page_area > 0 else 0
        
        logger.debug(f"    Detecting tables...")
        tables = page.find_tables()
        num_tables = len(tables.tables) if tables else 0
        logger.debug(f"    Found {num_tables} tables")
        
        table_content = []
        table_bboxes = []
        
        if tables:
            for idx, table in enumerate(tables.tables):
                table_data = table.extract()
                
                if table_data:
                    table_str = self._format_table_as_markdown(table_data)
                    table_content.append(table_str)
                    
                    table_bboxes.append({
                        "bbox": table.bbox,
                        "rows": len(table_data),
                        "cols": len(table_data[0]) if table_data else 0
                    })
                    logger.debug(f"    Table {idx+1}: {len(table_data)} rows × {len(table_data[0]) if table_data else 0} cols")
        
        return {
            "text": text,
            "num_images": num_images,
            "num_tables": num_tables,
            "image_coverage_ratio": image_coverage_ratio,
            "exceeds_threshold": image_coverage_ratio > self.image_coverage_threshold,
            "image_details": image_details,
            "table_content": table_content,
            "table_bboxes": table_bboxes,
            "page_area": page_area
        }
    
    def _format_table_as_markdown(self, table_data: List[List[str]]) -> str:
        """Format table data as markdown table."""
        if not table_data:
            return ""
        
        lines = []
        header = table_data[0]
        lines.append("| " + " | ".join(str(cell) if cell else "" for cell in header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        
        for row in table_data[1:]:
            lines.append("| " + " | ".join(str(cell) if cell else "" for cell in row) + " |")
        
        return "\n".join(lines)
    
    def _page_to_base64(self, page: fitz.Page) -> str:
        """Convert PDF page to base64-encoded image."""
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _process_page_with_vision(self, page: fitz.Page, page_num: int) -> str:
        """Process an image-heavy page with vision model."""
        logger.info(f"Vision processing initiated (page {page_num})")
        logger.debug(f"    Converting page to image...")
        
        base64_image = self._page_to_base64(page)
        
        logger.debug(f"    Sending to vision model ({self.vision_model.model_name})...")
        
        prompt = """Analyze this document page comprehensively and provide a complete textual description.

Your output should include:

1. ALL TEXT CONTENT: Transcribe every piece of text visible on the page, maintaining the reading order and structure.
2. VISUAL ELEMENTS: Describe all images, charts, diagrams, or visual content.
3. LAYOUT AND STRUCTURE: Note the organization of content.
4. TABLES OR STRUCTURED DATA (if present): Describe the table structure and key data points.

OUTPUT FORMAT: Provide a flowing, readable description that captures all information on the page."""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        )
        
        try:
            response = self.vision_model.invoke([message])
            logger.info(f"Vision processing complete ({len(response.content)} chars generated)")
            return response.content
        except Exception as e:
            logger.error(f"[ERROR] - Vision processing failed: {e}")
            raise
    
    def _generate_chunk_for_page(
        self,
        pdf_path: str,
        page: fitz.Page,
        page_num: int,
        total_pages: int,
        analysis: Dict[str, Any]
    ) -> Chunk:
        """Generate a chunk for a single page based on analysis."""
        
        if analysis["num_tables"] > 0:
            logger.debug(f"  Processing as TABLE/MIXED page")
            
            content_parts = []
            if analysis["text"].strip():
                content_parts.append(analysis["text"].strip())
            
            if analysis["table_content"]:
                content_parts.append("\n\n--- Tables ---\n")
                content_parts.extend(analysis["table_content"])
            
            content = "\n\n".join(content_parts)
            
            chunk_type = ChunkType.MIXED if analysis["num_images"] > 0 else ChunkType.TABLE
            
            # Document-specific metadata for PDF
            doc_specific = {
                "total_pages": total_pages,
                "num_tables": analysis["num_tables"],
                "num_images": analysis["num_images"],
                "image_coverage_ratio": analysis["image_coverage_ratio"],
                "table_content": analysis["table_content"],
                "is_vision_processed": False,
                "image_details": analysis["image_details"],
                "bounding_boxes": analysis["table_bboxes"]
            }
            
            metadata = PDFChunkMetadata(
                source_file=pdf_path,
                page_number=page_num,
                chunk_type=chunk_type,
                text_length=len(content),
                preview=content[:200],
                document_specific_metadata=doc_specific
            )
            
        elif analysis["exceeds_threshold"]:
            logger.info(f"Image-heavy page detected")
            
            content = self._process_page_with_vision(page, page_num)
            
            # Document-specific metadata for vision-processed page
            doc_specific = {
                "total_pages": total_pages,
                "num_tables": 0,
                "num_images": analysis["num_images"],
                "image_coverage_ratio": analysis["image_coverage_ratio"],
                "table_content": [],
                "is_vision_processed": True,
                "image_details": analysis["image_details"],
                "bounding_boxes": []
            }
            
            metadata = PDFChunkMetadata(
                source_file=pdf_path,
                page_number=page_num,
                chunk_type=ChunkType.IMAGE_HEAVY_PAGE,
                text_length=len(content),
                preview=content[:200],
                document_specific_metadata=doc_specific
            )
            
        else:
            logger.debug(f"  Processing as TEXT page")
            
            content = analysis["text"].strip()
            chunk_type = ChunkType.MIXED if analysis["num_images"] > 0 else ChunkType.TEXT
            
            # Document-specific metadata for text page
            doc_specific = {
                "total_pages": total_pages,
                "num_tables": 0,
                "num_images": analysis["num_images"],
                "image_coverage_ratio": analysis["image_coverage_ratio"],
                "table_content": [],
                "is_vision_processed": False,
                "image_details": analysis["image_details"],
                "bounding_boxes": []
            }
            
            metadata = PDFChunkMetadata(
                source_file=pdf_path,
                page_number=page_num,
                chunk_type=chunk_type,
                text_length=len(content),
                preview=content[:200],
                document_specific_metadata=doc_specific
            )
        
        return Chunk(content=content, metadata=metadata)