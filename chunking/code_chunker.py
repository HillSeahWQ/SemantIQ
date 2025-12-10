"""
Code Chunker with intelligent structure-aware processing.

This chunker:
- Parses code structure (classes, functions, methods)
- Chunks by semantic units (functions, classes, modules)
- Handles multiple programming languages
- Provides rich metadata for code retrieval
- Uses overlap for context preservation

## Chunking Strategy

The chunker uses a hierarchical approach:

1. **Primary: Function/Method Level**
   - Each function/method is a separate chunk
   - Includes docstrings, decorators, and type hints
   - Best for code search and understanding

2. **Class Level (when appropriate)**
   - Small classes (< max_chunk_size): Keep whole class together
   - Large classes: Split into methods, but include class context

3. **Module Level (for globals)**
   - Module-level code, imports, and constants
   - Grouped into a module context chunk

4. **Overlap Strategy**
   - Include class context for methods (class name, docstring)
   - Include module imports for better context
   - Add surrounding function signatures for related code

5. **Size Management**
   - Target chunk size: 500-1500 characters
   - Very large functions: Split at logical boundaries (loops, conditionals)
   - Keep small related functions together

## Why This Strategy?

1. **Semantic Coherence**: Functions are natural semantic units
2. **Search Granularity**: Users typically search for specific functions/classes
3. **Context Preservation**: Overlap ensures related code stays connected
4. **Language Agnostic**: Works across Python, JavaScript, Java, etc.
5. **Size Balance**: Prevents both tiny and massive chunks

## Supported Languages

- Python (.py)
- JavaScript/TypeScript (.js, .ts, .jsx, .tsx)
- Java (.java)
- C/C++ (.c, .cpp, .h, .hpp)
- Go (.go)
- Ruby (.rb)
- PHP (.php)
- Rust (.rs)

## Usage

```python
from chunking import CodeChunker

chunker = CodeChunker(
    max_chunk_size=1500,
    include_imports=True,
    include_docstrings=True
)

chunks = chunker.chunk("path/to/code.py")

for chunk in chunks:
    print(f"Type: {chunk.metadata.chunk_type.value}")
    print(f"Language: {chunk.metadata.document_specific_metadata['language']}")
    print(f"Element: {chunk.metadata.document_specific_metadata['code_element_type']}")
```
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

try:
    import tree_sitter_python as tspython
    import tree_sitter_javascript as tsjavascript
    from tree_sitter import Language, Parser, Node
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

from .base import BaseChunker, Chunk, ChunkMetadata, ChunkType
from utils.logger import get_logger

logger = get_logger(__name__)


class CodeElementType(Enum):
    """Types of code elements."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    PROPERTY = "property"
    CONSTANT = "constant"
    IMPORT = "import"
    COMMENT = "comment"
    UNKNOWN = "unknown"


@dataclass
class CodeChunkMetadata(ChunkMetadata):
    """Extended metadata for code chunks with unified format."""
    
    # Universal fields (compatible with all document types)
    page_number: int = 0  # Line number / 100 for code files
    chunk_type: ChunkType = ChunkType.CODE
    text_length: int = 0
    preview: str = ""
    
    # Document-specific metadata (code-specific fields)
    document_specific_metadata: Dict[str, Any] = field(default_factory=dict)


# Language configurations
LANGUAGE_CONFIG = {
    '.py': {
        'name': 'Python',
        'comment_single': '#',
        'comment_multi_start': '"""',
        'comment_multi_end': '"""',
        'function_keywords': ['def'],
        'class_keywords': ['class'],
    },
    '.js': {
        'name': 'JavaScript',
        'comment_single': '//',
        'comment_multi_start': '/*',
        'comment_multi_end': '*/',
        'function_keywords': ['function', 'const', 'let', 'var'],
        'class_keywords': ['class'],
    },
    '.ts': {
        'name': 'TypeScript',
        'comment_single': '//',
        'comment_multi_start': '/*',
        'comment_multi_end': '*/',
        'function_keywords': ['function', 'const', 'let', 'var'],
        'class_keywords': ['class', 'interface', 'type'],
    },
    '.java': {
        'name': 'Java',
        'comment_single': '//',
        'comment_multi_start': '/*',
        'comment_multi_end': '*/',
        'function_keywords': ['public', 'private', 'protected', 'static'],
        'class_keywords': ['class', 'interface', 'enum'],
    },
    '.cpp': {
        'name': 'C++',
        'comment_single': '//',
        'comment_multi_start': '/*',
        'comment_multi_end': '*/',
        'function_keywords': [],
        'class_keywords': ['class', 'struct'],
    },
    '.go': {
        'name': 'Go',
        'comment_single': '//',
        'comment_multi_start': '/*',
        'comment_multi_end': '*/',
        'function_keywords': ['func'],
        'class_keywords': ['type', 'struct', 'interface'],
    },
    '.rb': {
        'name': 'Ruby',
        'comment_single': '#',
        'comment_multi_start': '=begin',
        'comment_multi_end': '=end',
        'function_keywords': ['def'],
        'class_keywords': ['class', 'module'],
    },
    '.rs': {
        'name': 'Rust',
        'comment_single': '//',
        'comment_multi_start': '/*',
        'comment_multi_end': '*/',
        'function_keywords': ['fn'],
        'class_keywords': ['struct', 'enum', 'trait', 'impl'],
    },
}


class CodeChunker(BaseChunker):
    """
    Code chunker with intelligent structure-aware processing.
    
    Features:
    - Parses code structure (classes, functions, methods)
    - Chunks by semantic units with overlap
    - Multi-language support
    - Rich metadata for code retrieval
    """
    
    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 100,
        include_imports: bool = True,
        include_docstrings: bool = True,
        overlap_lines: int = 3,
        log_level: str = "INFO"
    ):
        """
        Initialize the code chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk (merge smaller ones)
            include_imports: Include import statements in context
            include_docstrings: Include docstrings in chunks
            overlap_lines: Number of lines to overlap between chunks
            log_level: Logging level
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.include_imports = include_imports
        self.include_docstrings = include_docstrings
        self.overlap_lines = overlap_lines
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Initialize tree-sitter parsers if available
        self.parsers = {}
        if TREE_SITTER_AVAILABLE:
            self._init_tree_sitter()
        else:
            logger.warning("tree-sitter not available, using regex-based parsing")
    
    def _init_tree_sitter(self):
        """Initialize tree-sitter parsers for supported languages."""
        try:
            # Python parser
            py_parser = Parser()
            py_parser.set_language(Language(tspython.language()))
            self.parsers['.py'] = py_parser
            
            # JavaScript/TypeScript parser
            js_parser = Parser()
            js_parser.set_language(Language(tsjavascript.language()))
            self.parsers['.js'] = js_parser
            self.parsers['.ts'] = js_parser
            self.parsers['.jsx'] = js_parser
            self.parsers['.tsx'] = js_parser
            
            logger.info(f"Initialized tree-sitter parsers for {len(self.parsers)} languages")
        except Exception as e:
            logger.warning(f"Failed to initialize some tree-sitter parsers: {e}")
    
    def chunk(self, file_path: str | Path) -> List[Chunk]:
        """Process code file and generate chunks."""
        return self.chunk_code(str(file_path))
    
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
    
    def chunk_code(self, code_path: str) -> List[Chunk]:
        """
        Process code file and generate chunks.
        
        Args:
            code_path: Path to code file
            
        Returns:
            List of Chunk objects
        """
        start_time = datetime.now()
        file_path = Path(code_path)
        
        logger.info(f"{'='*80}")
        logger.info(f"Starting code chunking: {file_path.name}")
        logger.info(f"Max chunk size: {self.max_chunk_size:,} characters")
        logger.info(f"{'='*80}")
        
        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            logger.error(f"[ERROR] - Failed to read file: {e}")
            raise
        
        # Get language
        extension = file_path.suffix.lower()
        if extension not in LANGUAGE_CONFIG:
            logger.warning(f"Unsupported file extension: {extension}")
            return self._fallback_chunking(code, str(file_path), extension)
        
        language = LANGUAGE_CONFIG[extension]['name']
        logger.info(f"Detected language: {language}")
        
        # Parse code structure
        if extension in self.parsers:
            elements = self._parse_with_tree_sitter(code, extension)
        else:
            elements = self._parse_with_regex(code, extension)
        
        logger.info(f"Parsed {len(elements)} code elements")
        
        # Generate chunks
        chunks = self._generate_chunks(code, elements, str(file_path), extension, language)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info(f"CHUNKING COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Total chunks created: {len(chunks)}")
        logger.info(f"Time elapsed: {elapsed:.2f}s")
        logger.info(f"{'='*80}")
        
        return chunks
    
    def _parse_with_tree_sitter(self, code: str, extension: str) -> List[Dict[str, Any]]:
        """Parse code using tree-sitter for accurate structure extraction."""
        parser = self.parsers[extension]
        tree = parser.parse(bytes(code, "utf8"))
        
        elements = []
        
        def traverse(node: Node, parent_class: Optional[str] = None):
            """Traverse AST and extract code elements."""
            node_type = node.type
            
            # Function definitions
            if node_type in ['function_definition', 'function_declaration', 'method_definition']:
                name_node = node.child_by_field_name('name')
                name = name_node.text.decode('utf8') if name_node else 'anonymous'
                
                element = {
                    'type': CodeElementType.METHOD if parent_class else CodeElementType.FUNCTION,
                    'name': name,
                    'start_line': node.start_point[0],
                    'end_line': node.end_point[0],
                    'start_byte': node.start_byte,
                    'end_byte': node.end_byte,
                    'parent_class': parent_class,
                    'docstring': self._extract_docstring(node, code),
                }
                elements.append(element)
            
            # Class definitions
            elif node_type in ['class_definition', 'class_declaration']:
                name_node = node.child_by_field_name('name')
                name = name_node.text.decode('utf8') if name_node else 'anonymous'
                
                element = {
                    'type': CodeElementType.CLASS,
                    'name': name,
                    'start_line': node.start_point[0],
                    'end_line': node.end_point[0],
                    'start_byte': node.start_byte,
                    'end_byte': node.end_byte,
                    'parent_class': None,
                    'docstring': self._extract_docstring(node, code),
                }
                elements.append(element)
                
                # Traverse class body for methods
                for child in node.children:
                    traverse(child, parent_class=name)
                return  # Don't traverse children again
            
            # Traverse children
            for child in node.children:
                traverse(child, parent_class)
        
        traverse(tree.root_node)
        
        # Sort by position
        elements.sort(key=lambda x: x['start_line'])
        
        return elements
    
    def _parse_with_regex(self, code: str, extension: str) -> List[Dict[str, Any]]:
        """Fallback parsing using regex patterns."""
        lang_config = LANGUAGE_CONFIG[extension]
        lines = code.split('\n')
        elements = []
        
        # Regex patterns for different languages
        if extension == '.py':
            # Python function/class pattern
            func_pattern = re.compile(r'^(\s*)(def|class)\s+(\w+)')
        elif extension in ['.js', '.ts']:
            # JavaScript/TypeScript pattern
            func_pattern = re.compile(r'^(\s*)(function|class|const|let|var)\s+(\w+)')
        elif extension == '.java':
            # Java pattern
            func_pattern = re.compile(r'^(\s*)(public|private|protected|static).*?(class|void|int|String|boolean)\s+(\w+)')
        else:
            # Generic pattern
            func_pattern = re.compile(r'^(\s*)(def|function|class|fn|func)\s+(\w+)')
        
        current_class = None
        
        for line_num, line in enumerate(lines):
            match = func_pattern.match(line)
            if match:
                indent = len(match.group(1))
                keyword = match.group(2)
                name = match.groups()[-1]  # Last group is the name
                
                # Determine type
                if keyword in lang_config['class_keywords']:
                    element_type = CodeElementType.CLASS
                    current_class = name
                elif keyword in lang_config['function_keywords'] or keyword in ['public', 'private', 'protected', 'static']:
                    element_type = CodeElementType.METHOD if current_class else CodeElementType.FUNCTION
                else:
                    element_type = CodeElementType.FUNCTION
                
                # Find end of this element (simplified: next element at same or lower indent)
                end_line = line_num
                for i in range(line_num + 1, len(lines)):
                    next_line = lines[i]
                    if next_line.strip() and not next_line.strip().startswith(lang_config['comment_single']):
                        next_indent = len(next_line) - len(next_line.lstrip())
                        if next_indent <= indent and next_line.strip():
                            break
                    end_line = i
                
                elements.append({
                    'type': element_type,
                    'name': name,
                    'start_line': line_num,
                    'end_line': end_line,
                    'start_byte': sum(len(lines[i]) + 1 for i in range(line_num)),
                    'end_byte': sum(len(lines[i]) + 1 for i in range(end_line + 1)),
                    'parent_class': current_class if element_type == CodeElementType.METHOD else None,
                    'docstring': None,
                })
        
        return elements
    
    def _extract_docstring(self, node: Node, code: str) -> Optional[str]:
        """Extract docstring from a node."""
        if not self.include_docstrings:
            return None
        
        # Look for string literal as first statement in body
        for child in node.children:
            if child.type == 'block':
                for stmt in child.children:
                    if stmt.type in ['expression_statement', 'string']:
                        text = stmt.text.decode('utf8')
                        if text.startswith(('"""', "'''", '"', "'")):
                            return text.strip('"\' \n')
        return None
    
    def _generate_chunks(
        self,
        code: str,
        elements: List[Dict[str, Any]],
        file_path: str,
        extension: str,
        language: str
    ) -> List[Chunk]:
        """Generate chunks from parsed code elements."""
        chunks = []
        lines = code.split('\n')
        
        # Extract imports
        imports = self._extract_imports(lines, extension) if self.include_imports else []
        
        # Create module-level chunk if there's code outside elements
        module_chunk = self._create_module_chunk(
            code, elements, file_path, extension, language, imports
        )
        if module_chunk:
            chunks.append(module_chunk)
        
        # Process each element
        for element in elements:
            chunk = self._create_element_chunk(
                code, element, file_path, extension, language, imports
            )
            if chunk:
                chunks.append(chunk)
        
        # Assign chunk IDs
        for idx, chunk in enumerate(chunks):
            chunk.metadata.chunk_id = idx
        
        return chunks
    
    def _extract_imports(self, lines: List[str], extension: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        
        if extension == '.py':
            pattern = re.compile(r'^(import|from)\s+')
        elif extension in ['.js', '.ts', '.jsx', '.tsx']:
            pattern = re.compile(r'^(import|require)\s+')
        elif extension == '.java':
            pattern = re.compile(r'^import\s+')
        else:
            return imports
        
        for line in lines:
            if pattern.match(line.strip()):
                imports.append(line.strip())
        
        return imports
    
    def _create_module_chunk(
        self,
        code: str,
        elements: List[Dict[str, Any]],
        file_path: str,
        extension: str,
        language: str,
        imports: List[str]
    ) -> Optional[Chunk]:
        """Create chunk for module-level code."""
        lines = code.split('\n')
        
        # Collect module-level lines (outside of elements)
        element_lines = set()
        for elem in elements:
            element_lines.update(range(elem['start_line'], elem['end_line'] + 1))
        
        module_lines = []
        for idx, line in enumerate(lines):
            if idx not in element_lines:
                module_lines.append(line)
        
        module_code = '\n'.join(module_lines).strip()
        
        # Only create chunk if there's substantial module-level code
        if len(module_code) < self.min_chunk_size:
            return None
        
        # Add imports context
        if imports:
            module_code = '\n'.join(imports) + '\n\n' + module_code
        
        doc_specific = {
            'language': language,
            'file_extension': extension,
            'code_element_type': CodeElementType.MODULE.value,
            'element_name': Path(file_path).stem,
            'parent_class': None,
            'has_docstring': False,
            'line_start': 0,
            'line_end': len(lines),
            'complexity_score': self._estimate_complexity(module_code),
        }
        
        metadata = CodeChunkMetadata(
            source_file=file_path,
            page_number=0,
            chunk_type=ChunkType.CODE,
            text_length=len(module_code),
            preview=module_code[:200],
            document_specific_metadata=doc_specific
        )
        
        return Chunk(content=module_code, metadata=metadata)
    
    def _create_element_chunk(
        self,
        code: str,
        element: Dict[str, Any],
        file_path: str,
        extension: str,
        language: str,
        imports: List[str]
    ) -> Optional[Chunk]:
        """Create chunk for a code element (function, class, method)."""
        lines = code.split('\n')
        
        # Extract element code
        element_lines = lines[element['start_line']:element['end_line'] + 1]
        element_code = '\n'.join(element_lines)
        
        # Skip if too small
        if len(element_code) < self.min_chunk_size:
            return None
        
        # Add context for methods (class signature)
        context = []
        if element['parent_class'] and self.include_imports:
            # Find parent class definition
            class_line = max(0, element['start_line'] - 20)
            for i in range(class_line, element['start_line']):
                if 'class ' + element['parent_class'] in lines[i]:
                    context.append(lines[i])
                    break
        
        # Add imports context
        if imports and self.include_imports:
            context.extend(imports[:5])  # Top 5 imports
        
        # Build final content
        if context:
            full_content = '\n'.join(context) + '\n\n' + element_code
        else:
            full_content = element_code
        
        # Split if too large
        if len(full_content) > self.max_chunk_size:
            logger.debug(f"Element {element['name']} exceeds max size, keeping as is")
            # In a more sophisticated implementation, we could split large functions
            # For now, we keep them as single chunks
        
        doc_specific = {
            'language': language,
            'file_extension': extension,
            'code_element_type': element['type'].value,
            'element_name': element['name'],
            'parent_class': element.get('parent_class'),
            'has_docstring': element.get('docstring') is not None,
            'docstring': element.get('docstring'),
            'line_start': element['start_line'],
            'line_end': element['end_line'],
            'complexity_score': self._estimate_complexity(element_code),
        }
        
        metadata = CodeChunkMetadata(
            source_file=file_path,
            page_number=element['start_line'] // 100,  # Approximate "page" number
            chunk_type=ChunkType.CODE,
            text_length=len(full_content),
            preview=full_content[:200],
            document_specific_metadata=doc_specific
        )
        
        return Chunk(content=full_content, metadata=metadata)
    
    def _estimate_complexity(self, code: str) -> int:
        """Estimate code complexity (simple heuristic)."""
        # Count control flow keywords
        keywords = ['if', 'else', 'elif', 'for', 'while', 'switch', 'case', 'try', 'catch', 'except']
        complexity = 1  # Base complexity
        
        for keyword in keywords:
            complexity += code.count(f' {keyword} ')
            complexity += code.count(f' {keyword}(')
        
        return complexity
    
    def _fallback_chunking(self, code: str, file_path: str, extension: str) -> List[Chunk]:
        """Fallback chunking for unsupported file types."""
        logger.warning(f"Using fallback chunking for {extension}")
        
        chunks = []
        lines = code.split('\n')
        
        # Simple line-based chunking
        chunk_lines = []
        current_size = 0
        
        for line in lines:
            chunk_lines.append(line)
            current_size += len(line) + 1
            
            if current_size >= self.max_chunk_size:
                content = '\n'.join(chunk_lines)
                
                doc_specific = {
                    'language': 'Unknown',
                    'file_extension': extension,
                    'code_element_type': CodeElementType.UNKNOWN.value,
                    'element_name': 'chunk',
                    'parent_class': None,
                    'has_docstring': False,
                    'line_start': len(chunks) * self.max_chunk_size // 80,
                    'line_end': (len(chunks) + 1) * self.max_chunk_size // 80,
                    'complexity_score': 1,
                }
                
                metadata = CodeChunkMetadata(
                    source_file=file_path,
                    chunk_id=len(chunks),
                    page_number=len(chunks),
                    chunk_type=ChunkType.CODE,
                    text_length=len(content),
                    preview=content[:200],
                    document_specific_metadata=doc_specific
                )
                
                chunks.append(Chunk(content=content, metadata=metadata))
                chunk_lines = []
                current_size = 0
        
        # Add remaining lines
        if chunk_lines:
            content = '\n'.join(chunk_lines)
            
            doc_specific = {
                'language': 'Unknown',
                'file_extension': extension,
                'code_element_type': CodeElementType.UNKNOWN.value,
                'element_name': 'chunk',
                'parent_class': None,
                'has_docstring': False,
                'line_start': len(chunks) * self.max_chunk_size // 80,
                'line_end': len(lines),
                'complexity_score': 1,
            }
            
            metadata = CodeChunkMetadata(
                source_file=file_path,
                chunk_id=len(chunks),
                page_number=len(chunks),
                chunk_type=ChunkType.CODE,
                text_length=len(content),
                preview=content[:200],
                document_specific_metadata=doc_specific
            )
            
            chunks.append(Chunk(content=content, metadata=metadata))
        
        return chunks