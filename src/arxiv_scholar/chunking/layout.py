"""Layout-Aware Chunker Implementation.

This module implements the LayoutAwareChunker, which uses Docling to visually
parse PDF layouts and semantically group content (e.g., Headers with Paragraphs,
Tables kept intact).
"""

import hashlib
import logging
from typing import Generator

from arxiv_scholar.schema import Document, Chunk
from arxiv_scholar.chunking.base import BaseChunker
from arxiv_scholar.chunking.sliding_window import SlidingWindowChunker

logger = logging.getLogger(__name__)

try:
    import torch
    # Apple Silicon MPS has a float64 bug in rt_detr_v2. 
    # We must monkey-patch torch to trick docling into running entirely on CPU.
    # This must happen at the module level BEFORE docling is ever imported.
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False
except ImportError:
    pass


class LayoutAwareChunker(BaseChunker):
    """Chunks documents based on their visual and structural layout.
    
    Relies on `docling` to parse the underlying PDF file and group
    sections logically. Falls back to SlidingWindowChunker if a layout
    block exceeds the maximum token/character limit.
    """

    def __init__(self, max_chunk_size: int = 1500) -> None:
        """Initializes the LayoutAwareChunker.
        
        Args:
            max_chunk_size: Maximum allowed size for a single layout chunk.
                            Blocks larger than this will be processed by the fallback chunker.
        """
        self.max_chunk_size = max_chunk_size
        self.fallback_chunker = SlidingWindowChunker(
            chunk_size=self.max_chunk_size, 
            chunk_overlap=200
        )
        
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
            from docling.datamodel.base_models import InputFormat
            from docling.chunking import HierarchicalChunker
            
            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=1, device=AcceleratorDevice.CPU
            )
            
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            self._hierarchical_chunker = HierarchicalChunker()
            self._is_ready = True
        except ImportError:
            logger.error(
                "docling is not installed. Please install it with `pip install docling` "
                "to use the LayoutAwareChunker."
            )
            self._is_ready = False

    def _hash_content(self, text: str) -> str:
        """Generates a stable ID for a chunk based on its text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def chunk(self, document: Document) -> Generator[Chunk, None, None]:
        """Yields layout-aware chunks from the document's source file."""
        if not self._is_ready:
            logger.warning("Docling not available. Falling back to sliding window chunking.")
            yield from self.fallback_chunker.chunk(document)
            return

        source_path = document.metadata.get("source_path")
        if not source_path:
            logger.warning(
                f"Document {document.id} missing 'source_path' metadata. "
                "LayoutAwareChunker requires the original file path."
            )
            return

        try:
            # Convert PDF into Docling's internal representation
            dl_doc = self._converter.convert(source_path).document
            
            # Use Docling's hierarchical chunker
            chunk_iter = self._hierarchical_chunker.chunk(dl_doc)
            
            chunk_index = 0
            for docling_chunk in chunk_iter:
                text = docling_chunk.text
                
                if not text or not text.strip():
                    continue
                
                # Check if this layout chunk is too large
                if len(text) > self.max_chunk_size:
                    logger.debug(
                        f"Layout block too large ({len(text)} chars). "
                        "Falling back to sliding window."
                    )
                    # Create a temporary document just for the fallback chunker
                    temp_doc = Document(
                        id=document.id,
                        content=text,
                        metadata=document.metadata
                    )
                    
                    for sub_chunk in self.fallback_chunker.chunk(temp_doc):
                        # Override the chunk_index logic to fit our master stream
                        sub_chunk.metadata["chunk_index"] = chunk_index
                        sub_chunk.metadata["chunking_strategy"] = "layout_aware_fallback"
                        yield sub_chunk
                        chunk_index += 1
                else:
                    # Normal layout-aware chunk
                    yield Chunk(
                        id=self._hash_content(text),
                        document_id=document.id,
                        content=text,
                        metadata={
                            **document.metadata,
                            "chunk_index": chunk_index,
                            "element_type": "LayoutBlock",
                            "chunking_strategy": "layout_aware"
                        }
                    )
                    chunk_index += 1
                    
        except Exception as e:
            logger.error(f"Failed to chunk document {document.id}: {e}", exc_info=True)
            return
