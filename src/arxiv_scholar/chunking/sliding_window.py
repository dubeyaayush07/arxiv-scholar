"""Sliding Window Chunker Implementation.

This module implements the SlidingWindowChunker, which splits text into
fixed-size chunks with a specified overlap. It is primarily used as a fallback
for overly large layout blocks.
"""

import hashlib
from typing import Generator
import logging

from arxiv_scholar.schema import Document, Chunk
from arxiv_scholar.chunking.base import BaseChunker

logger = logging.getLogger(__name__)


class SlidingWindowChunker(BaseChunker):
    """Splits a document's content by a fixed character size with overlap.

    Note: In a production setting, this would typically split by tokens
    using a tokenizer like tiktoken, but character splitting is used here
    for simplicity unless a tokenizer is injected.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        """Initializes the SlidingWindowChunker.

        Args:
            chunk_size: Maximum number of characters (or tokens) per chunk.
            chunk_overlap: Number of characters to overlap between chunks.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _hash_content(self, text: str) -> str:
        """Generates a stable ID for a chunk based on its text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def chunk(self, document: Document) -> Generator[Chunk, None, None]:
        """Yields overlapping chunks from the document's content."""
        text = document.content
        text_length = len(text)
        
        if text_length == 0:
            return

        chunk_index = 0
        start = 0
        
        while start < text_length:
            end = min(start + self.chunk_size, text_length)
            
            # If we're not at the end of the text, try to find a nice break point
            # (like a space or newline) to avoid cutting words in half.
            if end < text_length:
                # Look backwards for a space within a small window
                lookback_limit = max(start, end - 50)
                last_space = text.rfind(" ", lookback_limit, end)
                if last_space != -1:
                    end = last_space

            chunk_text = text[start:end].strip()
            
            if chunk_text:
                yield Chunk(
                    id=self._hash_content(chunk_text),
                    document_id=document.id,
                    content=chunk_text,
                    metadata={
                        **document.metadata,
                        "chunk_index": chunk_index,
                        "element_type": "Text",
                        "chunking_strategy": "sliding_window"
                    }
                )
                chunk_index += 1
            
            # Advance start pointer, accounting for overlap
            start = end - self.chunk_overlap
            
            # If we've reached the end of the text, break to avoid an infinite loop
            if end == text_length:
                break
            
            # Prevent infinite loop if overlap is somehow preventing progression
            if start <= end - self.chunk_size + self.chunk_overlap and end < text_length:
                start = end - self.chunk_overlap + 1 # force forward
