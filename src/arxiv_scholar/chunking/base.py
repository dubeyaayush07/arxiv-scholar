"""Abstract Base Class for Chunkers.

This module defines the BaseChunker interface, which must be implemented
by all chunking strategies (e.g., sliding window, layout-aware, semantic).
"""

from abc import ABC, abstractmethod
from typing import Generator
from arxiv_scholar.schema import Document, Chunk


class BaseChunker(ABC):
    """Abstract Base Class defining the contract for all chunkers.

    All implementations must yield Chunk objects iteratively.
    """

    @abstractmethod
    def chunk(self, document: Document) -> Generator[Chunk, None, None]:
        """Iteratively chunks a document into discrete segments.

        Args:
            document: The input Document object.

        Yields:
            Chunk: The chunked segments of the document.
        """
        pass
