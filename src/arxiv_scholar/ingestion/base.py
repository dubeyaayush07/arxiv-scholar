"""Abstract Base Class for Document Readers.

This module defines the DocumentReader interface, which must be implemented
by all document sources (e.g., local storage, cloud buckets, databases).
"""

from abc import ABC, abstractmethod
from typing import Generator
from arxiv_scholar.schema import Document


class DocumentReader(ABC):
    """Abstract Base Class defining the contract for all document readers.

    All implementations must yield Document objects iteratively (generator pattern)
    to support memory-efficient extraction over extremely large corpora.
    """

    @abstractmethod
    def read(self) -> Generator[Document, None, None]:
        """Iteratively reads documents from the configured source.

        Yields:
            Document: The parsed document schema with content and metadata.

        Raises:
            Exception: Implementations should raise relevant exceptions or handle
                       them defensively and continue processing, as appropriate.
        """
        pass
