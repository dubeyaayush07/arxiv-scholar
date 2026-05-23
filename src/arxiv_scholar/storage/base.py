"""Abstract Base Class for Vector Stores.

Defines the interface that all storage backends must implement.
This keeps the pipeline decoupled from any specific database.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any

from arxiv_scholar.schema import Chunk


class BaseVectorStore(ABC):
    """Contract for vector store backends (Qdrant, LanceDB, etc.).

    Every implementation must support:
      - Creating/ensuring a collection exists.
      - Upserting chunks with their pre-computed embedding vectors.
      - Searching by vector similarity.
    """

    @abstractmethod
    def ensure_collection(self, dimension: int) -> None:
        """Create the collection if it does not already exist.

        Args:
            dimension: The dimensionality of the embedding vectors.
        """

    @abstractmethod
    def upsert(
        self,
        chunks: List[Chunk],
        vectors: List[List[float]],
    ) -> int:
        """Insert or update chunks with their embedding vectors.

        Args:
            chunks: The text chunks to store.
            vectors: Corresponding embedding vectors (same length as chunks).

        Returns:
            The number of points successfully upserted.
        """

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find the nearest neighbours to a query vector.

        Args:
            query_vector: The dense vector to search with.
            top_k: Number of results to return.

        Returns:
            A list of dicts, each containing at minimum:
              - "id": the chunk id
              - "score": the similarity score
              - "content": the chunk text
              - "metadata": the chunk metadata
        """
