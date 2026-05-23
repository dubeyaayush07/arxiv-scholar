"""Qdrant Vector Store implementation.

Wraps the qdrant-client to provide upsert and search operations
against a Qdrant instance (remote server or in-memory for testing).
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from arxiv_scholar.schema import Chunk
from arxiv_scholar.storage.base import BaseVectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """Concrete vector store backed by Qdrant.

    Supports two modes:
      - **Server mode** (default): connects to a running Qdrant instance.
      - **In-memory mode**: pass `location=":memory:"` for unit tests.

    Args:
        collection_name: Name of the Qdrant collection.
        host: Qdrant server hostname (ignored in memory mode).
        port: Qdrant gRPC port (ignored in memory mode).
        location: If set to ":memory:", uses an ephemeral in-process store.
    """

    def __init__(
        self,
        collection_name: str,
        host: str = "localhost",
        port: int = 6333,
        location: Optional[str] = None,
    ) -> None:
        self.collection_name = collection_name

        if location == ":memory:":
            self._client = QdrantClient(location=":memory:")
            logger.info("QdrantVectorStore initialised in-memory (test mode).")
        else:
            self._client = QdrantClient(host=host, port=port)
            logger.info(
                "QdrantVectorStore connected to %s:%s, collection=%s",
                host, port, collection_name,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_collection(self, dimension: int) -> None:
        """Create the collection if it doesn't already exist."""
        collections = [
            c.name for c in self._client.get_collections().collections
        ]

        if self.collection_name in collections:
            logger.info("Collection '%s' already exists.", self.collection_name)
            return

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=Distance.COSINE,
            ),
        )
        logger.info(
            "Created collection '%s' (dim=%d, distance=COSINE).",
            self.collection_name, dimension,
        )

    def upsert(
        self,
        chunks: List[Chunk],
        vectors: List[List[float]],
    ) -> int:
        """Upsert chunks and their vectors into the collection.

        Each point stores:
          - id:      a deterministic UUID derived from the chunk's own id
          - vector:  the dense embedding
          - payload: {"content", "document_id", "metadata"}
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(vectors)} vectors."
            )

        points = [
            PointStruct(
                id=self._stable_uuid(chunk.id),
                vector=vector,
                payload={
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        self._client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        logger.debug("Upserted %d points into '%s'.", len(points), self.collection_name)
        return len(points)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Return the top-k nearest neighbours for a query vector."""
        hits = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        ).points

        return [
            {
                "id": hit.payload.get("chunk_id", str(hit.id)),
                "score": hit.score,
                "content": hit.payload.get("content", ""),
                "metadata": hit.payload.get("metadata", {}),
            }
            for hit in hits
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _stable_uuid(chunk_id: str) -> str:
        """Derive a deterministic UUID-v5 from the chunk's SHA-256 id.

        Qdrant requires either an integer or a UUID string as the point id.
        Using UUID-v5 (namespace + chunk_id) guarantees idempotent upserts:
        re-processing the same chunk will overwrite rather than duplicate.
        """
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
