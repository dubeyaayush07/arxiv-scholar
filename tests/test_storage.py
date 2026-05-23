"""Tests for the Qdrant vector store.

Uses qdrant-client's built-in in-memory mode so no Docker daemon is needed.
All embedding vectors are auto-generated random floats.
"""

import random
import uuid

import pytest

from arxiv_scholar.schema import Chunk
from arxiv_scholar.storage.qdrant_store import QdrantVectorStore

# ── Helpers ──────────────────────────────────────────────────────────────

DIMENSION = 8  # small dimension for fast tests


def _random_vector(dim: int = DIMENSION) -> list[float]:
    """Generate a random unit-ish vector."""
    return [random.uniform(-1, 1) for _ in range(dim)]


def _make_chunk(content: str, doc_id: str = "doc-1") -> Chunk:
    """Create a minimal Chunk for testing."""
    return Chunk(
        id=uuid.uuid4().hex,
        document_id=doc_id,
        content=content,
        metadata={"chunk_index": 0, "chunking_strategy": "test"},
    )


@pytest.fixture
def store() -> QdrantVectorStore:
    """Provides a fresh in-memory Qdrant store per test."""
    s = QdrantVectorStore(
        collection_name="test_collection",
        location=":memory:",
    )
    s.ensure_collection(dimension=DIMENSION)
    return s


# ── Tests ────────────────────────────────────────────────────────────────


class TestEnsureCollection:
    """Verify collection creation and idempotency."""

    def test_creates_collection(self, store: QdrantVectorStore):
        """Collection should exist after ensure_collection."""
        names = [
            c.name for c in store._client.get_collections().collections
        ]
        assert "test_collection" in names

    def test_idempotent(self, store: QdrantVectorStore):
        """Calling ensure_collection twice must not raise."""
        store.ensure_collection(dimension=DIMENSION)  # second call
        names = [
            c.name for c in store._client.get_collections().collections
        ]
        assert names.count("test_collection") == 1


class TestUpsert:
    """Verify upsert stores data correctly."""

    def test_single_chunk(self, store: QdrantVectorStore):
        chunk = _make_chunk("Attention is all you need.")
        vector = _random_vector()

        count = store.upsert([chunk], [vector])

        assert count == 1
        info = store._client.get_collection("test_collection")
        assert info.points_count == 1

    def test_batch_upsert(self, store: QdrantVectorStore):
        chunks = [_make_chunk(f"chunk-{i}") for i in range(10)]
        vectors = [_random_vector() for _ in range(10)]

        count = store.upsert(chunks, vectors)

        assert count == 10
        info = store._client.get_collection("test_collection")
        assert info.points_count == 10

    def test_idempotent_upsert(self, store: QdrantVectorStore):
        """Re-upserting the same chunk must overwrite, not duplicate."""
        chunk = _make_chunk("Duplicate test")
        vector_v1 = _random_vector()
        vector_v2 = _random_vector()

        store.upsert([chunk], [vector_v1])
        store.upsert([chunk], [vector_v2])

        info = store._client.get_collection("test_collection")
        assert info.points_count == 1  # not 2

    def test_mismatched_lengths_raises(self, store: QdrantVectorStore):
        """chunks and vectors must be the same length."""
        chunks = [_make_chunk("a"), _make_chunk("b")]
        vectors = [_random_vector()]  # only one vector

        with pytest.raises(ValueError, match="Mismatch"):
            store.upsert(chunks, vectors)


class TestSearch:
    """Verify search returns sensible results."""

    def test_returns_results(self, store: QdrantVectorStore):
        chunks = [_make_chunk(f"doc-{i}") for i in range(5)]
        vectors = [_random_vector() for _ in range(5)]
        store.upsert(chunks, vectors)

        results = store.search(query_vector=_random_vector(), top_k=3)

        assert len(results) == 3
        for r in results:
            assert "id" in r
            assert "score" in r
            assert "content" in r
            assert "metadata" in r

    def test_top_k_limits_results(self, store: QdrantVectorStore):
        chunks = [_make_chunk(f"doc-{i}") for i in range(20)]
        vectors = [_random_vector() for _ in range(20)]
        store.upsert(chunks, vectors)

        results = store.search(query_vector=_random_vector(), top_k=5)
        assert len(results) == 5

    def test_nearest_neighbour_ordering(self, store: QdrantVectorStore):
        """The known-similar vector should rank first."""
        # Create a known query vector
        query = [1.0] * DIMENSION

        # One chunk very close to the query, rest are random
        close_chunk = _make_chunk("I am the closest")
        close_vector = [0.99] * DIMENSION  # nearly identical

        far_chunks = [_make_chunk(f"far-{i}") for i in range(5)]
        far_vectors = [[-1.0] * DIMENSION for _ in range(5)]

        store.upsert([close_chunk] + far_chunks, [close_vector] + far_vectors)

        results = store.search(query_vector=query, top_k=3)

        assert results[0]["content"] == "I am the closest"
        assert results[0]["score"] > results[-1]["score"]

    def test_payload_integrity(self, store: QdrantVectorStore):
        """Stored payload fields must round-trip correctly."""
        chunk = Chunk(
            id="abc123",
            document_id="paper-42",
            content="The loss function is defined as...",
            metadata={"chunk_index": 7, "chunking_strategy": "layout_aware"},
        )
        store.upsert([chunk], [_random_vector()])

        results = store.search(query_vector=_random_vector(), top_k=1)

        assert len(results) == 1
        hit = results[0]
        assert hit["content"] == "The loss function is defined as..."
        assert hit["metadata"]["chunk_index"] == 7
        assert hit["metadata"]["chunking_strategy"] == "layout_aware"
