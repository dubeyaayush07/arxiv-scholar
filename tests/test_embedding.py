"""Unit tests for the embedding layer."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from arxiv_scholar.embedding.base import BaseEmbedder


def test_base_embedder_cannot_be_instantiated() -> None:
    """Verifies that BaseEmbedder cannot be directly instantiated."""
    with pytest.raises(TypeError):
        BaseEmbedder()  # type: ignore


@patch("arxiv_scholar.embedding.st_embedder._resolve_device", return_value="cpu")
@patch("sentence_transformers.SentenceTransformer")
def test_st_embedder_embed(mock_st_class: MagicMock, mock_device: MagicMock) -> None:
    """Verifies SentenceTransformerEmbedder produces correctly shaped output."""
    # Setup mock model
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 1024
    mock_model.encode.return_value = np.random.rand(3, 1024).astype(np.float32)
    mock_st_class.return_value = mock_model

    from arxiv_scholar.embedding.st_embedder import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(model_name="BAAI/bge-m3", device="cpu")

    # Embed 3 texts
    texts = ["hello world", "arXiv paper", "quantum computing"]
    vectors = embedder.embed(texts)

    # Verify shape and types
    assert len(vectors) == 3
    assert len(vectors[0]) == 1024
    assert all(isinstance(v, float) for v in vectors[0])

    # Verify properties
    assert embedder.dimension == 1024
    assert embedder.model_name == "BAAI/bge-m3"
    assert embedder.device == "cpu"


@patch("arxiv_scholar.embedding.st_embedder._resolve_device", return_value="cpu")
@patch("sentence_transformers.SentenceTransformer")
def test_st_embedder_empty_input(mock_st_class: MagicMock, mock_device: MagicMock) -> None:
    """Verifies that embedding an empty list returns an empty list."""
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 1024
    mock_st_class.return_value = mock_model

    from arxiv_scholar.embedding.st_embedder import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder(model_name="BAAI/bge-m3", device="cpu")
    vectors = embedder.embed([])

    assert vectors == []
    mock_model.encode.assert_not_called()


def test_resolve_device_explicit_override() -> None:
    """Verifies that passing an explicit device skips auto-detection."""
    from arxiv_scholar.embedding.st_embedder import _resolve_device

    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda") == "cuda"
    assert _resolve_device("mps") == "mps"
