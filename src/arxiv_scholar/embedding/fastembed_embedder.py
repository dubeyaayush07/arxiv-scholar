"""FastEmbed Embedding Backend.

This module implements a lightweight CPU-only embedding backend using the
FastEmbed library (by Qdrant). It uses ONNX Runtime instead of PyTorch,
resulting in a much smaller dependency footprint.

Recommended for:
- Lightweight API servers that only embed user queries (not bulk ingestion)
- Docker/serverless deployments where image size matters
- Environments without GPU access

NOT recommended for:
- Bulk embedding of millions of chunks (too slow on CPU)
- Apple Silicon GPU acceleration (ONNX Runtime doesn't support MPS)
"""

import logging
from typing import List

from arxiv_scholar.embedding.base import BaseEmbedder

logger = logging.getLogger(__name__)


class FastEmbedEmbedder(BaseEmbedder):
    """Embedding backend using FastEmbed (ONNX Runtime, CPU-only).

    This embedder is designed for lightweight deployments where PyTorch
    is too heavy. It produces the same vectors as the SentenceTransformer
    backend for the same model, but runs exclusively on CPU.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 256,
    ) -> None:
        """Initializes the FastEmbed embedder.

        Args:
            model_name: The embedding model identifier (default: BAAI/bge-m3).
            batch_size: Number of texts to encode per batch. FastEmbed's default
                        is 256, which works well for CPU throughput.
        """
        self._model_name = model_name
        self._batch_size = batch_size

        try:
            from fastembed import TextEmbedding

            logger.info(f"Loading FastEmbed model '{model_name}' (CPU/ONNX)...")
            self._model = TextEmbedding(model_name)
            # FastEmbed doesn't expose dimension directly; we infer it
            # by embedding a single dummy string.
            dummy = list(self._model.embed(["dimension probe"]))
            self._dimension = len(dummy[0])
            logger.info(
                f"Model loaded. Dimension: {self._dimension}, Device: CPU (ONNX)"
            )
        except ImportError:
            raise ImportError(
                "fastembed is not installed. "
                "Please run: pip install fastembed"
            )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Encodes a list of texts into dense vectors using ONNX Runtime.

        Args:
            texts: A list of text strings to embed.

        Returns:
            A list of embedding vectors (each a list of floats).
        """
        if not texts:
            return []

        embeddings = list(self._model.embed(texts, batch_size=self._batch_size))
        return [embedding.tolist() for embedding in embeddings]

    @property
    def dimension(self) -> int:
        """Returns the dimensionality of the embedding vectors."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Returns the model identifier."""
        return self._model_name
