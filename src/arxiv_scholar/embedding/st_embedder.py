"""SentenceTransformer Embedding Backend.

This module implements the primary embedding backend using the sentence-transformers
library with PyTorch. It supports automatic device detection (CUDA, Apple MPS, CPU)
and is the recommended backend for both local development and GPU-accelerated
production workloads.
"""

import logging
import platform
from typing import List

from arxiv_scholar.embedding.base import BaseEmbedder

logger = logging.getLogger(__name__)



def _resolve_device(requested: str = "auto") -> str:
    """Determines the best available compute device for PyTorch inference.

    Uses platform-level hardware detection for Apple Silicon instead of
    torch.backends.mps.is_available(), because the LayoutAwareChunker
    monkey-patches that function to False (to work around a Docling bug).

    Args:
        requested: The desired device string. Use "auto" for automatic detection,
                   or explicitly pass "cuda", "mps", or "cpu".

    Returns:
        A PyTorch device string: "cuda", "mps", or "cpu".
    """
    if requested != "auto":
        return requested

    try:
        import torch

        # CUDA (NVIDIA GPU) takes top priority
        if torch.cuda.is_available():
            return "cuda"

        # For Apple Silicon, detect via platform instead of torch's
        # potentially monkey-patched MPS functions (see layout.py)
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                torch.zeros(1, device="mps")
                return "mps"
            except Exception:
                pass

        return "cpu"

    except ImportError:
        return "cpu"


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedding backend using sentence-transformers (PyTorch).

    This is the recommended embedder for:
    - Local development on Apple Silicon (uses MPS GPU)
    - Cloud batch processing on NVIDIA GPUs (uses CUDA)
    - CPU fallback when no GPU is available

    The model is loaded once at init and reused for all embed() calls.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "auto",
        batch_size: int = 32,
    ) -> None:
        """Initializes the SentenceTransformer embedder.

        Args:
            model_name: HuggingFace model identifier (default: BAAI/bge-m3).
            device: Compute device — "auto", "cuda", "mps", or "cpu".
            batch_size: Number of texts to encode in a single forward pass.
                        Larger batches are faster but use more memory.
        """
        self._model_name = model_name
        self._batch_size = batch_size
        self._device = _resolve_device(device)

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(
                f"Loading embedding model '{model_name}' on device '{self._device}'..."
            )
            self._model = SentenceTransformer(model_name, device=self._device)
            self._dimension = self._model.get_embedding_dimension()
            logger.info(
                f"Model loaded. Dimension: {self._dimension}, Device: {self._device}"
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Please run: pip install sentence-transformers"
            )

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Encodes a list of texts into normalized dense vectors.

        Args:
            texts: A list of text strings to embed.

        Returns:
            A list of embedding vectors (each a list of floats), L2-normalized.
        """
        if not texts:
            return []

        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=len(texts) > self._batch_size,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        """Returns the dimensionality of the embedding vectors (1024 for BGE-M3)."""
        return self._dimension

    @property
    def model_name(self) -> str:
        """Returns the HuggingFace model identifier."""
        return self._model_name

    @property
    def device(self) -> str:
        """Returns the compute device this embedder is running on."""
        return self._device
