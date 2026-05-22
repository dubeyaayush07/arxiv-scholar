"""Abstract Base Class for Embedders.

This module defines the BaseEmbedder interface, which must be implemented
by all embedding backends (e.g., SentenceTransformers, FastEmbed, OpenAI API).
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """Abstract Base Class defining the contract for all embedding backends.

    All implementations must accept a list of text strings and return
    a list of dense vector embeddings (one per input string).
    """

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of text strings into dense vectors.

        Args:
            texts: A list of text strings to embed.

        Returns:
            A list of embedding vectors, where each vector is a list of floats.
            The length of the outer list matches the length of the input texts.
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the dimensionality of the embedding vectors produced by this model."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the name/identifier of the underlying embedding model."""
        pass
