"""Core schemas and data models for scholar-rag.

This module contains the basic data structures used across the ingestion,
processing, embedding, and retrieval pipelines.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """Represents a raw document ingested from a data source.

    Attributes:
        id: Unique identifier for the document, e.g., SHA-256 hash of content or path.
        content: The raw text content extracted from the document source.
        metadata: Key-value metadata containing source details, arXiv ID, title, etc.
    """

    id: str = Field(
        ...,
        description="Unique identifier of the document, typically a hash of the content or path.",
    )
    content: str = Field(
        ...,
        description="The raw text content extracted from the source document.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata dictionary containing source information, arXiv ID, author, and title.",
    )

    model_config = ConfigDict(
        frozen=True,
        json_schema_extra={
            "example": {
                "id": "sha256_hash_here",
                "content": "This is an example arXiv paper content...",
                "metadata": {
                    "source": "/mock_data/2101.00001.pdf",
                    "arxiv_id": "2101.00001",
                    "title": "A Sample arXiv Paper on RAG",
                },
            }
        }
    )


class Chunk(BaseModel):
    """Represents a discrete segment of a Document after chunking.

    Attributes:
        id: Unique identifier for the chunk, usually a hash of its content.
        document_id: The ID of the parent Document this chunk belongs to.
        content: The text content of the chunk.
        metadata: Chunk-specific metadata (e.g., chunk_index, element_type, bounding_boxes).
    """

    id: str = Field(
        ...,
        description="Unique identifier for the chunk.",
    )
    document_id: str = Field(
        ...,
        description="The ID of the parent Document.",
    )
    content: str = Field(
        ...,
        description="The text content of the chunk.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk-specific metadata including element type, index, etc.",
    )

    model_config = ConfigDict(frozen=True)

