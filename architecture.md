# Arxiv-Scholar E2E Pipeline Architecture

This document describes the end-to-end ingestion and retrieval flow for the `arxiv-scholar` application.

## Architecture Diagram

```mermaid
flowchart TD
    %% Define components
    A[ArxivUnifiedEngine] -->|Downloads PDFs| B(LocalDirectoryReader)
    B -->|Yields Documents| C(LayoutAwareChunker)
    C -->|Produces Chunks| D(Embedding Service)
    
    %% Split to embedders
    D -->|Dense Embeddings| E[FastEmbedEmbedder]
    D -->|Sparse Embeddings| F[SparseBM25Embedder]
    
    %% Qdrant
    E --> G[(QdrantVectorStore)]
    F --> G
    
    %% Search Flow
    H[User Query] --> I[Embed Query]
    I -->|Query Dense| E
    I -->|Query Sparse| F
    E -.->|Dense Vector| J(Hybrid Search)
    F -.->|Sparse Vector| J
    J -.->|Prefetch & Fusion| G
    G -->|RRF Ranked Results (Top 100)| L(Jina Cross-Encoder Reranker)
    L -->|Final Reranked Results (Top 20)| K[Final Context]
```

## Component Details

### 1. `ArxivUnifiedEngine`
The download engine responsible for fetching PDF papers from arXiv in batches. It ensures the ingestion pipeline runs effectively by pulling chunks of files at a time to avoid out-of-memory errors and rate-limits, orchestrating the massive 1TB dataset fetching.

### 2. `LocalDirectoryReader`
Iterates over the local `DOWNLOAD_DIR`, reading the raw bytes of the downloaded PDF files and converting them into unstructured text schemas encapsulated by the `Document` data model.

### 3. `LayoutAwareChunker`
A specialized chunker that preserves the layout context of a scientific PDF. It segments the unstructured text of the `Document` into `Chunk` objects up to a defined token or size limit.

### 4. `FastEmbedEmbedder` (Dense Embedder)
A lightweight CPU-optimized embedding backend built using FastEmbed and ONNX Runtime. It vectorizes the string `content` of each `Chunk` into high-dimensional semantic vectors. This operates very fast on commodity hardware without relying on heavy PyTorch dependencies.

### 5. `SparseBM25Embedder` (Sparse Embedder)
A sparse representation embedder utilizing the `Qdrant/bm25` model through FastEmbed. It generates indices and values mapping to word frequencies. This supplements the dense representations to capture exact keyword matching.

### 6. `QdrantVectorStore`
The core vector database wrapper built around `qdrant-client`.
- **Upsert**: Ingests each `Chunk` along with its Dense Vector and Sparse Vector.
- **Hybrid Search**: Performs retrieval utilizing a combination of prefetching dense vectors and sparse vectors simultaneously.
- **Reciprocal Rank Fusion (RRF)**: Merges the scores of the exact keyword matching and semantic matching, returning the highest-ranked 100 chunks.

### 7. Jina Cross-Encoder Reranker
A lightweight ONNX CPU-optimized Cross-Encoder (`jina-reranker-v1-tiny-en`).
- Takes the Top 100 results from Qdrant, truncates them to 500 characters, and performs a secondary cross-attention scoring against the user query.
- Re-sorts the results to drastically improve precision and Recall@20 before yielding the final Top 20 context.
