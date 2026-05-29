# Arxiv-Scholar RAG Pipeline - AI Context

Welcome, future AI agent! This file contains the critical context, business logic, architectural decisions, and open questions for the `arxiv-scholar` project. 
This project follows the principles of building robust RAG applications as outlined here: [90-day-applied-ai-bootcamp](https://gist.github.com/TRINETRA-DEVKATTE/d6950ebe23873350f760d16415f09da3).

## Business Logic & Project Goals
- **Objective:** Build a high-performance RAG pipeline over the entire arXiv dataset (~1 TB of PDFs, ~2.4M papers).
- **Core Constraints:**
  - **Latency:** p99 end-to-end latency MUST be `< 250 ms`.
  - **Accuracy:** `Recall@20` MUST be `≥ 0.85` on hard-negative evaluation sets.
- **Philosophy:** No heavy abstractions. No LangChain. Everything is written natively in Python for explicit control over failure modes and optimization.

## Architectural Decisions Made So Far
1. **Base Embedding:** We are currently using `BAAI/bge-small-en-v1.5` via the `fastembed` library (ONNX CPU). We selected this because it is extremely fast and lightweight.
2. **Vector Store:** `Qdrant`. We use its built-in Reciprocal Rank Fusion (RRF) to merge Dense (`bge-small`) and Sparse (`BM25`) search queries natively.
3. **Reranker Pipeline:** We implemented a 2-stage retrieval pipeline. 
   - *Decision:* We swapped out the heavy PyTorch `bge-reranker-v2-m3` for the ultra-lightweight `jinaai/jina-reranker-v1-tiny-en` ONNX model.
   - *Decision:* We use a `fetch_limit` of 100 from Qdrant, and truncate the text to 500 characters before passing it to the reranker.
   - *Result:* We achieved `0.80 Recall` and `1.9s p99 latency` on a local Mac CPU.

## Current Bottlenecks (Important for Next Steps)
- **The Latency/Accuracy Tradeoff:** We discovered mathematically that we cannot hit `Recall@20 >= 0.85` AND `p99 latency < 250ms` on commodity Mac CPUs using a 2-stage reranker pipeline. 
- *Why?* To hit 0.85 Recall, we must pull 100+ documents from Qdrant (because `bge-small` misses the target document in the top 40). Processing 100 documents through an ONNX Cross-Encoder takes ~1.5 to 2.0 seconds, blowing past the 250ms budget.

## Open Questions & Pending Architecture Shifts
1. **ColBERT / Late Interaction:** Should we drop the 2-stage reranker pipeline entirely and migrate the database to use ColBERT (Late Interaction)? ColBERT natively scores multi-vector representations inside Qdrant via `MaxSim` in milliseconds, offering cross-encoder accuracy at dense-search speeds. (This is the highly recommended next step).
2. **GPU Deployment:** If deployed to AWS with Nvidia GPUs (e.g., A10g/H100), the current 2-stage pipeline (fetching 100 chunks) will easily execute in `< 50ms`. Are we building for local Mac execution, or cloud GPU execution?
