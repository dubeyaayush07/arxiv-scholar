# arxiv-scholar

**Retrieval system over the arXiv corpus (~2.4M papers, ~5M chunks)**

## Project Objective
The goal is to build a high-performance, robust Retrieval-Augmented Generation (RAG) system over the entire arXiv dataset (~1 TB of PDFs). The system is built from scratch without high-level abstraction frameworks (like LangChain) to expose failure modes and guarantee deep architectural control.

### Hard Requirements
- **Latency**: p99 end-to-end latency < 250 ms for top-20 retrieval.
- **Recall**: Recall@20 ≥ 0.85 on a held-out, hard-negative evaluation set.
- **Robustness**: Survives queries ranging from naive ("attention is all you need") to compositional ("papers between 2019–2021 that critique the attention mechanism on long-context tasks").
- **Adversarial Resilience**: Survives adversarial query sets (paraphrases, negations, multi-hop).

### Tech Stack
- **Embedding**: `BAAI/bge-small-en-v1.5` (Dense) + `BM25/SPLADE` (Sparse) via `fastembed` (ONNX CPU).
- **Vector Store**: `Qdrant` with Reciprocal Rank Fusion (RRF).
- **Reranker**: `jinaai/jina-reranker-v1-tiny-en` (ONNX C++ runtime).
- **Orchestration**: Custom retrieval graph rolled in pure Python (No LangChain).
- **Evaluation**: Custom programmatic metrics (`eval_dataset.jsonl`).

### Success Metrics (Programmatic)
- **Recall**: `Recall@20` on hard-negative set ≈ 0.80
- **Latency**: p99 end-to-end latency ≈ 1.9s for a heavily optimized 2-stage pipeline on commodity Mac CPU.
- **Cost**: Cost per 1k queries < $0.05.
