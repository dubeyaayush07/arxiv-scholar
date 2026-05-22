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
- **Embedding**: `BGE-M3` or `NV-Embed-v2` (late-interaction support), baselined against `text-embedding-3-large`.
- **Vector Store**: `Qdrant` or `LanceDB` (Pick one, benchmarking both is a stretch goal).
- **Sparse Retrieval**: `BM25` via `Tantivy` or `Pyserini`.
- **Reranker**: `BGE-reranker-v2-m3` or `Cohere Rerank 3`.
- **Orchestration**: Custom retrieval graph rolled in pure Python (No LangChain).
- **Evaluation**: `Ragas` + custom programmatic metrics.

### Success Metrics (Programmatic)
- **Recall**: `recall@k` (k=5, 10, 20) on hard-negative set ≥ 0.85 at k=20
- **nDCG**: `nDCG@10` ≥ 0.72
- **Observability**: p50/p95/p99 latency tracked in Prometheus.
- **Cost**: Cost per 1k queries < $0.05.
