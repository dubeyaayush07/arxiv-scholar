import os
import argparse
import logging

from configs import config
from arxiv_scholar.download.arxiv_ingestion import ArxivUnifiedEngine
from arxiv_scholar.ingestion.local import LocalDirectoryReader
from arxiv_scholar.chunking.layout import LayoutAwareChunker
from arxiv_scholar.embedding.st_embedder import SentenceTransformerEmbedder
from arxiv_scholar.embedding.fastembed_embedder import FastEmbedEmbedder, SparseBM25Embedder
from arxiv_scholar.storage.qdrant_store import QdrantVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class PipelineOrchestrator:
    """
    Orchestrates the massive 1TB arXiv dataset ingestion.
    Runs an infinite loop fetching batches, extracting text, chunking, and embedding.
    """
    def __init__(self, download_dir: str, state_file: str):
        # Override config paths for sandboxing/trials
        os.environ["DOWNLOAD_DIR"] = download_dir
        os.environ["STATE_FILE"] = state_file
        config.DOWNLOAD_DIR = download_dir
        config.STATE_FILE = state_file
        
        self.engine = ArxivUnifiedEngine()
        self.chunker = LayoutAwareChunker(max_chunk_size=2000)
        
        if config.EMBEDDING_BACKEND == "fastembed":
            self.embedder = FastEmbedEmbedder(
                model_name=config.EMBEDDING_MODEL,
                batch_size=config.EMBEDDING_BATCH_SIZE,
            )
        else:
            self.embedder = SentenceTransformerEmbedder(
                model_name=config.EMBEDDING_MODEL,
                device=config.EMBEDDING_DEVICE,
                batch_size=config.EMBEDDING_BATCH_SIZE,
            )

        self.sparse_embedder = SparseBM25Embedder(batch_size=config.EMBEDDING_BATCH_SIZE)

        # Storage
        if download_dir == "trial_batch":
            self.store = QdrantVectorStore(
                collection_name=config.QDRANT_COLLECTION,
                location=":memory:",
            )
        else:
            self.store = QdrantVectorStore(
                collection_name=config.QDRANT_COLLECTION,
                host=config.QDRANT_HOST,
                port=config.QDRANT_PORT,
            )
        self.store.ensure_collection(dimension=self.embedder.dimension)

    def process_batch(self, batch_size: int = 50) -> bool:
        """Processes a single batch. Returns True if files were processed, False if caught up."""
        logger.info(f"Fetching batch of size {batch_size}...")
        paths = self.engine.get_batch(batch_size=batch_size)
        
        if not paths:
            logger.info("No files found. The archive is fully caught up.")
            return False
            
        logger.info(f"Downloaded {len(paths)} PDFs.")
        
        # Instantiate reader to scan the download directory
        reader = LocalDirectoryReader(directory_path=config.DOWNLOAD_DIR)
        
        total_chunks = 0
        total_embedded = 0
        for document in reader.read():
            logger.info(f"Parsing and chunking document: {document.metadata.get('filename')}")
            # Chunking phase
            chunks = list(self.chunker.chunk(document))
            total_chunks += len(chunks)
            
            # Embedding phase
            if chunks:
                texts = [chunk.content for chunk in chunks]
                vectors = self.embedder.embed(texts)
                sparse_vectors = self.sparse_embedder.embed(texts)
                total_embedded += len(vectors)
                logger.info(
                    f"  Embedded {len(vectors)} chunks "
                    f"(dim={self.embedder.dimension})"
                )

                # Storage phase — upsert into Qdrant
                upserted = self.store.upsert(chunks, vectors, sparse_vectors=sparse_vectors)
                logger.info(f"  Stored {upserted} points in Qdrant.")

        logger.info(
            f"Successfully processed {len(paths)} documents "
            f"into {total_chunks} chunks, {total_embedded} embeddings."
        )
        
        # Cleanup phase prevents disk space exhaustion
        logger.info("Cleaning up batch from disk...")
        self.engine.cleanup_batch(paths)
        return True

    def run(self, max_batches: int = None, batch_size: int = 50):
        """Runs the pipeline continuously or up to a specific number of batches."""
        logger.info("🚀 Starting Arxiv-Scholar Pipeline")
        batches_processed = 0
        
        while True:
            if max_batches and batches_processed >= max_batches:
                logger.info(f"Reached max batches ({max_batches}). Stopping.")
                break
                
            has_more = self.process_batch(batch_size=batch_size)
            if not has_more:
                break
                
            batches_processed += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the arxiv-scholar ingestion pipeline.")
    parser.add_argument("--trial", action="store_true", help="Run a small trial batch and exit.")
    args = parser.parse_args()
    
    if args.trial:
        logger.info("--- TRIAL MODE ENABLED ---")
        orchestrator = PipelineOrchestrator(download_dir="trial_batch", state_file="trial_state.json")
        orchestrator.run(max_batches=1, batch_size=2)
    else:
        orchestrator = PipelineOrchestrator(download_dir=config.DOWNLOAD_DIR, state_file=config.STATE_FILE)
        # Infinite loop for production backfill
        orchestrator.run(batch_size=50)
