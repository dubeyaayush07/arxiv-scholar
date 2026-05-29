import os
from pathlib import Path

# Base directory of the repository (arxiv-scholar)
BASE_DIR = Path(__file__).resolve().parent.parent

# Centralized configurations for the download/ingestion module
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", str(BASE_DIR / "arxiv_batch"))
STATE_FILE = os.getenv("STATE_FILE", "ingestion_state.json")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "arxiv-dataset")
GCS_BASE_PREFIX = os.getenv("GCS_BASE_PREFIX", "arxiv/arxiv/pdf/")

# Embedding configuration
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")

# Qdrant storage configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "arxiv_papers")
