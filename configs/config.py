import os
from pathlib import Path

# Base directory of the repository (arxiv-scholar)
BASE_DIR = Path(__file__).resolve().parent.parent

# Centralized configurations for the download/ingestion module
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", str(BASE_DIR / "arxiv_batch"))
STATE_FILE = os.getenv("STATE_FILE", "ingestion_state.json")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "arxiv-dataset")
GCS_BASE_PREFIX = os.getenv("GCS_BASE_PREFIX", "arxiv/arxiv/pdf/")
