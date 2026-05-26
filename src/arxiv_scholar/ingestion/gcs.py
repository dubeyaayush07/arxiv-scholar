"""Google Cloud Storage (GCS) Bucket Reader Template.

This module provides a template for GCS Bucket Reader. It implements the
DocumentReader interface and outlines how to stream PDF data from a cloud bucket.
"""

import io
import logging
from typing import Generator, Optional
import fitz  # PyMuPDF

from arxiv_scholar.ingestion.base import DocumentReader
from arxiv_scholar.schema import Document

logger = logging.getLogger(__name__)

# Note: The google-cloud-storage library is required for cloud run.
# Install via: pip install google-cloud-storage
try:
    from google.cloud import storage  # type: ignore
except ImportError:
    storage = None


class GCSBucketReader(DocumentReader):
    """Template for reading PDF documents directly from a Google Cloud Storage bucket.

    Streams files iteratively from the bucket and parses them in-memory to avoid
    disk storage overhead in serverless environments (e.g., Cloud Run).
    """

    def __init__(
        self,
        bucket_name: str,
        prefix: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> None:
        """Initializes the GCSBucketReader.

        Args:
            bucket_name: Name of the GCS bucket.
            prefix: Optional directory/folder path prefix to filter blobs.
            project_id: Optional GCP project ID for credentials.
        """
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.project_id = project_id
        self._client = None
        self._bucket = None

    def _initialize_client(self) -> None:
        """Initializes GCS storage client dynamically to handle missing dependency gracefully."""
        if storage is None:
            raise ImportError(
                "The 'google-cloud-storage' package is required but not installed. "
                "Please run: pip install google-cloud-storage"
            )
        if not self._client:
            self._client = storage.Client(project=self.project_id)
            self._bucket = self._client.bucket(self.bucket_name)

    def read(self) -> Generator[Document, None, None]:
        """Iterates over blobs in the GCS bucket and yields Document objects.

        Note: Reads blobs dynamically to keep memory usage constant.
        """
        try:
            self._initialize_client()
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            raise e

        # List all blobs in the bucket with optional prefix filtering
        assert self._bucket is not None
        blobs = self._bucket.list_blobs(prefix=self.prefix)

        for blob in blobs:
            # Only process PDF files
            if not blob.name.lower().endswith(".pdf"):
                continue

            try:
                # Stream blob bytes directly into memory
                blob_bytes = blob.download_as_bytes()
                
                # Initialize PyMuPDF (fitz) reader from memory stream
                doc = fitz.open("pdf", blob_bytes)
                text_pages = []
                for page in doc:
                    text = page.get_text()
                    if text:
                        text_pages.append(text)
                
                content = "\n".join(text_pages)
                
                pdf_metadata = doc.metadata or {}
                doc.close()

                # Calculate SHA-256 hash from blob md5 or download bytes
                # GCS blob has a md5_hash attribute (in base64 format)
                import base64
                import binascii
                
                if blob.md5_hash:
                    try:
                        md5_decoded = base64.b64decode(blob.md5_hash)
                        doc_id = binascii.hexlify(md5_decoded).decode("utf-8")
                    except Exception:
                        doc_id = blob.id or blob.name
                else:
                    import hashlib
                    doc_id = hashlib.sha256(blob_bytes).hexdigest()

                # Build metadata dictionary
                metadata = {
                    "source": f"gs://{self.bucket_name}/{blob.name}",
                    "filename": blob.name.split("/")[-1],
                    "content_type": blob.content_type,
                    "updated": str(blob.updated),
                    "size_bytes": blob.size,
                }

                # Add pdf properties to metadata if available
                title = pdf_metadata.get("/Title") or pdf_metadata.get("title")
                if title:
                    metadata["title"] = str(title).strip()

                if "title" not in metadata:
                    metadata["title"] = blob.name.split("/")[-1].replace(".pdf", "")

                # Check for arXiv ID patterns in filename or content
                from arxiv_scholar.ingestion.local import ARXIV_ID_REGEX
                match = ARXIV_ID_REGEX.search(blob.name)
                if not match:
                    match = ARXIV_ID_REGEX.search(content[:2000])

                if match:
                    metadata["arxiv_id"] = match.group(1)

                yield Document(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                )

            except Exception as e:
                logger.error(f"Failed to process blob {blob.name}: {e}", exc_info=True)
                continue
