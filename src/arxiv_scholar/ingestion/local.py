"""Local Directory Reader Implementation.

This module implements the LocalDirectoryReader, which reads PDF documents
from a local directory using the pypdf library, extracting text and metadata defensively.
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, Generator, Union, Optional
import fitz  # PyMuPDF

from arxiv_scholar.ingestion.base import DocumentReader
from arxiv_scholar.schema import Document

logger = logging.getLogger(__name__)

# Pattern matching modern arXiv IDs (e.g., 2101.00001 or 2101.00001v2)
# and legacy arXiv IDs (e.g., hep-th/9703030)
ARXIV_ID_REGEX = re.compile(
    r"\b(?:arxiv:)?(\d{4}\.\d{4,5}(?:v\d+)?|[a-zA-Z\-]+(?:\.[a-zA-Z\-]+)*/\d{7}(?:v\d+)?)(?![a-zA-Z0-9])",
    re.IGNORECASE,
)


class LocalDirectoryReader(DocumentReader):
    """Document reader that scans a local directory for PDFs and extracts their content.

    Processes files iteratively using Python generators to maintain a constant
    memory footprint.
    """

    def __init__(
        self,
        directory_path: Union[str, Path],
        recursive: bool = False,
        file_glob: str = "*.pdf",
    ) -> None:
        """Initializes the LocalDirectoryReader.

        Args:
            directory_path: Path to the directory containing documents.
            recursive: Whether to scan subdirectories recursively.
            file_glob: Glob pattern for filtering files (defaults to "*.pdf").
        """
        self.directory_path = Path(directory_path)
        self.recursive = recursive
        self.file_glob = file_glob

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Computes the SHA-256 hash of a file's contents for deduplication and identification."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _extract_arxiv_id(self, text: str, filename: str) -> Optional[str]:
        """Attempts to extract an arXiv ID from the filename or text content."""
        # Try matching filename first
        if match := ARXIV_ID_REGEX.search(filename):
            return match.group(1)

        # Try matching first 2000 characters of the text content
        if match := ARXIV_ID_REGEX.search(text[:2000]):
            return match.group(1)

        return None

    def read(self) -> Generator[Document, None, None]:
        """Iterates over files in the directory and yields Document objects.

        Defensively catches parsing errors and logs them, ensuring that a single
        corrupted PDF does not crash the entire ingestion pipeline.
        """
        if not self.directory_path.exists():
            logger.error(f"Directory path does not exist: {self.directory_path}")
            return

        # Select file scanning method based on recursion flag
        file_generator = (
            self.directory_path.rglob(self.file_glob)
            if self.recursive
            else self.directory_path.glob(self.file_glob)
        )

        for file_path in file_generator:
            if not file_path.is_file():
                continue

            try:
                # Calculate hash for unique document ID
                doc_id = self._calculate_file_hash(file_path)

                # Extract text using pypdf
                text_content, pdf_metadata = self._parse_pdf(file_path)

                # Build metadata dict
                metadata: Dict[str, Any] = {
                    "source_path": str(file_path.resolve()),
                    "filename": file_path.name,
                    "file_size_bytes": file_path.stat().st_size,
                }

                # Attempt to extract title
                title = pdf_metadata.get("/Title") or pdf_metadata.get("title")
                if title:
                    # Clean title if it is a pypdf wrapper or string
                    title_str = str(title).strip()
                    if title_str:
                        metadata["title"] = title_str
                
                # Fallback title if none found
                if "title" not in metadata:
                    metadata["title"] = file_path.stem

                # Attempt to extract arXiv ID
                arxiv_id = self._extract_arxiv_id(text_content, file_path.name)
                if arxiv_id:
                    metadata["arxiv_id"] = arxiv_id

                yield Document(
                    id=doc_id,
                    content=text_content,
                    metadata=metadata,
                )

            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}", exc_info=True)
                continue

    def _parse_pdf(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Parses a PDF file and extracts text and metadata.

        Args:
            file_path: Path to the PDF file.

        Returns:
            A tuple of (extracted_text, pdf_metadata_dictionary).
        """
        text_pages = []
        pdf_metadata = {}

        try:
            doc = fitz.open(file_path)
            if doc.metadata:
                pdf_metadata = dict(doc.metadata)

            for page in doc:
                text = page.get_text()
                if text:
                    text_pages.append(text)
            
            doc.close()
        except Exception as e:
            logger.error(f"Error reading PDF with fitz: {e}")

        return "\n".join(text_pages), pdf_metadata
