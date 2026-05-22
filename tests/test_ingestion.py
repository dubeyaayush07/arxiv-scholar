"""Unit tests for the scholar-rag ingestion layer."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from typing import Any
from pydantic import ValidationError

from arxiv_scholar.ingestion.local import ARXIV_ID_REGEX, LocalDirectoryReader
from arxiv_scholar.schema import Document


def test_document_schema_validation() -> None:
    """Verifies Document validation and immutability."""
    # Valid Document creation
    doc = Document(
        id="test-id",
        content="Test content",
        metadata={"title": "Test Title", "arxiv_id": "2101.00001"},
    )
    assert doc.id == "test-id"
    assert doc.content == "Test content"
    assert doc.metadata["title"] == "Test Title"

    # Frozen check (immutability)
    with pytest.raises(ValidationError):
        # In Pydantic v2, mutating frozen models raises ValidationError
        doc.id = "new-id"  # type: ignore

    # Missing required field
    with pytest.raises(ValidationError):
        Document(content="Missing ID")  # type: ignore


def test_arxiv_id_regex() -> None:
    """Verifies that the arXiv ID regex matches standard formats."""
    # New format
    assert ARXIV_ID_REGEX.search("2101.00001").group(1) == "2101.00001"
    assert ARXIV_ID_REGEX.search("arxiv:2101.00001v2").group(1) == "2101.00001v2"
    assert ARXIV_ID_REGEX.search("2101.00001v3").group(1) == "2101.00001v3"
    
    # Legacy format
    assert ARXIV_ID_REGEX.search("hep-th/9703030").group(1) == "hep-th/9703030"
    assert ARXIV_ID_REGEX.search("math.GT/0307245").group(1) == "math.GT/0307245"

    # Non-matching strings
    assert ARXIV_ID_REGEX.search("not-an-arxiv-id") is None
    assert ARXIV_ID_REGEX.search("123.45") is None


@patch("arxiv_scholar.ingestion.local.pypdf.PdfReader")
@patch("builtins.open", new_callable=mock_open, read_data=b"mock pdf content")
@patch("pathlib.Path.glob")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_file")
@patch("pathlib.Path.stat")
def test_local_directory_reader_success(
    mock_stat: MagicMock,
    mock_is_file: MagicMock,
    mock_exists: MagicMock,
    mock_glob: MagicMock,
    mock_file: MagicMock,
    mock_pdf_reader: MagicMock,
) -> None:
    """Verifies LocalDirectoryReader yields Document objects with correct metadata."""
    # Setup mocks
    mock_exists.return_value = True
    mock_is_file.return_value = True
    mock_stat.return_value.st_size = 12345
    
    mock_pdf_path = Path("/mock_data/2101.00001.pdf")
    mock_glob.return_value = [mock_pdf_path]
    
    # Mock PDF reader behavior
    mock_reader_instance = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "This is the content of the paper arXiv:2101.00001"
    mock_reader_instance.pages = [mock_page]
    mock_reader_instance.metadata = {"/Title": "Mock Paper Title"}
    mock_pdf_reader.return_value = mock_reader_instance

    # Initialize and read
    reader = LocalDirectoryReader(directory_path="/mock_data")
    documents = list(reader.read())

    # Assertions
    assert len(documents) == 1
    doc = documents[0]
    assert isinstance(doc, Document)
    assert doc.content == "This is the content of the paper arXiv:2101.00001"
    assert doc.metadata["arxiv_id"] == "2101.00001"
    assert doc.metadata["title"] == "Mock Paper Title"
    assert doc.metadata["filename"] == "2101.00001.pdf"
    assert doc.metadata["file_size_bytes"] == 12345
    assert doc.id is not None  # SHA-256 hash generated


@patch("arxiv_scholar.ingestion.local.pypdf.PdfReader")
@patch("builtins.open", new_callable=mock_open, read_data=b"corrupted pdf data")
@patch("pathlib.Path.glob")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.is_file")
@patch("pathlib.Path.stat")
def test_local_directory_reader_defensive_parsing(
    mock_stat: MagicMock,
    mock_is_file: MagicMock,
    mock_exists: MagicMock,
    mock_glob: MagicMock,
    mock_file: MagicMock,
    mock_pdf_reader: MagicMock,
) -> None:
    """Verifies that the reader logs errors and continues when parsing fails."""
    mock_exists.return_value = True
    mock_is_file.return_value = True
    mock_stat.return_value.st_size = 54321
    
    file_bad = Path("/mock_data/bad.pdf")
    file_good = Path("/mock_data/2101.00001.pdf")
    mock_glob.return_value = [file_bad, file_good]

    # Mock PdfReader to raise exception for bad.pdf, and succeed for good.pdf
    def mock_pdf_init(path_or_stream: Any) -> MagicMock:
        if str(path_or_stream).endswith("bad.pdf"):
            raise ValueError("Corrupt file structure")
        
        mock_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Good paper text 2101.00001"
        mock_reader.pages = [mock_page]
        mock_reader.metadata = {"/Title": "Good Paper"}
        return mock_reader

    mock_pdf_reader.side_effect = mock_pdf_init

    # Initialize and read
    reader = LocalDirectoryReader(directory_path="/mock_data")
    documents = list(reader.read())

    # Should skip bad.pdf and return good.pdf
    assert len(documents) == 1
    assert documents[0].metadata["filename"] == "2101.00001.pdf"
    assert documents[0].metadata["arxiv_id"] == "2101.00001"
    assert documents[0].metadata["file_size_bytes"] == 54321

