"""Unit tests for document service."""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from src.services.document_service import DocumentService
from src.core.exceptions import DocumentLoadError


class TestDocumentService:
    """Test document service functionality."""

    def test_initialization(self):
        """Test service initialization."""
        service = DocumentService(chunk_size=500, chunk_overlap=50)
        assert service.chunk_size == 500
        assert service.chunk_overlap == 50
        assert service.text_splitter is not None

    def test_initialization_defaults(self):
        """Test default initialization."""
        service = DocumentService()
        assert service.chunk_size > 0
        assert service.chunk_overlap >= 0
        assert service.pdf_loader == "pypdf"

    def test_load_document_success(self, document_service, sample_document):
        """Test successful document loading."""
        chunks = document_service.load_document(sample_document)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_load_document_not_found(self, document_service):
        """Test loading non-existent document."""
        with pytest.raises(DocumentLoadError, match="File not found"):
            document_service.load_document(Path("nonexistent.txt"))

    def test_load_document_invalid_encoding(self, document_service, tmp_path):
        """Test loading document with invalid encoding."""
        doc_path = tmp_path / "invalid.txt"
        doc_path.write_bytes(b'\x80\x81\x82')
        
        with pytest.raises(DocumentLoadError):
            document_service.load_document(doc_path)

    @patch('urllib.request.urlretrieve')
    def test_download_document_success(self, mock_retrieve, document_service, tmp_path):
        """Test successful document download."""
        mock_retrieve.return_value = None
        
        filepath = document_service.download_document(
            "https://example.com/doc.txt",
            "doc.txt"
        )
        
        assert isinstance(filepath, Path)
        mock_retrieve.assert_called_once()

    @patch('urllib.request.urlretrieve')
    def test_download_document_failure(self, mock_retrieve, document_service):
        """Test failed document download."""
        mock_retrieve.side_effect = Exception("Network error")
        
        with pytest.raises(DocumentLoadError, match="Failed to download"):
            document_service.download_document(
                "https://example.com/doc.txt",
                "doc.txt"
            )

    def test_chunking_parameters(self, sample_document):
        """Test different chunking parameters."""
        service = DocumentService(chunk_size=100, chunk_overlap=20)
        chunks = service.load_document(sample_document)
        
        # Verify chunks are created
        assert len(chunks) > 0
        
        # Verify chunk size constraint
        for chunk in chunks:
            assert len(chunk.page_content) <= 150  # Some tolerance

    def test_is_supported_txt(self):
        """Test file type checking for .txt."""
        assert DocumentService.is_supported("test.txt") is True
        assert DocumentService.is_supported("test.TXT") is True

    def test_is_supported_pdf(self):
        """Test file type checking for .pdf."""
        assert DocumentService.is_supported("test.pdf") is True
        assert DocumentService.is_supported("test.PDF") is True

    def test_is_not_supported(self):
        """Test unsupported file types."""
        assert DocumentService.is_supported("test.docx") is False
        assert DocumentService.is_supported("test.jpg") is False
        assert DocumentService.is_supported("test") is False

    def test_unsupported_file_type(self, document_service, tmp_path):
        """Test loading unsupported file type."""
        doc_path = tmp_path / "test.docx"
        doc_path.write_text("test")
        
        with pytest.raises(DocumentLoadError, match="Unsupported file type"):
            document_service.load_document(doc_path)

    @patch('src.services.document_service.PyPDFLoader')
    def test_get_pdf_loader_pypdf(self, mock_loader, document_service, tmp_path):
        """Test PyPDF loader selection."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b'%PDF-1.4')
        
        document_service.pdf_loader = "pypdf"
        loader = document_service._get_pdf_loader(pdf_path)
        
        mock_loader.assert_called_once_with(str(pdf_path))

    @patch('src.services.document_service.UnstructuredPDFLoader')
    def test_get_pdf_loader_unstructured(self, mock_loader, document_service, tmp_path):
        """Test Unstructured PDF loader selection."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b'%PDF-1.4')
        
        document_service.pdf_loader = "unstructured"
        loader = document_service._get_pdf_loader(pdf_path)
        
        mock_loader.assert_called_once_with(str(pdf_path))

    def test_get_pdf_loader_unknown_fallback(self, document_service, tmp_path):
        """Test fallback for unknown PDF loader."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b'%PDF-1.4')
        
        document_service.pdf_loader = "unknown_loader"
        
        # Should fall back to pypdf without raising
        with patch('src.services.document_service.PyPDFLoader') as mock_loader:
            loader = document_service._get_pdf_loader(pdf_path)
            mock_loader.assert_called_once()

    @patch('urllib.request.urlretrieve')
    def test_load_from_url(self, mock_retrieve, document_service, tmp_path, sample_document):
        """Test loading document from URL."""
        # Mock download to use sample document
        def mock_download(url, filepath):
            # Copy sample document content
            Path(filepath).write_text(sample_document.read_text())
        
        mock_retrieve.side_effect = mock_download
        
        chunks = document_service.load_from_url(
            "https://example.com/doc.txt",
            "downloaded.txt"
        )
        
        assert len(chunks) > 0
        mock_retrieve.assert_called_once()
