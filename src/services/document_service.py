"""Document loading and processing service with PDF support."""
import urllib.request
from pathlib import Path
from typing import Optional, List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)

# Opzionali - decommentare se installati
# from langchain_community.document_loaders import PDFPlumberLoader
# from langchain_community.document_loaders import PyMuPDFLoader

from ..core.config import settings
from ..core.exceptions import DocumentLoadError
from ..core.logging_config import get_logger

logger = get_logger(__name__)


class DocumentService:
    """Handles document loading and chunking with multi-format support."""

    # Supported file extensions
    SUPPORTED_EXTENSIONS = {'.txt', '.pdf'}

    def __init__(
        self, 
        chunk_size: Optional[int] = None, 
        chunk_overlap: Optional[int] = None,
        pdf_loader: str = "pypdf"  # pypdf, pymupdf, pdfplumber, unstructured
    ):
        """Initialize document service."""
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.pdf_loader = pdf_loader
        
        # Use RecursiveCharacterTextSplitter for better PDF handling
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )

    def _get_pdf_loader(self, filepath: Path):
        """Get appropriate PDF loader based on configuration."""
        filepath_str = str(filepath)
        
        loaders = {
            "pypdf": lambda: PyPDFLoader(filepath_str),
            # "pymupdf": lambda: PyMuPDFLoader(filepath_str),
            # "pdfplumber": lambda: PDFPlumberLoader(filepath_str),
            "unstructured": lambda: UnstructuredPDFLoader(filepath_str),
        }
        
        loader_fn = loaders.get(self.pdf_loader)
        if not loader_fn:
            logger.warning(
                "unknown_pdf_loader",
                loader=self.pdf_loader,
                fallback="pypdf"
            )
            return PyPDFLoader(filepath_str)
        
        try:
            return loader_fn()
        except ImportError as e:
            logger.error(
                "pdf_loader_import_error",
                loader=self.pdf_loader,
                error=str(e)
            )
            raise DocumentLoadError(
                f"PDF loader '{self.pdf_loader}' not available. "
                f"Install required package: {e}"
            ) from e

    def download_document(self, url: str, filename: str) -> Path:
        """Download document from URL."""
        try:
            filepath = settings.data_dir / filename
            logger.info("downloading_document", url=url, filename=filename)
            urllib.request.urlretrieve(url, filepath)
            logger.info("document_downloaded", filepath=str(filepath))
            return filepath
        except Exception as e:
            logger.error("download_failed", url=url, error=str(e))
            raise DocumentLoadError(f"Failed to download document: {e}") from e

    def load_document(self, filepath: Path) -> List:
        """Load and split document into chunks based on file type."""
        try:
            logger.info("loading_document", filepath=str(filepath))
            
            if not filepath.exists():
                raise DocumentLoadError(f"File not found: {filepath}")
            
            # Determine file type
            extension = filepath.suffix.lower()
            
            if extension not in self.SUPPORTED_EXTENSIONS:
                raise DocumentLoadError(
                    f"Unsupported file type: {extension}. "
                    f"Supported: {', '.join(self.SUPPORTED_EXTENSIONS)}"
                )
            
            # Load based on file type
            if extension == '.pdf':
                loader = self._get_pdf_loader(filepath)
                logger.info("using_pdf_loader", loader=self.pdf_loader)
            else:  # .txt
                loader = TextLoader(str(filepath), encoding='utf-8')
            
            documents = loader.load()
            
            # Split into chunks
            texts = self.text_splitter.split_documents(documents)
            
            logger.info(
                "document_loaded",
                num_pages=len(documents),
                num_chunks=len(texts),
                file_type=extension
            )
            
            return texts
            
        except Exception as e:
            logger.error("document_load_failed", filepath=str(filepath), error=str(e))
            raise DocumentLoadError(f"Failed to load document: {e}") from e

    def load_from_url(self, url: str, filename: str) -> List:
        """Download and load document from URL."""
        filepath = self.download_document(url, filename)
        return self.load_document(filepath)

    @staticmethod
    def is_supported(filename: str) -> bool:
        """Check if file type is supported."""
        return Path(filename).suffix.lower() in DocumentService.SUPPORTED_EXTENSIONS
