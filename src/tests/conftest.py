"""Pytest configuration and fixtures."""
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Force HuggingFace embeddings in tests (override .env)
os.environ['EMBEDDING_PROVIDER'] = 'huggingface'
os.environ['EMBEDDING_MODEL'] = 'sentence-transformers/all-MiniLM-L6-v2'

from src.core.config import settings
from src.services.document_service import DocumentService
from src.services.rag_service import RAGService


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Setup test environment."""
    # Ensure test configuration
    settings.embedding_provider = "huggingface"
    settings.debug = True


@pytest.fixture
def mock_settings(tmp_path):
    """Mock settings with temporary paths."""
    settings.data_dir = tmp_path / "data"
    settings.logs_dir = tmp_path / "logs"
    settings.data_dir.mkdir(exist_ok=True)
    settings.logs_dir.mkdir(exist_ok=True)
    settings.embedding_provider = "huggingface"  # Force for tests
    return settings


@pytest.fixture
def sample_document(tmp_path):
    """Create a sample document for testing."""
    doc_path = tmp_path / "test_document.txt"
    content = """
    This is a test document.
    It contains multiple lines.
    We will use it for testing the RAG system.
    The document has enough content for chunking.
    """
    doc_path.write_text(content)
    return doc_path


@pytest.fixture
def mock_llm():
    """Mock Ollama LLM with Runnable interface."""
    from langchain_core.runnables import RunnableLambda
    
    def mock_invoke(x):
        if isinstance(x, dict):
            return "Test response"
        return "Test response"
    
    # Create a proper Runnable mock
    llm = RunnableLambda(mock_invoke)
    return llm


@pytest.fixture
def mock_embeddings():
    """Mock HuggingFace embeddings."""
    embeddings = Mock()
    embeddings.embed_documents = Mock(return_value=[[0.1, 0.2, 0.3]])
    embeddings.embed_query = Mock(return_value=[0.1, 0.2, 0.3])
    return embeddings


@pytest.fixture
def document_service():
    """Create document service instance."""
    return DocumentService()


@pytest.fixture
def rag_service(monkeypatch):
    """Create RAG service instance with test settings."""
    monkeypatch.setenv('EMBEDDING_PROVIDER', 'huggingface')
    settings.embedding_provider = "huggingface"
    return RAGService()
