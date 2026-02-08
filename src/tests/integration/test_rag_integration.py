"""Unit tests for RAG service."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain_core.runnables import RunnableLambda

from src.services.rag_service import RAGService
from src.services.document_service import DocumentService
from src.domain.models import QueryRequest, QueryResponse
from src.core.exceptions import LLMConnectionError, QueryError, EmbeddingError


class TestRAGService:
    """Test RAG service functionality."""

    def test_initialization(self):
        """Test service initialization."""
        service = RAGService(
            model_name="test-model",
            temperature=0.5,
            language="it"
        )
        assert service.model_name == "test-model"
        assert service.temperature == 0.5
        assert service.language == "it"

    @patch('src.services.rag_service.Ollama')
    def test_initialize_llm_success(self, mock_ollama, rag_service):
        """Test successful LLM initialization."""
        mock_llm = RunnableLambda(lambda x: "hi")
        mock_ollama.return_value = mock_llm
        
        rag_service.initialize_llm()
        
        assert rag_service.llm is not None
        mock_ollama.assert_called_once()

    @patch('src.services.rag_service.Ollama')
    def test_initialize_llm_failure(self, mock_ollama, rag_service):
        """Test LLM initialization failure."""
        mock_ollama.side_effect = Exception("Connection error")
        
        with pytest.raises(LLMConnectionError):
            rag_service.initialize_llm()

    @patch('src.services.rag_service.HuggingFaceEmbeddings')
    def test_initialize_embeddings_success(self, mock_embeddings, rag_service):
        """Test successful embeddings initialization."""
        mock_emb = Mock()
        mock_embeddings.return_value = mock_emb
        
        rag_service.initialize_embeddings()
        
        assert rag_service.embeddings is not None
        mock_embeddings.assert_called_once()

    @patch('src.services.rag_service.HuggingFaceEmbeddings')
    def test_initialize_embeddings_failure(self, mock_embeddings, rag_service):
        """Test embeddings initialization failure."""
        mock_embeddings.side_effect = Exception("Model load error")
        
        with pytest.raises(EmbeddingError):
            rag_service.initialize_embeddings()

    def test_language_instruction(self, rag_service):
        """Test language-specific instructions."""
        rag_service.language = "en"
        assert "English" in rag_service._get_language_instruction()
        
        rag_service.language = "it"
        assert "italiano" in rag_service._get_language_instruction()

    def test_clean_response(self, rag_service):
        """Test response cleaning."""
        dirty = "<think>reasoning</think>This is the answer.\n\n\n"
        clean = rag_service._clean_response(dirty)
        
        assert "<think>" not in clean
        assert "This is the answer." in clean
        assert clean.strip() == "This is the answer."

    def test_query_without_chain(self, rag_service):
        """Test query without initialized chain."""
        request = QueryRequest(question="Test question")
        
        with pytest.raises(QueryError):
            rag_service.query(request)

    @patch('src.services.rag_service.Chroma')
    def test_create_vectorstore(self, mock_chroma, rag_service, mock_embeddings):
        """Test vectorstore creation."""
        rag_service.embeddings = mock_embeddings
        mock_vectorstore = Mock()
        mock_chroma.from_documents.return_value = mock_vectorstore
        
        documents = [Mock()]
        rag_service.create_vectorstore(documents)
        
        assert rag_service.vectorstore is not None
        mock_chroma.from_documents.assert_called_once()

    def test_create_qa_chain_without_vectorstore(self, rag_service):
        """Test QA chain creation without vectorstore."""
        with pytest.raises(QueryError):
            rag_service.create_qa_chain()

    def test_clear_memory(self, rag_service):
        """Test memory clearing."""
        rag_service.memory = Mock()
        rag_service.clear_memory()
        rag_service.memory.clear.assert_called_once()

    def test_clear_memory_without_memory(self, rag_service):
        """Test memory clearing when no memory exists."""
        rag_service.clear_memory()  # Should not raise error


@pytest.mark.integration
class TestRAGIntegration:
    """Integration tests for complete RAG workflow."""

    @patch('src.services.rag_service.Ollama')
    @patch('src.services.rag_service.HuggingFaceEmbeddings')
    @patch('src.services.rag_service.Chroma')
    @patch('src.services.rag_service.RetrievalQA')
    def test_full_rag_workflow(
        self, mock_retrieval_qa, mock_chroma, mock_embeddings, mock_ollama, sample_document
    ):
        """Test complete RAG workflow from document to query."""
        # Setup mocks with proper Runnable
        mock_llm = RunnableLambda(lambda x: "Test answer")
        mock_ollama.return_value = mock_llm
        
        mock_emb = Mock()
        mock_embeddings.return_value = mock_emb
        
        mock_vs = Mock()
        mock_retriever = Mock()
        mock_vs.as_retriever = Mock(return_value=mock_retriever)
        mock_chroma.from_documents.return_value = mock_vs
        
        # Mock the QA chain
        mock_chain = Mock()
        mock_chain.invoke = Mock(return_value={"result": "Test answer"})
        mock_retrieval_qa.from_chain_type.return_value = mock_chain
        
        # Load document
        doc_service = DocumentService()
        documents = doc_service.load_document(sample_document)
        
        # Initialize RAG
        rag_service = RAGService()
        rag_service.initialize_llm()
        rag_service.initialize_embeddings()
        rag_service.create_vectorstore(documents)
        rag_service.create_qa_chain(use_memory=False)
        
        # Query
        request = QueryRequest(question="Test question")
        response = rag_service.query(request)
        
        assert response.answer == "Test answer"
        assert response.question == "Test question"
        assert response.processing_time >= 0  

    @patch('src.services.rag_service.Ollama')
    @patch('src.services.rag_service.HuggingFaceEmbeddings')
    @patch('src.services.rag_service.Chroma')
    @patch('src.services.rag_service.ConversationalRetrievalChain')
    def test_conversational_workflow(
        self, mock_conv_chain, mock_chroma, mock_embeddings, mock_ollama, sample_document
    ):
        """Test conversational RAG with memory."""
        # Setup mocks
        mock_llm = RunnableLambda(lambda x: "Answer")
        mock_ollama.return_value = mock_llm
        
        mock_emb = Mock()
        mock_embeddings.return_value = mock_emb
        
        mock_vs = Mock()
        mock_retriever = Mock()
        mock_vs.as_retriever = Mock(return_value=mock_retriever)
        mock_chroma.from_documents.return_value = mock_vs
        
        # Mock conversational chain
        mock_chain = Mock()
        mock_chain.invoke = Mock(return_value={"answer": "Response"})
        mock_conv_chain.from_llm.return_value = mock_chain
        
        # Setup RAG with memory
        doc_service = DocumentService()
        documents = doc_service.load_document(sample_document)
        
        rag_service = RAGService()
        rag_service.initialize_llm()
        rag_service.create_vectorstore(documents)
        rag_service.create_qa_chain(use_memory=True)
        
        # Multiple queries
        request1 = QueryRequest(question="First question", use_memory=True)
        response1 = rag_service.query(request1)
        
        request2 = QueryRequest(question="Follow-up", use_memory=True)
        response2 = rag_service.query(request2)
        
        assert mock_chain.invoke.call_count == 2
        assert response1.answer == "Response"
        assert response2.answer == "Response"
        assert response1.processing_time >= 0 
        assert response2.processing_time >= 0  
