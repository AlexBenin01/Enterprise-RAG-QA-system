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

class TestRAGServiceEdgeCases:
    """Test edge cases and error paths."""

    def test_create_prompt_different_languages(self, rag_service):
        """Test prompt creation for different languages."""
        rag_service.language = "en"
        prompt_en = rag_service._create_qa_prompt()
        assert "English" in prompt_en.template
        
        rag_service.language = "it"
        prompt_it = rag_service._create_qa_prompt()
        assert "italiano" in prompt_it.template
        
        rag_service.language = "zh"
        prompt_zh = rag_service._create_qa_prompt()
        assert "中文" in prompt_zh.template

    def test_language_instruction_fallback(self, rag_service):
        """Test language instruction with unknown language."""
        rag_service.language = "unknown_lang"
        instruction = rag_service._get_language_instruction()
        # Should fallback to English
        assert "English" in instruction

    @patch('src.services.rag_service.Ollama')
    def test_initialize_llm_with_custom_params(self, mock_ollama):
        """Test LLM initialization with custom parameters."""
        mock_llm = RunnableLambda(lambda x: "hi")
        mock_ollama.return_value = mock_llm
        
        service = RAGService(
            model_name="custom-model",
            base_url="http://custom:11434",
            temperature=0.8
        )
        service.initialize_llm()
        
        assert service.llm is not None
        assert service.model_name == "custom-model"
        assert service.temperature == 0.8

    @patch('src.services.rag_service.HuggingFaceEmbeddings')
    @patch('src.services.rag_service.Chroma')
    def test_create_vectorstore_initializes_embeddings(self, mock_chroma, mock_embeddings, rag_service):
        """Test that vectorstore creation initializes embeddings if needed."""
        mock_emb = Mock()
        mock_embeddings.return_value = mock_emb
        mock_vs = Mock()
        mock_chroma.from_documents.return_value = mock_vs
        
        # Embeddings not initialized
        assert rag_service.embeddings is None
        
        documents = [Mock()]
        rag_service.create_vectorstore(documents)
        
        # Embeddings should be initialized
        assert rag_service.embeddings is not None

    def test_clean_response_multiple_patterns(self, rag_service):
        """Test response cleaning with various patterns."""
        # Multiple think tags
        text1 = "<think>step1</think>Answer<think>step2</think>"
        assert rag_service._clean_response(text1) == "Answer"
        
        # Mixed case
        text2 = "<THINK>reasoning</THINK>Result"
        assert rag_service._clean_response(text2) == "Result"
        
        # Multiple newlines
        text3 = "Line1\n\n\n\nLine2"
        assert "Line1\nLine2" in rag_service._clean_response(text3)
        
        # Leading/trailing whitespace
        text4 = "   Answer   "
        assert rag_service._clean_response(text4) == "Answer"

    @patch('src.services.rag_service.Chroma')
    @patch('src.services.rag_service.RetrievalQA')
    def test_create_qa_chain_initializes_llm(self, mock_qa, mock_chroma, rag_service, mock_embeddings):
        """Test that QA chain creation initializes LLM if needed."""
        rag_service.embeddings = mock_embeddings
        mock_vs = Mock()
        mock_vs.as_retriever = Mock(return_value=Mock())
        rag_service.vectorstore = mock_vs
        
        mock_chain = Mock()
        mock_qa.from_chain_type.return_value = mock_chain
        
        # LLM not initialized
        assert rag_service.llm is None
        
        # Mock initialize_llm to set the llm
        with patch.object(rag_service, 'initialize_llm') as mock_init:
            # Quando viene chiamato, imposta llm
            def set_llm():
                rag_service.llm = RunnableLambda(lambda x: "test")
            
            mock_init.side_effect = set_llm
            
            rag_service.create_qa_chain(use_memory=False)
            
            # Should have tried to initialize
            mock_init.assert_called_once()


    def test_query_with_existing_chain(self, rag_service):
        """Test query when chain already exists."""
        # Setup mock chain
        mock_chain = Mock()
        mock_chain.invoke = Mock(return_value={"result": "Answer"})
        rag_service.qa_chain = mock_chain
        
        request = QueryRequest(question="Test", use_memory=False)
        response = rag_service.query(request)
        
        assert response.answer == "Answer"
        mock_chain.invoke.assert_called_once()

    def test_query_stream_generates_chunks(self, rag_service):
        """Test that streaming generates multiple chunks."""
        # Setup mock
        mock_chain = Mock()
        mock_chain.invoke = Mock(return_value={"result": "Word1 Word2 Word3"})
        rag_service.qa_chain = mock_chain
        
        request = QueryRequest(question="Test", use_memory=False)
        
        chunks = list(rag_service.query_stream(request))
        
        # Should have multiple chunks (3 words = 3 chunks)
        assert len(chunks) >= 3
        
        # Each chunk should be incremental
        # First chunk has Word1, last chunk has all words
        assert "Word1" in chunks[0]
        assert "Word3" in chunks[-1]



    def test_clear_memory_when_none(self, rag_service):
        """Test clearing memory when it doesn't exist."""
        rag_service.memory = None
        # Should not raise error
        rag_service.clear_memory()

    @patch('src.services.rag_service.HuggingFaceEmbeddings')
    def test_embeddings_error_handling(self, mock_embeddings, rag_service):
        """Test embeddings initialization error path."""
        mock_embeddings.side_effect = ImportError("Module not found")
        
        with pytest.raises(EmbeddingError, match="Failed to initialize embeddings"):
            rag_service.initialize_embeddings()

    @patch('src.services.rag_service.Chroma')
    def test_vectorstore_error_handling(self, mock_chroma, rag_service, mock_embeddings):
        """Test vectorstore creation error path."""
        rag_service.embeddings = mock_embeddings
        mock_chroma.from_documents.side_effect = Exception("Chroma error")
        
        with pytest.raises(EmbeddingError, match="Failed to create vectorstore"):
            rag_service.create_vectorstore([Mock()])

