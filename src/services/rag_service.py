"""RAG service with streaming support."""
import re
import time
from typing import Generator, Optional

from langchain.chains import RetrievalQA
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma

from ..core.config import settings
from ..core.exceptions import EmbeddingError, LLMConnectionError, QueryError
from ..core.logging_config import get_logger
from ..domain.models import QueryRequest, QueryResponse

logger = get_logger(__name__)


class RAGService:
    """Production-grade RAG service with streaming."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        language: Optional[str] = None,
    ):
        """Initialize RAG service."""
        self.model_name = model_name or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url
        self.temperature = temperature or settings.ollama_temperature
        self.language = language or settings.default_language
        
        self.llm: Optional[Ollama] = None
        self.embeddings: Optional[HuggingFaceEmbeddings | OllamaEmbeddings] = None
        self.vectorstore: Optional[Chroma] = None
        self.qa_chain: Optional[RetrievalQA | ConversationalRetrievalChain] = None
        self.memory: Optional[ConversationBufferMemory] = None
        
        logger.info("rag_service_initialized", model=self.model_name)

    def initialize_llm(self) -> None:
        """Initialize Ollama LLM."""
        try:
            logger.info("initializing_llm", base_url=self.base_url, model=self.model_name)
            
            self.llm = Ollama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                num_ctx=settings.ollama_num_ctx,
                top_p=settings.ollama_top_p,
                top_k=settings.ollama_top_k,
                repeat_penalty=settings.ollama_repeat_penalty,
            )
            
            # Connection test
            _ = self.llm.invoke("hi")
            logger.info("llm_ready", model=self.model_name)
            
        except Exception as e:
            logger.error("llm_initialization_failed", error=str(e))
            raise LLMConnectionError(f"Failed to initialize LLM: {e}") from e

    def initialize_embeddings(self) -> None:
        """Initialize embedding model."""
        try:
            if settings.embedding_provider == "ollama":
                logger.info(
                    "initializing_ollama_embeddings", 
                    model=settings.ollama_embedding_model,
                    base_url=self.base_url
                )
                
                self.embeddings = OllamaEmbeddings(
                    model=settings.ollama_embedding_model,
                    base_url=self.base_url,
                )
                
                logger.info("ollama_embeddings_ready")
                
            else:  # huggingface
                logger.info("initializing_embeddings", model=settings.embedding_model)
                
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=settings.embedding_model,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                
                logger.info("embeddings_ready")
            
        except Exception as e:
            logger.error("embeddings_initialization_failed", error=str(e))
            raise EmbeddingError(f"Failed to initialize embeddings: {e}") from e

    def create_vectorstore(self, documents: list) -> None:
        """Create vector store from documents."""
        try:
            if not self.embeddings:
                self.initialize_embeddings()
            
            logger.info("creating_vectorstore", num_documents=len(documents))
            
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            logger.info("vectorstore_created")
            
        except Exception as e:
            logger.error("vectorstore_creation_failed", error=str(e))
            raise EmbeddingError(f"Failed to create vectorstore: {e}") from e

    def _get_language_instruction(self) -> str:
        """Get language-specific instruction."""
        instructions = {
            'en': "Answer concisely in English. Do not show reasoning.",
            'it': "Rispondi in modo conciso in italiano. Non mostrare il ragionamento.",
            'zh': "用中文简洁回答。不要显示推理过程。"
        }
        return instructions.get(self.language, instructions['en'])

    def _create_qa_prompt(self) -> PromptTemplate:
        """Create optimized QA prompt."""
        lang_inst = self._get_language_instruction()
        template = f"""Context:
{{context}}

Question: {{question}}

{lang_inst}

Answer:"""
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _clean_response(self, text: str) -> str:
        """Clean response from thinking tags and extra whitespace."""
        # Remove <think>...</think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        # Remove extra whitespace
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'^\s+|\s+$', '', text)
        return text.strip()

    def create_qa_chain(self, use_memory: bool = False) -> None:
        """Create Q&A chain with optional memory."""
        if not self.vectorstore:
            raise QueryError("No vectorstore available. Load documents first.")
        
        if not self.llm:
            self.initialize_llm()
        
        try:
            if use_memory:
                self.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
                
                self.qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.vectorstore.as_retriever(
                        search_kwargs={"k": settings.retrieval_k}
                    ),
                    memory=self.memory,
                    return_source_documents=False,
                )
                logger.info("conversational_chain_created")
            else:
                prompt = self._create_qa_prompt()
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vectorstore.as_retriever(
                        search_kwargs={"k": settings.retrieval_k}
                    ),
                    return_source_documents=False,
                    chain_type_kwargs={"prompt": prompt},
                )
                logger.info("qa_chain_created")
                
        except Exception as e:
            logger.error("chain_creation_failed", error=str(e))
            raise QueryError(f"Failed to create Q&A chain: {e}") from e

    def query(self, request: QueryRequest) -> QueryResponse:
        """Execute query and return response."""
        if not self.qa_chain:
            self.create_qa_chain(use_memory=request.use_memory)
        
        try:
            start_time = time.time()
            logger.info("executing_query", question=request.question)
            
            if request.use_memory:
                result = self.qa_chain.invoke({"question": request.question})
                answer = result.get("answer", str(result))
            else:
                result = self.qa_chain.invoke({"query": request.question})
                answer = result.get("result", str(result))
            
            answer = self._clean_response(answer)
            processing_time = time.time() - start_time
            
            logger.info("query_completed", processing_time=processing_time)
            
            return QueryResponse(
                answer=answer,
                question=request.question,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error("query_execution_failed", error=str(e))
            raise QueryError(f"Failed to execute query: {e}") from e

    def query_stream(self, request: QueryRequest) -> Generator[str, None, None]:
        """Execute query with streaming response."""
        if not self.qa_chain:
            self.create_qa_chain(use_memory=request.use_memory)
        
        try:
            logger.info("executing_stream_query", question=request.question)
            
            # For streaming, we'll simulate word-by-word output
            # In production, you'd use LangChain's streaming callbacks
            response = self.query(request)
            
            words = response.answer.split()
            for word in words:
                yield word + " "
                time.sleep(0.05)  # Simulate streaming delay
                
        except Exception as e:
            logger.error("stream_query_failed", error=str(e))
            raise QueryError(f"Failed to execute streaming query: {e}") from e

    def clear_memory(self) -> None:
        """Clear conversation memory."""
        if self.memory:
            self.memory.clear()
            logger.info("memory_cleared")
