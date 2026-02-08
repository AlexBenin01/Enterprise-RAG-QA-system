"""Custom exceptions for the application."""


class RAGException(Exception):
    """Base exception for RAG system."""
    pass


class DocumentLoadError(RAGException):
    """Raised when document loading fails."""
    pass


class EmbeddingError(RAGException):
    """Raised when embedding generation fails."""
    pass


class LLMConnectionError(RAGException):
    """Raised when LLM connection fails."""
    pass


class QueryError(RAGException):
    """Raised when query execution fails."""
    pass


class ConfigurationError(RAGException):
    """Raised when configuration is invalid."""
    pass
