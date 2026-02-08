"""Configuration management with environment variables."""
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Application
    app_name: str = Field(default="RAG Document Q&A", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="production", description="Environment: development, staging, production")
    debug: bool = Field(default=False, description="Debug mode")

    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama server URL")
    ollama_model: str = Field(default="qwen3:8b", description="Ollama model name")
    ollama_temperature: float = Field(default=0.2, ge=0.0, le=1.0, description="LLM temperature")
    ollama_num_ctx: int = Field(default=2048, description="Context window size")
    ollama_top_p: float = Field(default=0.9, description="Top-p sampling")
    ollama_top_k: int = Field(default=40, description="Top-k sampling")
    ollama_repeat_penalty: float = Field(default=1.1, description="Repeat penalty")

    # RAG Configuration
    embedding_provider: str = Field(
        default="huggingface", 
        description="Embedding provider: huggingface or ollama"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model"
    )
    ollama_embedding_model: str = Field(
        default="nomic-embed-text", 
        description="Ollama embedding model (when provider=ollama)"
    )
    chunk_size: int = Field(default=800, ge=100, le=2000, description="Text chunk size")
    chunk_overlap: int = Field(default=100, ge=0, le=500, description="Chunk overlap")
    retrieval_k: int = Field(default=3, ge=1, le=10, description="Number of documents to retrieve")

    pdf_loader: str = Field(
    default="pypdf",
    description="PDF loader: pypdf, pymupdf, pdfplumber, unstructured"
    )

    
    # Language
    default_language: str = Field(default="en", description="Default language: en, it, zh")

    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "data")
    logs_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent.parent / "logs")
    
    # Gradio UI
    gradio_server_name: str = Field(default="0.0.0.0", description="Gradio server host")
    gradio_server_port: int = Field(default=7860, ge=1024, le=65535, description="Gradio server port")
    gradio_share: bool = Field(default=False, description="Create public share link")
    gradio_auth: Optional[tuple[str, str]] = Field(default=None, description="Basic auth (username, password)")

    def model_post_init(self, __context: object) -> None:
        """Create directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
