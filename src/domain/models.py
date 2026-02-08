"""Domain models and data structures."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class Document:
    """Represents a loaded document."""
    filename: str
    content: str
    metadata: dict = field(default_factory=dict)
    loaded_at: datetime = field(default_factory=datetime.now)


@dataclass
class QueryRequest:
    """Query request model."""
    question: str
    language: str = "en"
    use_memory: bool = False


@dataclass
class QueryResponse:
    """Query response model."""
    answer: str
    question: str
    sources: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class StreamChunk:
    """Represents a chunk of streamed response."""
    content: str
    is_final: bool = False
    metadata: Optional[dict] = None
