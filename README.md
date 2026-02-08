# 🚀 Enterprise RAG Q&A System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-55%20passed-success)](https://github.com)
[![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen)](https://github.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![WCAG 2 AAA](https://img.shields.io/badge/WCAG-2%20AAA-green.svg)](https://www.w3.org/WAI/WCAG2AAA-Conformance)

> **Production-grade RAG (Retrieval-Augmented Generation) system with WCAG 2 AAA accessible interface, streaming responses, and enterprise architecture.**

Built with LangChain, Ollama, Gradio, and ChromaDB for document-based question answering with conversational memory.

---

## ✨ Key Features

### 🎯 Core Capabilities
- **🤖 RAG Pipeline**: State-of-the-art retrieval-augmented generation
- **💬 Streaming Responses**: Real-time word-by-word answer generation
- **🧠 Conversational Memory**: Context-aware multi-turn conversations
- **📄 Multi-Format Support**: PDF, TXT with extensible architecture
- **🌍 Multilingual**: English, Italian, Chinese support

### ♿ Accessibility First
- **WCAG 2 AAA Compliant**: Highest accessibility standards
- **7:1 Contrast Ratio**: Enhanced visibility for all users
- **Keyboard Navigation**: Full functionality without mouse
- **Screen Reader Optimized**: Proper ARIA labels and semantic HTML
- **Focus Indicators**: Clear visual feedback for navigation

### 🏗️ Enterprise Architecture
- **Clean Architecture**: Separation of concerns (Core/Domain/Services/Presentation)
- **Type Safety**: Full type hints with Pydantic validation
- **Structured Logging**: JSON logs for production monitoring
- **Configuration Management**: Environment-based settings
- **Comprehensive Testing**: Unit and integration tests with pytest
- **Security**: Input validation, error handling, logging

### 🔧 Technology Stack
- **LLM**: Ollama (local inference, privacy-first)
- **Embeddings**: HuggingFace Sentence Transformers or Ollama
- **Vector Store**: ChromaDB (persistent, high-performance)
- **Framework**: LangChain 0.1.x
- **Frontend**: Gradio 4.x with streaming
- **Testing**: pytest with coverage reporting

---

## 📁 Project Structure


```bash 
enterprise-rag-qa-system/
├── src/
│ ├── core/ # Configuration, logging, exceptions
│ │ ├── config.py # Pydantic settings with .env support
│ │ ├── exceptions.py # Custom exception hierarchy
│ │ └── logging_config.py # Structured logging (JSON + console)
│ ├── domain/ # Domain models
│ │ └── models.py # QueryRequest, QueryResponse, Document
│ ├── services/ # Business logic
│ │ ├── rag_service.py # RAG pipeline with streaming
│ │ └── document_service.py # Document loading (PDF, TXT)
│ └── presentation/ # UI layer
│ └── gradio_ui.py # WCAG AAA compliant interface
├── tests/
│ ├── unit/ # Unit tests
│ └── integration/ # Integration tests
├── data/ # Document storage
├── logs/ # Application logs
├── .env.example # Environment template
├── requirements.txt # Production dependencies
├── requirements-dev.txt # Development dependencies
├── pyproject.toml # Project configuration
└── main.py # Application entry point
```


---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
```bash
python --version
```
2. **Ollama - Install Ollama**
```bash
   # Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull your model
ollama pull qwen3:1.7b

# Pull embedding model (optional, for local embeddings)
ollama pull nomic-embed-text
```
**Installation**
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/enterprise-rag-qa-system.git
cd enterprise-rag-qa-system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Run the application
python main.py
```
The application will start at http://localhost:7860


**⚙️ Configuration**
Environment Variables (.env)
```bash
# Application
APP_NAME="RAG Document Q&A"
ENVIRONMENT="production"  # development, staging, production
DEBUG=false

# Ollama Configuration
OLLAMA_BASE_URL="http://localhost:11434"
OLLAMA_MODEL="qwen3:1.7b"
OLLAMA_TEMPERATURE=0.2

# Embeddings
EMBEDDING_PROVIDER="ollama"  # ollama or huggingface
OLLAMA_EMBEDDING_MODEL="nomic-embed-text"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# RAG Parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=3

# PDF Support
PDF_LOADER="pypdf"  # pypdf, pymupdf, pdfplumber, unstructured

# Gradio Interface
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=false

```


Supported Models
LLM Models (Ollama)
qwen3:1.7b - Fast, efficient (recommended for 8GB RAM)

qwen3:8b - Better quality

llama3.1:8b - Alternative

mistral:7b - Good performance

Embedding Models
Ollama (Local, Privacy-First):

nomic-embed-text - Recommended

mxbai-embed-large - Higher quality

HuggingFace (Cloud):

sentence-transformers/all-MiniLM-L6-v2 - Fast

BAAI/bge-small-en-v1.5 - Better quality

intfloat/multilingual-e5-small - Multilingual

📖 Usage
1. Upload Document
Navigate to "Document Upload" tab:

Upload a .txt or .pdf file

Or provide a URL to a document

Wait for indexing (embedding creation)

2. Ask Questions
Switch to "Chat Interface" tab:

Select your preferred language (English, Italian, Chinese)

Type your question

Watch the AI stream the response in real-time

Continue the conversation with follow-up questions

3. Accessibility Features
Keyboard Navigation: Tab, Shift+Tab, Enter

Screen Reader: Full ARIA support

High Contrast: 7:1 ratio for WCAG AAA

Focus Indicators: Clear visual feedback

**🧪 Testing**
Run All Tests
```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests with coverage
pytest

# Generate HTML coverage report
pytest --cov=src --cov-report=html

# View coverage
open htmlcov/index.html
```

**Run Specific Tests**
 
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_rag_service.py

# Specific test function
pytest tests/unit/test_rag_service.py::TestRAGService::test_initialization

```
