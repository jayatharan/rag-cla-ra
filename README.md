# PDF CLA-RA System

A production-ready **Closed-Loop Adaptive Retrieval-Augmented Generation (CLA-RA)** system for PDFs. This system is designed to provide highly accurate, citation-backed answers from PDF documents, prioritizing factual correctness and hallucination prevention.

## 🎯 Objective

Initialize a production-ready Python project implementing a complete **PDF → Closed-Loop Adaptive Retrieval-Augmented Generation (CLA-RA) pipeline**.

The system:
1.  Extracts structured text from PDFs (including page numbers).
2.  Intelligently chunks content (page-bound).
3.  Generates embeddings and stores them in Qdrant.
4.  Implements iterative adaptive retrieval.
5.  Validates context relevance before answering.
6.  Generates answers strictly from validated context.
7.  **ALWAYS cites**: Document name, Page number, and Exact quoted text.
8.  Refuses to answer if insufficient evidence exists.

**The system differs from standard RAG by enforcing a strict closed-loop verification process to eliminate hallucinations.**

## 🚀 Features

-   **Robust Ingestion**: Uses `PyMuPDF` for high-fidelity text extraction, with `pytesseract` OCR fallback for scanned documents.
-   **Smart Chunking**: Page-aware chunking that respects sentence boundaries and maintains context with overlap.
-   **Vector Storage**: High-performance vector search using **Qdrant**.
-   **CLA-RA Pipeline**:
    -   **Initial Retrieval**: Semantic search with `BAAI/bge-small-en` embeddings.
    -   **Relevance Scoring**: LLM-based evaluation of retrieved chunks (0-1 score).
    -   **Adaptive Refinement**: Automatically refines queries and re-retrieves if initial context is insufficient.
    -   **Citation Validation**: Post-generation check to ensure every claim is backed by a direct quote from the source text.
-   **API**: Clean **FastAPI** interface for ingestion and querying.

## 🏗️ Architecture

### High-Level Flow

1.  **Ingestion**: PDF $\to$ Text Extraction $\to$ Chunking $\to$ Embedding $\to$ Qdrant Store.
2.  **Retrieval Loop**:
    -   Query $\to$ Vector Search $\to$ Top-K Chunks.
    -   **Relevance Check**: Are chunks relevant? If no $\to$ Refine Query $\to$ Loop.
3.  **Generation**:
    -   Validated Context + Query $\to$ LLM Answer.
4.  **Verification**:
    -   Extract Citations $\to$ Verify Quotes against Source Text.
    -   If Valid $\to$ Return Answer.
    -   If Invalid $\to$ Retry Generation.

### Project Structure

```
pdf_clara/
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── ingestion.py            # PDF text extraction (PyMuPDF + OCR)
│   ├── chunking.py             # Content chunking logic
│   ├── embeddings.py           # Embedding generation
│   ├── vector_store.py         # Qdrant interface
│   ├── retrieval.py            # Search logic
│   ├── refinement.py           # Query refinement strategies
│   ├── relevance_scoring.py    # Context relevance evaluation
│   ├── citation_validator.py   # Hallucination check & citation verification
│   ├── clara_pipeline.py       # Main orchestration logic
│   ├── answer_generator.py     # LLM response generation
│   └── schemas.py              # Pydantic data models
├── storage/
│   └── documents/              # PDF storage
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Docker services (App, Qdrant, Postgres)
├── .env.example                # Environment variables template
└── README.md                   # Project documentation
```

## 🛠️ Setup

### 1. Environment Configuration

Copy `.env.example` to `.env` and set your OpenAI API key:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=your-key-here
```

### 2. Running with Docker (Recommended)

The easiest way to run the full stack (API + Database + Vector Store).

```bash
# 1. Build and start services
docker-compose up --build

# The API will be available at http://localhost:8000
# Swagger UI: http://localhost:8000/docs
```

### 3. Manual Setup (Local Development)

If you prefer running the Python app locally while keeping databases in Docker:

1.  **Start Infrastructure**:
    ```bash
    docker-compose up -d qdrant postgres
    ```

2.  **Install Dependencies**:
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    # Note: Requires Tesseract OCR installed on your system for OCR fallback
    ```

3.  **Run Application**:
    ```bash
    uvicorn app.main:app --reload
    ```

## 📖 Usage

### 1. Ingest a Document

Upload a PDF to be indexed.

```bash
curl -X POST "http://localhost:8000/ingest" \
     -F "file=@/path/to/your/document.pdf"
```

### 2. Query the System

Ask a question. The system will perform the full retrieval and validation loop.

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the key findings regarding climate change?"}'
```

### Response Format

The system returns a JSON object with the answer, confidence score, and verified sources.

```json
{
  "answer": "The study concludes that...",
  "confidence_score": 0.95,
  "iterations_used": 1,
  "sources": [
    {
      "filename": "report.pdf",
      "page": 12,
      "chunk_id": "uuid...",
      "quoted_text": "Global temperatures have risen by...",
      "similarity_score": 0.88,
      "relevance_score": 0.92
    }
  ]
}
```

## 🔐 Engineering Standards

-   **Type Hints**: Fully typed codebase.
-   **Pydantic**: Robust data validation.
-   **Async/Await**: High-concurrency support.
-   **Logging**: Structured JSON logging for observability.
-   **Error Handling**: Graceful failure modes.
