# PDF CLA-RA System

A production-ready Closed-Loop Adaptive Retrieval-Augmented Generation (CLA-RA) system for PDFs.

## 🚀 Features

- **PDF Ingestion**: Robust text extraction using PyMuPDF with Tesseract OCR fallback.
- **Intelligent Chunking**: Page-aware, sentence-boundary respecting chunking with overlap.
- **Vector Storage**: Uses Qdrant for scalable vector search.
- **CLA-RA Pipeline**:
  - Initial retrieval with relevance scoring.
  - Adaptive query refinement loops.
  - Strict citation validation.
  - "I don't know" for insufficient data.
- **FastAPI Interface**: Clean REST API for ingestion and querying.

## 🛠️ Setup

### 1. Environment Configuration
Copy `.env.example` to `.env` and set your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY
```

## 🐳 Running with Docker (Recommended)

1.  **Set Environment Variable**:
    Ensure your OpenAI API key is available to Docker (or set it in `.env`):
    ```bash
    export OPENAI_API_KEY=your_key_here
    ```

2.  **Build and Run**:
    ```bash
    docker-compose up --build
    ```

    This will start:
    -   `pdf_clara_api`: The FastAPI application (Port 8000)
    -   `pdf_clara_qdrant`: Vector database (Port 6333)
    -   `pdf_clara_postgres`: Metadata database (Port 5432)

    The API will be available at `http://localhost:8000`.
    Swagger UI is available at `http://localhost:8000/docs`.

### Manual Setup (Local Development)

1.  **Install Dependencies**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
    *Note: Requires Tesseract OCR installed on your system.*

2.  **Start Infrastructure Only**
    ```bash
    docker-compose up -d qdrant postgres
    ```

3.  **Run the Application**
    ```bash
    uvicorn app.main:app --reload
    ```

## 📖 Usage

### Ingest a PDF
```bash
curl -X POST "http://localhost:8000/ingest" -F "file=@/path/to/your/document.pdf"
```

### Query
```bash
curl -X POST "http://localhost:8000/query" -H "Content-Type: application/json" -d '{"query": "What is the main conclusion of the paper?"}'
```

## 🏗️ Architecture

1.  **Ingestion**: PDFs -> Text -> Chunks -> Embeddings -> Qdrant.
2.  **Retrieval**: Query -> Embed -> Search -> Score -> Filter.
3.  **Refinement**: If retrieval is poor -> LLM Refines Query -> Recursion.
4.  **Generation**: Validated Context -> LLM Answer -> JSON.
5.  **Validation**: Answer -> Check Citations -> Pass/Fail.
