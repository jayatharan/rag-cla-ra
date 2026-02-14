# PDF CLA-RA System --- Project Initialization Prompt

## 🎯 Objective

Initialize a production-ready Python project implementing a complete:

> **PDF → Closed-Loop Adaptive Retrieval-Augmented Generation (CLA-RA)
> pipeline**

The system must:

1.  Extract structured text from PDFs (including page numbers)
2.  Intelligently chunk content (page-bound)
3.  Generate embeddings
4.  Store vectors in Qdrant
5.  Implement iterative adaptive retrieval
6.  Validate context relevance before answering
7.  Generate answers strictly from validated context
8.  Always cite:
    -   Document name
    -   Page number
    -   Exact quoted text
9.  Refuse to answer if insufficient evidence exists

The system must NEVER hallucinate.\
The system must ALWAYS verify retrieved context before answering.

------------------------------------------------------------------------

## 🏗️ High-Level CLA-RA Architecture

### 🔁 Closed-Loop Retrieval Flow

    User Query
    → Initial Embedding
    → Initial Vector Retrieval (Top-K)
    → Relevance Evaluation (LLM scoring)
    → If insufficient:
       - Query Refinement
       - Re-embed
       - Re-retrieve
       - Merge & Deduplicate
    → Confidence Check
    → Final Answer Generation (context-restricted)
    → Citation Validation
    → Response

------------------------------------------------------------------------

## 1️⃣ Storage Layer

-   Store PDFs in: `/storage/documents/`
-   PostgreSQL for metadata
-   Qdrant for vector storage
-   UUID-based document IDs
-   Multi-document support

------------------------------------------------------------------------

## 2️⃣ PDF Processing

Use:

-   `PyMuPDF`
-   Fallback to `pytesseract` OCR if no text layer

Extract:

-   Page number
-   Raw text
-   Heading candidates
-   Character count

------------------------------------------------------------------------

## 3️⃣ Chunking Strategy

-   500 tokens
-   100 token overlap
-   Never cross page boundaries
-   Heading-aware chunk prioritization

------------------------------------------------------------------------

## 4️⃣ Embeddings

-   `sentence-transformers`
-   Model: `BAAI/bge-small-en`
-   Normalize embeddings
-   Batch insertion into Qdrant
-   Store full chunk text in payload

------------------------------------------------------------------------

## 5️⃣ Qdrant Configuration

-   Distance metric: `cosine`
-   Payload indexing enabled
-   Indexed fields:
    -   document_id
    -   filename
    -   page_number
    -   heading
-   Support metadata filtering
-   Support collection recreation
-   Support document deletion

------------------------------------------------------------------------

## 6️⃣ CLA-RA Retrieval Layer

### Step 1 --- Initial Retrieval

-   Embed query
-   Retrieve Top 5

### Step 2 --- Relevance Evaluation

LLM scores each chunk (0--1 relevance).\
Discard chunks below threshold (e.g., 0.6).

### Step 3 --- Adaptive Query Refinement

Trigger if: - Fewer than 2 relevant chunks - Combined relevance below
threshold - Multi-hop or ambiguous query

Limit to max 2 refinement loops.

### Step 4 --- Context Consolidation

-   Merge unique chunks
-   Deduplicate
-   Re-rank
-   Select top validated chunks

------------------------------------------------------------------------

## 7️⃣ Strict Answer Rules

-   Use ONLY validated context
-   Quote exact supporting sentence
-   If insufficient:

> "The document does not contain sufficient information to answer this
> question."

------------------------------------------------------------------------

## 8️⃣ Citation Enforcement

After answer generation:

1.  Extract quoted sentences
2.  Verify existence in retrieved chunks
3.  If mismatch → reject and retry once
4.  No citation → no answer

------------------------------------------------------------------------

## 📦 Response Format

``` json
{
  "answer": "...",
  "confidence_score": 0.82,
  "iterations_used": 2,
  "sources": [
    {
      "filename": "...",
      "page": 3,
      "chunk_id": "...",
      "quoted_text": "...",
      "similarity_score": 0.87,
      "relevance_score": 0.91
    }
  ]
}
```

------------------------------------------------------------------------

## 📂 Project Structure

    pdf_clara/
    │
    ├── app/
    │   ├── main.py
    │   ├── ingestion.py
    │   ├── chunking.py
    │   ├── embeddings.py
    │   ├── vector_store.py
    │   ├── retrieval.py
    │   ├── refinement.py
    │   ├── relevance_scoring.py
    │   ├── citation_validator.py
    │   ├── clara_pipeline.py
    │
    ├── storage/
    │   └── documents/
    │
    ├── requirements.txt
    ├── docker-compose.yml
    ├── README.md
    └── .env.example

------------------------------------------------------------------------

## 🔐 Engineering Requirements

-   Type hints everywhere
-   Pydantic schemas
-   Structured JSON logging
-   Async where appropriate
-   Environment variables via `.env`
-   Dependency injection
-   Graceful exception handling
-   No hardcoded secrets

------------------------------------------------------------------------

## 📌 Final Instruction for Code Generator

You are a senior AI systems architect specializing in high-accuracy
retrieval systems.

Generate a clean, production-grade, scalable implementation of a
Closed-Loop Adaptive RAG (CLA-RA) PDF system.

Prioritize factual accuracy over latency.\
Hallucination prevention is more important than speed.
