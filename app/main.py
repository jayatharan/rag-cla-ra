from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import shutil
import logging
from uuid import uuid4
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.ingestion import PDFProcessor
from app.chunking import Chunker
from app.embeddings import EmbeddingsService
from app.vector_store import VectorStore
from app.clara_pipeline import ClaraPipeline
from app.schemas import Chunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PDF CLA-RA System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.retrieval import RetrievalService
from app.answer_generator import AnswerGenerator
from app.citation_validator import CitationValidator
from app.refinement import QueryRefiner
from app.relevance_scoring import RelevanceScorer

# Initialize services
pdf_processor = PDFProcessor()
chunker = Chunker()
embeddings_service = EmbeddingsService()
vector_store = VectorStore()

# Initialize retrieval components
relevance_scorer = RelevanceScorer()
query_refiner = QueryRefiner()
retrieval_service = RetrievalService(
    embeddings_service=embeddings_service,
    vector_store=vector_store,
    relevance_scorer=relevance_scorer,
    query_refiner=query_refiner
)

answer_generator = AnswerGenerator()
citation_validator = CitationValidator()

pipeline = ClaraPipeline(
    retrieval_service=retrieval_service, 
    answer_generator=answer_generator, 
    citation_validator=citation_validator
)

# Ensure storage directory exists
STORAGE_PATH = os.getenv("STORAGE_PATH", "./storage/documents")
os.makedirs(STORAGE_PATH, exist_ok=True)

class QueryRequest(BaseModel):
    query: str

class IngestionResponse(BaseModel):
    filename: str
    document_id: str
    chunks_created: int
    message: str

@app.on_event("startup")
async def startup_event():
    vector_store.ensure_collection()

@app.post("/ingest", response_model=IngestionResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    filename = file.filename
    file_path = os.path.join(STORAGE_PATH, filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Ingesting file: {filename}")
        
        # 1. Process PDF
        document = pdf_processor.process(file_path)
        
        # 2. Chunk
        chunks = chunker.chunk(document)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No text extracted from PDF.")
            
        # 3. Embed
        texts = [chunk.text for chunk in chunks]
        embeddings = embeddings_service.embed_documents(texts)
        
        # 4. Store
        vector_store.upsert_chunks(chunks, embeddings, filename)
        
        return IngestionResponse(
            filename=filename,
            document_id=str(document.metadata.document_id),
            chunks_created=len(chunks),
            message="Document ingested successfully."
        )
        
    except Exception as e:
        logger.error(f"Error ingesting file {filename}: {e}")
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_pipeline(request: QueryRequest):
    try:
        response = await pipeline.run(request.query)
        return response
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}
