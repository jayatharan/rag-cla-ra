import logging
from typing import Dict, Any, List

from app.retrieval import RetrievalService
from app.answer_generator import AnswerGenerator
from app.citation_validator import CitationValidator
from app.schemas import Chunk

logger = logging.getLogger(__name__)

class ClaraPipeline:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        answer_generator: AnswerGenerator,
        citation_validator: CitationValidator
    ):
        self.retrieval_service = retrieval_service
        self.answer_generator = answer_generator
        self.citation_validator = citation_validator

    async def run(self, query: str) -> Dict[str, Any]:
        logger.info(f"Starting CLA-RA pipeline for query: {query}")
        
        # 1. Retrieval
        chunks, iterations = await self.retrieval_service.retrieve(query)
        
        if not chunks:
            logger.info("No relevant chunks found.")
            return {
                "answer": "The document does not contain sufficient information to answer this question.",
                "confidence_score": 0.0,
                "iterations_used": iterations,
                "sources": []
            }

        # 2. Generation & Validation Loop (Max 1 retry)
        for attempt in range(2):
            logger.info(f"Generating answer (Attempt {attempt+1})")
            response = await self.answer_generator.generate_answer(query, chunks, iterations)
            
            # Hydrate sources with chunk text for validation
            sources = response.get("sources", [])
            for source in sources:
                chunk_id = source.get("chunk_id")
                # Find matching chunk
                # chunk_id from LLM might be a string, chunk.chunk_id is UUID
                matching_chunk = next((c for c, _ in chunks if str(c.chunk_id) == str(chunk_id)), None)
                if matching_chunk:
                    source["chunk_text"] = matching_chunk.text
                else:
                    logger.warning(f"Source chunk_id {chunk_id} not found in retrieved chunks. Validation may fail.")

            # 3. Validate
            is_valid = self.citation_validator.validate(response.get("answer", ""), sources)
            
            if is_valid:
                logger.info("Answer validated successfully.")
                return response
            
            logger.warning("Citation validation failed. Retrying...")
            # Ideally retry with feedback, but for simplicity we just regenerate.
            # Maybe restrict chunks to only high confidence ones?
            
        logger.error("Failed to generate valid answer after retries.")
        return {
            "answer": "Failed to generate a valid answer with correct citations.",
            "confidence_score": 0.0,
            "iterations_used": iterations,
            "sources": []
        }
