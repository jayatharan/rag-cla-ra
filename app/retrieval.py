import logging
from typing import List, Tuple, Set
import asyncio

from app.schemas import Chunk
from app.embeddings import EmbeddingsService
from app.vector_store import VectorStore
from app.relevance_scoring import RelevanceScorer
from app.refinement import QueryRefiner

logger = logging.getLogger(__name__)

class RetrievalService:
    def __init__(
        self,
        embeddings_service: EmbeddingsService,
        vector_store: VectorStore,
        relevance_scorer: RelevanceScorer,
        query_refiner: QueryRefiner
    ):
        self.embeddings_service = embeddings_service
        self.vector_store = vector_store
        self.relevance_scorer = relevance_scorer
        self.query_refiner = query_refiner
        
        # Config
        self.max_iterations = 2
        self.relevance_threshold = 0.6
        self.min_relevant_chunks = 2

    async def retrieve(self, query: str) -> Tuple[List[Tuple[Chunk, float]], int]:
        """
        Executes the closed-loop retrieval process.
        Returns a list of (Chunk, score) tuples and the number of iterations used.
        """
        iteration = 0
        all_chunks: List[Chunk] = []
        seen_chunk_ids: Set[str] = set()

        current_query = query
        queries_used = [query]

        while iteration <= self.max_iterations:
            logger.info(f"Iteration {iteration}: Retrieving for query '{current_query}'")
            
            # 1. Embed and Retrieve
            query_vector = self.embeddings_service.embed_query(current_query)
            # Use lower threshold for initial retrieval to get candidates
            results = self.vector_store.search(query_vector, limit=5, score_threshold=0.3)
            
            # Convert results to Chunks (reconstructing from payload)
            new_chunks = []
            for point in results:
                # Payload is a dict, need to convert to Chunk object
                # We need document_id and chunk_id
                # The point.id is the chunk_id (UUID string)
                payload = point.payload
                chunk = Chunk(
                    chunk_id=point.id,
                    document_id=payload.get("document_id"),
                    page_number=payload.get("page_number"),
                    text=payload.get("text"),
                    token_count=payload.get("token_count"),
                    heading=payload.get("heading")
                )
                if str(chunk.chunk_id) not in seen_chunk_ids:
                    new_chunks.append(chunk)
                    seen_chunk_ids.add(str(chunk.chunk_id))
            
            all_chunks.extend(new_chunks)
            
            # 2. Score and Filter available chunks
            # We re-score ALL chunks against the original query to ensure relevance to the user's intent
            # Or should we score against the refined query? The prompt says "Relevance Evaluation (LLM scoring)"
            # Usually we score against the original query to ensure we are drifting away.
            # However, if the query was ambiguous, maybe refined is better.
            # Let's score against the ORIGINAL query for final selection, but maybe intermediate steps differ.
            # Prompt says: "Relevance Evaluation ... If insufficient: Query Refinement".
            # So we evaluate what we have.
            
            validated_chunks = await self.relevance_scorer.filter_chunks(query, all_chunks, self.relevance_threshold)
            
            # 3. Check sufficiency
            # "Trigger if: Fewer than 2 relevant chunks - Combined relevance below threshold"
            is_sufficient = len(validated_chunks) >= self.min_relevant_chunks
            
            if is_sufficient or iteration == self.max_iterations:
                logger.info(f"Retrieval complete at iteration {iteration}. Found {len(validated_chunks)} relevant chunks.")
                return validated_chunks, iteration + 1
            
            # 4. Refine Query
            iteration += 1
            logger.info("Insufficient information, refining query...")
            current_query = await self.query_refiner.refine_query(current_query)
            queries_used.append(current_query)
            
        return validated_chunks, iteration
