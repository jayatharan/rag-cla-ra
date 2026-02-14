from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
from typing import List, Tuple

from app.schemas import Chunk

logger = logging.getLogger(__name__)

class RelevanceScorer:
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a relevance evaluator. You will be given a query and a document chunk. "
                       "Your task is to rate the relevance of the chunk to the query on a scale of 0.0 to 1.0. "
                       "0.0 means completely irrelevant, 1.0 means highly relevant and contains the exact answer. "
                       "Output ONLY the numeric score, nothing else."),
            ("user", "Query: {query}\n\nChunk: {chunk_text}")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()

    async def score_chunk(self, query: str, chunk: Chunk) -> float:
        try:
            score_str = await self.chain.ainvoke({"query": query, "chunk_text": chunk.text})
            score = float(score_str.strip())
            return score
        except Exception as e:
            logger.error(f"Error scoring chunk {chunk.chunk_id}: {e}")
            return 0.0

    async def filter_chunks(self, query: str, chunks: List[Chunk], threshold: float = 0.6) -> List[Tuple[Chunk, float]]:
        scored_chunks = []
        for chunk in chunks:
            score = await self.score_chunk(query, chunk)
            if score >= threshold:
                scored_chunks.append((chunk, score))
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks
