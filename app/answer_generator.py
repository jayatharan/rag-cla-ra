from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from app.schemas import Chunk

logger = logging.getLogger(__name__)

class AnswerSource(BaseModel):
    filename: str
    page: int
    chunk_id: str
    quoted_text: str
    similarity_score: float = 0.0
    relevance_score: float = 0.0

class AnswerResponse(BaseModel):
    answer: str
    confidence_score: float
    iterations_used: int
    sources: List[AnswerSource]

class AnswerGenerator:
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
        
        self.parser = JsonOutputParser(pydantic_object=AnswerResponse)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a strict, factual assistant. You answer questions based ONLY on the provided context.
            Rules:
            1. Use ONLY the provided context.
            2. If the answer is not in the context, output: "The document does not contain sufficient information to answer this question."
            3. You must cite your sources by providing the exact quoted text from the context.
            4. Do not hallucinate.
            5. Return the response in the specified JSON format.
            """),
            ("user", """Context:
            {context}
            
            Question: {question}
            Iterations used: {iterations}
            
            {format_instructions}
            """)
        ])
        
        self.chain = self.prompt | self.llm | self.parser

    async def generate_answer(self, query: str, chunks: List[tuple[Chunk, float]], iterations: int) -> Dict[str, Any]:
        context_str = ""
        for i, (chunk, score) in enumerate(chunks):
            context_str += f"Source {i+1} (ID: {chunk.chunk_id}, Page: {chunk.page_number}, File: {chunk.document_id}):\n{chunk.text}\n\n"
            
        try:
            response = await self.chain.ainvoke({
                "context": context_str,
                "question": query,
                "iterations": iterations,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Enrich sources with metadata from chunks since LLM might not have all details or might hallucinate IDs?
            # Actually, we should trust the LLM to pick the right quote, but we can verify/fill metadata.
            # The prompt asks for filename, page, etc.
            # Let's map back if possible, or just rely on the LLM to output what we gave it in the context header.
            # To be safe, we can match chunk_id if the LLM outputs it.
            
            return response
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": "An error occurred while generating the answer.",
                "confidence_score": 0.0,
                "iterations_used": iterations,
                "sources": []
            }
