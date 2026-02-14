from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging

logger = logging.getLogger(__name__)

class QueryRefiner:
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert search query optimizer. The user's original query did not yield sufficient results. "
                       "Your task is to generate a REFINED query that is more likely to retrieve relevant information. "
                       "Simplify the query, use more standard terminology, or break it down if complex. "
                       "Output ONLY the refined query text."),
            ("user", "Original Query: {query}\n\nRefined Query:")
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()

    async def refine_query(self, query: str) -> str:
        try:
            refined_query = await self.chain.ainvoke({"query": query})
            logger.info(f"Refined query: '{query}' -> '{refined_query}'")
            return refined_query.strip()
        except Exception as e:
            logger.error(f"Error refining query: {e}")
            return query
