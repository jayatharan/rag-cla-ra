from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)

class EmbeddingsService:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
    
    def embed_query(self, query: str) -> List[float]:
        # BGE models often benefit from an instruction for queries
        # "Represent this sentence for searching relevant passages: "
        instruction = "Represent this sentence for searching relevant passages: "
        embedding = self.model.encode(instruction + query, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(documents, normalize_embeddings=True)
        return embeddings.tolist()
