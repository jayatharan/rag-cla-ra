from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Optional, Any
import os
import logging
from uuid import UUID

from app.schemas import Chunk

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, collection_name: str = "pdf_collection"):
        self.host = os.getenv("QDRANT_HOST", "localhost")
        self.port = int(os.getenv("QDRANT_PORT", 6333))
        self.client = QdrantClient(host=self.host, port=self.port)
        self.collection_name = collection_name
        
    def ensure_collection(self, vector_size: int = 384, recreate: bool = False):
        if recreate:
            logger.info(f"Recreating collection {self.collection_name}")
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )
            self._create_indexes()
        elif not self.client.collection_exists(self.collection_name):
            logger.info(f"Creating collection {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
            )
            self._create_indexes()
            
    def _create_indexes(self):
        logger.info(f"Creating indexes for {self.collection_name}")
        # Indexed fields: document_id, filename, page_number, heading
        self.client.create_payload_index(self.collection_name, "document_id", models.PayloadSchemaType.KEYWORD)
        self.client.create_payload_index(self.collection_name, "filename", models.PayloadSchemaType.KEYWORD)
        self.client.create_payload_index(self.collection_name, "page_number", models.PayloadSchemaType.INTEGER)
        self.client.create_payload_index(self.collection_name, "heading", models.PayloadSchemaType.KEYWORD)

    def upsert_chunks(self, chunks: List[Chunk], embeddings: List[List[float]], filename: str):
        logger.info(f"Upserting {len(chunks)} chunks into {self.collection_name}")
        points = []
        for i, chunk in enumerate(chunks):
            points.append(models.PointStruct(
                id=str(chunk.chunk_id),
                vector=embeddings[i],
                payload={
                    "document_id": str(chunk.document_id),
                    "filename": filename,
                    "page_number": chunk.page_number,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "heading": chunk.heading
                }
            ))
            
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_vector: List[float], limit: int = 5, score_threshold: float = 0.0) -> List[models.ScoredPoint]:
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold
        )
        return result.points
        
    def delete_document(self, document_id: str):
        logger.info(f"Deleting document {document_id}")
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                )
            )
        )
