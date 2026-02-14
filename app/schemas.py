from pydantic import BaseModel, Field
from typing import List, Optional
from uuid import UUID, uuid4

class PDFMetadata(BaseModel):
    filename: str
    total_pages: int
    document_id: UUID = Field(default_factory=uuid4)

class PageContent(BaseModel):
    page_number: int
    text: str
    character_count: int
    heading_candidates: List[str] = Field(default_factory=list)

class ExtractedDocument(BaseModel):
    metadata: PDFMetadata
    pages: List[PageContent]

class Chunk(BaseModel):
    chunk_id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    page_number: int
    text: str
    token_count: int
    heading: Optional[str] = None
