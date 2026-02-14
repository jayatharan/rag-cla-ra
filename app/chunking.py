import tiktoken
from typing import List, Optional
from uuid import uuid4
import logging

from app.schemas import ExtractedDocument, Chunk

logger = logging.getLogger(__name__)

class Chunker:
    def __init__(self, model_name: str = "cl100k_base", chunk_size: int = 500, chunk_overlap: int = 100):
        self.tokenizer = tiktoken.get_encoding(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: ExtractedDocument) -> List[Chunk]:
        logger.info(f"Chunking document {document.metadata.filename} with {len(document.pages)} pages.")
        chunks = []
        
        for page in document.pages:
            page_chunks = self._chunk_page(
                page.text,
                page.page_number,
                document.metadata.document_id,
                page.heading_candidates
            )
            chunks.extend(page_chunks)
        
        logger.info(f"Generated {len(chunks)} chunks.")
        return chunks

    def _chunk_page(self, text: str, page_number: int, document_id: str, headings: List[str]) -> List[Chunk]:
        chunks = []
        # Simple sentence splitting for now. Could be improved with nltk or spaCy.
        # Splitting by newline and period is a basic heuristic.
        sentences = text.replace('\n', ' ').split('. ') 
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        current_chunk_tokens = []
        current_chunk_text = []
        
        for sentence in sentences:
            sentence_tokens = self.tokenizer.encode(" " + sentence)
            
            if len(current_chunk_tokens) + len(sentence_tokens) > self.chunk_size:
                # Commit current chunk
                if current_chunk_text:
                    chunk_text = " ".join(current_chunk_text).strip()
                    chunks.append(self._create_chunk(chunk_text, page_number, document_id, headings))
                    
                    # Handle overlap
                    # Keep last N tokens that fit within overlap
                    overlap_tokens = []
                    overlap_text = []
                    tokens_processed = 0
                    
                    # Iterate backwards to fill overlap
                    for i in range(len(current_chunk_text)-1, -1, -1):
                        segment = current_chunk_text[i]
                        segment_tokens = self.tokenizer.encode(segment if i == 0 else " " + segment)
                        if tokens_processed + len(segment_tokens) <= self.chunk_overlap:
                            overlap_tokens.insert(0, segment_tokens)
                            overlap_text.insert(0, segment)
                            tokens_processed += len(segment_tokens)
                        else:
                            break
                    
                    current_chunk_tokens = [t for sublist in overlap_tokens for t in sublist]
                    current_chunk_text = overlap_text

            current_chunk_tokens.extend(sentence_tokens)
            current_chunk_text.append(sentence)
        
        # Add the last chunk
        if current_chunk_text:
            chunk_text = " ".join(current_chunk_text).strip()
            chunks.append(self._create_chunk(chunk_text, page_number, document_id, headings))
            
        return chunks

    def _create_chunk(self, text: str, page_number: int, document_id: str, headings: List[str]) -> Chunk:
        # Determine if this chunk contains any of the headings
        chunk_heading = None
        for heading in headings:
            if heading in text:
                chunk_heading = heading
                break
                
        return Chunk(
            document_id=document_id,
            page_number=page_number,
            text=text,
            token_count=len(self.tokenizer.encode(text)),
            heading=chunk_heading
        )
