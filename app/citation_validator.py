import logging
from typing import List, Dict, Any
from app.schemas import Chunk

logger = logging.getLogger(__name__)

class CitationValidator:
    def __init__(self):
        pass

    def validate(self, answer: str, sources: List[Dict[str, Any]]) -> bool:
        """
        Validates that the quoted text in sources actually exists in the provided chunks.
        The sources list is expected to contain 'quoted_text' and the corresponding chunk text/content.
        """
        if not sources:
            logger.warning("No sources provided for validation.")
            return False

        for source in sources:
            quoted_text = source.get("quoted_text", "").strip()
            chunk_text = source.get("chunk_text", "").strip() # Assuming source object has full chunk text for verification

            if not quoted_text:
                logger.warning("Source missing quoted_text.")
                return False
            
            # Normalize whitespace for robust comparison
            # Remove all whitespace to ignore formatting differences (newlines, extra spaces)
            quoted_text_norm = "".join(quoted_text.split())
            chunk_text_norm = "".join(chunk_text.split())

            if quoted_text_norm not in chunk_text_norm:
                logger.warning(f"Citation mismatch: Quote '{quoted_text}' not found in chunk text (normalized).")
                return False

        logger.info("All citations validated successfully.")
        return True
