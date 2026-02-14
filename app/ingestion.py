import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import logging
from typing import List, Optional
from uuid import uuid4
from collections import Counter

from app.schemas import PDFMetadata, PageContent, ExtractedDocument

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self):
        pass

    def process(self, file_path: str) -> ExtractedDocument:
        logger.info(f"Processing PDF: {file_path}")
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            logger.error(f"Failed to open PDF {file_path}: {e}")
            raise e

        pages_content = []
        total_pages = len(doc)

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            heading_candidates = self._extract_headings(page)
            
            # Fallback to OCR if text is empty or too short
            if not text.strip() or len(text.strip()) < 50:
                logger.info(f"Page {page_num} has insufficient text, attempting OCR.")
                text = self._perform_ocr(page)
                # Re-check headings after OCR? OCR usually doesn't give font sizes easily with pytesseract directly 
                # unless using hocr/data. For simplicity, we might lose headings in OCR mode or implement a complex heuristic.
                # Let's stick to text extraction for OCR for now.

            pages_content.append(PageContent(
                page_number=page_num,
                text=text,
                character_count=len(text),
                heading_candidates=heading_candidates
            ))

        metadata = PDFMetadata(
            filename=file_path.split("/")[-1],
            total_pages=total_pages,
            document_id=uuid4()
        )

        return ExtractedDocument(metadata=metadata, pages=pages_content)

    def _extract_headings(self, page: fitz.Page) -> List[str]:
        headings = []
        blocks = page.get_text("dict")["blocks"]
        
        # Calculate most common font size
        font_sizes = []
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        font_sizes.append(span["size"])
        
        if not font_sizes:
            return []

        # Assume body text is the most common font size
        counter = Counter(font_sizes)
        common_size = counter.most_common(1)[0][0]
        
        # Heuristic: Heading is significantly larger than body text (e.g., > 1.2x)
        min_heading_size = common_size * 1.2

        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["size"] >= min_heading_size:
                            text = span["text"].strip()
                            if text:
                                headings.append(text)
        
        # Deduplicate and basic cleanup
        return list(dict.fromkeys(headings))

    def _perform_ocr(self, page: fitz.Page) -> str:
        pix = page.get_pixmap()
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        text = pytesseract.image_to_string(image)
        return text
