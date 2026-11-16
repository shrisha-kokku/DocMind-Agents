"""
pdf_loader.py - PDF bytes -> list of pages with clean text.

PyMuPDF is used because it is very fast and reads page by page, so even
large PDFs (hundreds of pages) load in a couple of seconds.
"""
import re
from dataclasses import dataclass

import pymupdf  # PyMuPDF


@dataclass
class Page:
    number: int   # 1-based page number (used later for citations)
    text: str


def extract_pages(data: bytes) -> list[Page]:
    """Return the non-empty pages of a PDF. Raises ValueError for unusable files."""
    pages: list[Page] = []
    # Open straight from memory (bytes) - no temp file needed.
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        if doc.needs_pass:
            raise ValueError("This PDF is password protected.")
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = re.sub(r"[ \t]+", " ", text)       # collapse spaces/tabs
            text = re.sub(r"\n{3,}", "\n\n", text)    # collapse blank lines
            text = text.strip()
            if text:                                   # skip blank / image-only pages
                pages.append(Page(number=index, text=text))
    return pages
