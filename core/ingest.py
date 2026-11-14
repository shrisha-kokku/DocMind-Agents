"""
ingest.py - the whole "upload -> searchable" pipeline in one function.

PDF bytes -> pages -> chunks -> embeddings -> FAISS
"""
import hashlib
from typing import Callable, Optional

import config
from core.chunker import chunk_pages
from core.pdf_loader import extract_pages
from core.vector_store import VectorStore


def file_hash(data: bytes) -> str:
    """Fingerprint of the file content, used to skip duplicate uploads."""
    return hashlib.sha1(data).hexdigest()


def ingest_pdf(store: VectorStore, filename: str, data: bytes,
               progress: Optional[Callable[[float, str], None]] = None) -> dict:
    """Index one PDF. Returns a small stats dict. Raises ValueError if unusable."""
    digest = file_hash(data)
    if store.has_hash(digest):
        return {"status": "duplicate", "name": filename}

    if progress:
        progress(0.0, "Reading PDF...")
    pages = extract_pages(data)
    if not pages:
        raise ValueError("No extractable text found (maybe a scanned/image-only PDF).")

    if progress:
        progress(0.0, "Splitting into chunks...")
    chunks = chunk_pages(pages, filename, config.CHUNK_SIZE, config.CHUNK_OVERLAP)

    name = store.add_document(filename, digest, chunks, pages=len(pages), progress=progress)
    return {"status": "indexed", "name": name, "pages": len(pages), "chunks": len(chunks)}
