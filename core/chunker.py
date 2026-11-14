"""
chunker.py - split page text into overlapping chunks.

Strategy ("recursive" splitting): try to cut at the most natural boundary
first (paragraph -> line -> sentence -> word). Only if a piece is still too
big do we fall back to the next, smaller boundary. Small pieces are then
merged back up to CHUNK_SIZE, and each new chunk starts with the tail of the
previous one (overlap) so a sentence cut at a boundary is not lost.
"""
import re
from dataclasses import dataclass

import config

# Regex boundaries, from "biggest" to "smallest".
SEPARATORS = [r"\n\s*\n", r"\n", r"(?<=[.!?])\s+", r"\s+"]


@dataclass
class Chunk:
    text: str
    source: str   # file name
    page: int     # page number the chunk came from


def _split(text: str, limit: int, seps: list[str]) -> list[str]:
    """Recursively break `text` into pieces no longer than `limit`."""
    if len(text) <= limit:
        return [text]
    if not seps:  # last resort: hard cut (e.g. one giant word / no spaces)
        return [text[i:i + limit] for i in range(0, len(text), limit)]
    pieces: list[str] = []
    for part in re.split(seps[0], text):
        part = part.strip()
        if not part:
            continue
        if len(part) > limit:
            pieces.extend(_split(part, limit, seps[1:]))  # try a finer separator
        else:
            pieces.append(part)
    return pieces


def _tail(text: str, n: int) -> str:
    """Last ~n characters of text, starting at a word boundary."""
    tail = text[-n:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else tail


def split_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Split one text into overlapping chunks of at most ~`size` characters."""
    # pieces are (size - overlap) long so that tail + piece still fits in `size`
    pieces = _split(text.strip(), size - overlap, SEPARATORS)
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 1 + len(piece) > size:
            chunks.append(current)
            current = f"{_tail(current, overlap)} {piece}"   # start next chunk with overlap
        else:
            current = f"{current} {piece}".strip()
    if current:
        chunks.append(current)
    return chunks


def chunk_pages(pages, source: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    """Chunk every page separately so each chunk keeps its page number."""
    result: list[Chunk] = []
    for page in pages:
        for text in split_text(page.text, size, overlap):
            result.append(Chunk(text=text, source=source, page=page.number))
    return result
