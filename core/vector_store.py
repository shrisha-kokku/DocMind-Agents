"""
vector_store.py - embeddings + FAISS index for ALL uploaded PDFs.

One store holds every document. Each chunk remembers which file/page it came
from, so we can (a) cite sources and (b) filter by file name.

Embeddings are normalised, so the inner product (what FAISS IndexFlatIP
computes) equals cosine similarity: 1.0 = identical meaning, ~0 = unrelated.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import faiss
import numpy as np

from core.chunker import Chunk


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class FastEmbedder:
    """Local transformer embedding model (ONNX, no GPU, no API key).
    The model is downloaded/loaded lazily on first use."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        vectors = np.array(list(self._load().embed(texts)), dtype="float32")
        return _normalize(vectors)

    def embed_query(self, query: str) -> np.ndarray:
        # query_embed adds the special query prefix BGE models were trained with
        vector = np.array(list(self._load().query_embed(query)), dtype="float32")
        return _normalize(vector)


@dataclass
class Hit:
    chunk: Chunk
    score: float


class VectorStore:
    def __init__(self, embedder):
        self.embedder = embedder              # anything with embed_passages / embed_query
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: list[Chunk] = []         # row i of the index <-> self.chunks[i]
        self.docs: dict[str, dict] = {}       # file name -> {"hash", "pages", "chunk_ids"}

    # ---------- documents ----------
    def has_hash(self, file_hash: str) -> bool:
        return any(d["hash"] == file_hash for d in self.docs.values())

    def document_names(self) -> list[str]:
        return list(self.docs)

    def find_document(self, name: str) -> Optional[str]:
        """Forgiving file-name lookup (case-insensitive, partial match)."""
        name = name.strip().lower()
        for doc in self.docs:
            if doc.lower() == name:
                return doc
        matches = [doc for doc in self.docs if name and name in doc.lower()]
        return matches[0] if len(matches) == 1 else None

    def get_document_chunks(self, name: str) -> list[Chunk]:
        return [self.chunks[i] for i in self.docs[name]["chunk_ids"]]

    def unique_name(self, name: str) -> str:
        """Two different files with the same name must not overwrite each other."""
        if name not in self.docs:
            return name
        stem, dot, ext = name.rpartition(".")
        n = 2
        while f"{stem} ({n}).{ext}" in self.docs:
            n += 1
        return f"{stem} ({n}).{ext}"

    # ---------- indexing ----------
    def add_document(self, name: str, file_hash: str, chunks: list[Chunk], pages: int,
                     progress: Optional[Callable[[float, str], None]] = None,
                     batch_size: int = 64) -> str:
        """Embed chunks in batches (so big PDFs show progress) and add to FAISS."""
        name = self.unique_name(name)
        first_id = len(self.chunks)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = self.embedder.embed_passages([c.text for c in batch])
            if self.index is None:
                self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
            self.chunks.extend(batch)
            if progress:
                done = min(start + batch_size, len(chunks))
                progress(done / len(chunks), f"Embedding {done}/{len(chunks)} chunks")
        self.docs[name] = {"hash": file_hash, "pages": pages,
                           "chunk_ids": list(range(first_id, len(self.chunks)))}
        return name

    # ---------- searching ----------
    def search(self, query: str, k: int = 5, filename: Optional[str] = None) -> list[Hit]:
        """Top-k most similar chunks (optionally only inside one file)."""
        if self.index is None or self.index.ntotal == 0:
            return []
        query_vec = self.embedder.embed_query(query).reshape(1, -1)
        # With a file filter we fetch everything and filter afterwards (exact search is fast).
        fetch = self.index.ntotal if filename else min(k, self.index.ntotal)
        scores, ids = self.index.search(query_vec, fetch)
        hits: list[Hit] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx]
            if filename and chunk.source != filename:
                continue
            hits.append(Hit(chunk=chunk, score=float(score)))
            if len(hits) == k:
                break
        return hits
