"""
Offline tests - no API key and no model download needed.
We replace the real embedding model with a tiny fake one (hashed bag-of-words).
Run:  pytest -q
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf  # noqa: E402

from core.chunker import split_text  # noqa: E402
from core.ingest import ingest_pdf  # noqa: E402
from core.pdf_loader import extract_pages  # noqa: E402
from core.vector_store import VectorStore  # noqa: E402


class FakeEmbedder:
    """Words -> hashed vector. Similar words => similar vectors. Good enough for tests."""
    dim = 256

    def _vec(self, text):
        v = np.zeros(self.dim, dtype="float32")
        for word in text.lower().split():
            v[hash(word) % self.dim] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed_passages(self, texts):
        return np.array([self._vec(t) for t in texts], dtype="float32")

    def embed_query(self, q):
        return self._vec(q)


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 780), text)
    return doc.tobytes()


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = split_text(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 301 for c in chunks)
    # overlap: end of chunk N appears at the start of chunk N+1
    assert chunks[0].split()[-1] in chunks[1]


def test_short_text_is_one_chunk():
    assert split_text("Hello world.", size=300, overlap=50) == ["Hello world."]


def test_pdf_pages_are_extracted():
    pages = extract_pages(make_pdf(["First page about cats.", "Second page about dogs."]))
    assert [p.number for p in pages] == [1, 2]
    assert "cats" in pages[0].text


def test_search_finds_right_document_and_filters():
    store = VectorStore(FakeEmbedder())
    ingest_pdf(store, "cats.pdf", make_pdf(["Cats purr and sleep all day. Cats love fish."]))
    ingest_pdf(store, "cars.pdf", make_pdf(["Engines burn petrol. Cars have four wheels and brakes."]))

    hits = store.search("do cats purr", k=1)
    assert hits[0].chunk.source == "cats.pdf"

    only_cars = store.search("do cats purr", k=3, filename="cars.pdf")
    assert all(h.chunk.source == "cars.pdf" for h in only_cars)


def test_duplicate_upload_is_skipped_and_same_name_is_renamed():
    store = VectorStore(FakeEmbedder())
    data = make_pdf(["Same content."])
    assert ingest_pdf(store, "a.pdf", data)["status"] == "indexed"
    assert ingest_pdf(store, "a.pdf", data)["status"] == "duplicate"
    other = make_pdf(["Different content."])
    assert ingest_pdf(store, "a.pdf", other)["name"] == "a (2).pdf"
