"""
context.py - the object shared by every agent and tool during one run.

Agents SDK concept: "context". You create ONE object, pass it to Runner.run(...,
context=obj), and every tool receives it via RunContextWrapper. It is NOT sent
to the LLM - it is just your own Python data (here: the vector store and a
progress callback). This avoids global variables.

"""
from dataclasses import dataclass
from typing import Callable

from core.vector_store import VectorStore


def _no_progress(message: str) -> None:
    pass


@dataclass
class AppContext:
    store: VectorStore                                  # all indexed PDFs
    progress: Callable[[str], None] = _no_progress      # UI hook: tools report what they are doing
