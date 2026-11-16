"""
rag_agent.py - the Retrieval-Augmented Generation agent.

Flow: question -> (LLM decides to call) search_documents tool -> top-k chunks
come back -> LLM writes an answer grounded ONLY in those chunks, with citations.
"""
from agents import Agent, ModelSettings, RunContextWrapper, function_tool

import config
from core.llm import get_model
from pdf_agents.context import AppContext


@function_tool(strict_mode=False)  # non-strict schema = widest compatibility with Groq/OpenRouter models
async def search_documents(ctx: RunContextWrapper[AppContext], query: str, filename: str = "") -> str:
    """Search the uploaded PDF documents for passages relevant to a query.

    Args:
        query: What to look for. Use a clear, self-contained search phrase.
        filename: Optional. Restrict the search to one uploaded file. Use an empty string to search all files.
    """
    store = ctx.context.store
    ctx.context.progress(f"Searching documents for: {query}")

    target = None
    if filename:
        target = store.find_document(filename)
        if target is None:  # fallback: unknown file name
            return f"No uploaded file matches '{filename}'. Available files: {', '.join(store.document_names())}"

    hits = store.search(query, k=config.TOP_K, filename=target)
    relevant = [h for h in hits if h.score >= config.MIN_SCORE]
    if not relevant:  # fallback: nothing relevant -> tell the LLM, so it does not invent an answer
        return "NO_RELEVANT_PASSAGES_FOUND"

    ctx.context.progress(f"Found {len(relevant)} relevant passages")
    return "\n\n".join(
        f"[{i}] {h.chunk.source} (page {h.chunk.page}) | relevance {h.score:.2f}\n{h.chunk.text}"
        for i, h in enumerate(relevant, start=1)
    )


RAG_INSTRUCTIONS = """You answer questions about the user's uploaded PDF documents.

Rules:
1. ALWAYS call search_documents first. If the first search is weak, try once more with different wording.
2. Answer ONLY from the returned passages. Never use outside knowledge or guess.
3. If the tool returns NO_RELEVANT_PASSAGES_FOUND, or the passages do not contain the answer, say clearly
   that the documents do not contain this information.
4. Cite sources inline like (file.pdf, p. 4).
5. Format: a short direct answer first, then supporting details as bullets if useful. Keep it clear and concise."""


def _instructions(ctx: RunContextWrapper[AppContext], agent: Agent[AppContext]) -> str:
    # Agents SDK concept: DYNAMIC instructions. Instead of a fixed string you can pass a
    # function; the SDK calls it on every run, so the prompt can include live data.
    names = ctx.context.store.document_names()
    return RAG_INSTRUCTIONS + "\n\nUploaded files: " + (", ".join(names) or "none")


def build_rag_agent() -> Agent[AppContext]:
    return Agent[AppContext](
        name="RAGAgent",
        instructions=_instructions,
        model=get_model(),
        model_settings=ModelSettings(temperature=0.1),  # low temperature = factual, less creative
        tools=[search_documents],
    )
