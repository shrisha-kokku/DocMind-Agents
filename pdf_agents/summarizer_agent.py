"""
summarizer_agent.py - the Summarization agent (map-reduce for long documents).

Why map-reduce: a big PDF does not fit in one LLM prompt.
  MAP    : split the document into groups of chunks, summarise each group separately.
  REDUCE : merge the partial summaries (in batches, repeatedly) into one final summary.
"""
import numpy as np
from agents import Agent, ModelSettings, RunContextWrapper, Runner, function_tool

import config
from core.llm import get_model
from pdf_agents.context import AppContext


# ---------------- helpers (plain Python, no SDK) ----------------
def _make_groups(chunks) -> list[str]:
    """Join consecutive chunks into text blocks of ~SUMMARY_GROUP_CHARS characters."""
    groups, current, size = [], [], 0
    for chunk in chunks:
        if current and size + len(chunk.text) > config.SUMMARY_GROUP_CHARS:
            groups.append("\n".join(current))
            current, size = [], 0
        current.append(chunk.text)
        size += len(chunk.text)
    if current:
        groups.append("\n".join(current))
    return groups


def _evenly_sample(groups: list[str], limit: int) -> list[str]:
    """For huge documents keep `limit` groups spread evenly from start to end."""
    if len(groups) <= limit:
        return groups
    positions = np.linspace(0, len(groups) - 1, limit).round().astype(int)
    return [groups[i] for i in sorted(set(positions))]


async def _ask_llm(agent: Agent, prompt: str) -> str:
    # Agents SDK concept: Runner.run(agent, input) runs the agent loop until it
    # produces a final answer, and returns a RunResult. `.final_output` is the text.
    result = await Runner.run(agent, prompt)
    return str(result.final_output).strip()


async def _summarize_one(name: str, ctx: AppContext) -> str:
    """Map-reduce summary of a single document."""
    all_groups = _make_groups(ctx.store.get_document_chunks(name))
    groups = _evenly_sample(all_groups, config.SUMMARY_MAX_GROUPS)

    # Two tiny helper agents with NO tools - they only transform text.
    model = get_model()
    mapper = Agent(name="SectionSummarizer", model=model, model_settings=ModelSettings(temperature=0.2),
                   instructions="You summarise document excerpts faithfully and concisely. Never add facts that are not in the excerpt.")
    merger = Agent(name="SummaryMerger", model=model, model_settings=ModelSettings(temperature=0.2),
                            instructions="You merge partial summaries into one coherent summary. Format: one short overview sentence, then the key points as bullet points. Remove repetition, keep the original order of ideas, keep important facts and numbers. Never add new facts.")
    # ---- MAP ----
    partials = []
    for i, group in enumerate(groups, start=1):
        ctx.progress(f"Summarising '{name}': part {i}/{len(groups)}")
        partials.append(await _ask_llm(
            mapper,
            f"Summarise this excerpt (part {i} of {len(groups)}) of the document '{name}' in at most 150 words. "
            f"Keep key points, facts and figures.\n\n{group}"))

    # ---- REDUCE (repeat until one summary is left) ----
    while len(partials) > 1:
        ctx.progress(f"Merging {len(partials)} partial summaries of '{name}'")
        batch_size = config.SUMMARY_REDUCE_BATCH
        merged = []
        for start in range(0, len(partials), batch_size):
            batch = partials[start:start + batch_size]
            is_last_round = len(partials) <= batch_size
            words = 350 if is_last_round else 200
            merged.append(await _ask_llm(
                merger,
                f"Merge these partial summaries (in document order) into ONE coherent summary of at most {words} words:\n\n"
                + "\n\n".join(f"- {p}" for p in batch)))
        partials = merged

    summary = partials[0]
    if len(groups) < len(all_groups):
        summary += (f"\n\n_(Large document: summarised from {len(groups)} of {len(all_groups)} "
                    f"sections spread evenly across it.)_")
    return summary


# ---------------- the tool ----------------
@function_tool(strict_mode=False)  # non-strict schema = widest compatibility with Groq/OpenRouter models
async def summarize_documents(ctx: RunContextWrapper[AppContext], filename: str = "") -> str:
    """Create a summary of the uploaded PDF documents.

    Args:
        filename: Optional. The one file to summarise. Use an empty string to summarise every uploaded file.
    """
    store = ctx.context.store
    if not store.document_names():
        return "No documents have been uploaded yet."

    if filename:
        target = store.find_document(filename)
        if target is None:  # fallback: unknown file name
            return f"No uploaded file matches '{filename}'. Available files: {', '.join(store.document_names())}"
        names = [target]
    else:
        names = store.document_names()

    sections = []
    for name in names:
        sections.append(f"### {name}\n{await _summarize_one(name, ctx.context)}")
    return "\n\n".join(sections)


SUMMARIZER_INSTRUCTIONS = """You produce summaries of the user's uploaded PDF documents.

Rules:
1. Call summarize_documents. Pass `filename` only if the user names a specific file; otherwise pass an empty string.
2. Present the result as a clean, coherent summary: one short overview paragraph, then the key points as bullets.
3. If there are several documents, give one section per document, then a brief combined takeaway.
4. Use ONLY the tool result. Do not add facts. Avoid repetition."""


def build_summarizer_agent() -> Agent[AppContext]:
    return Agent[AppContext](
        name="SummarizationAgent",
        instructions=SUMMARIZER_INSTRUCTIONS,
        model=get_model(),
        model_settings=ModelSettings(temperature=0.2),
        tools=[summarize_documents],
        tool_use_behavior="stop_on_first_tool",
    )