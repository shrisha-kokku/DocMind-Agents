"""
planner_agent.py - the "brain" that routes every message to the right agent.

Agents SDK concept: AGENTS AS TOOLS. `agent.as_tool(...)` wraps a whole agent
so that another agent can call it exactly like a function. The planner reads
the user's message, and the LLM itself decides which tool (= which agent) to
call - there is no if/else keyword routing in our code.
"""
from agents import Agent, ModelSettings, RunContextWrapper, Runner
from agents.items import ToolCallItem

import config
from core.llm import get_model
from pdf_agents.context import AppContext
from pdf_agents.rag_agent import build_rag_agent
from pdf_agents.summarizer_agent import build_summarizer_agent

QA_TOOL = "document_qa"
SUMMARY_TOOL = "document_summary"

PLANNER_INSTRUCTIONS = f"""You are the planner of a PDF assistant. Decide how to handle each user message.

Tools:
- {QA_TOOL}: for any question about the content of the documents (facts, details, explanations, comparisons, "what does it say about...").
- {SUMMARY_TOOL}: when the user wants a summary, overview, gist, TL;DR or main points of a document.

Rules:
1. Choose the tool from the MEANING of the message, not from keywords.
2. When you call a tool, write a complete, self-contained request. The tool cannot see the chat history, so
   resolve words like "it" or "that" using earlier messages. Mention the file name if the user named one.
3. Answer directly (no tool) ONLY for greetings or questions about what you can do. For ANY other question, even general knowledge ones, call document_qa. Never answer from your own knowledge.
4. If no documents are uploaded, ask the user to upload PDFs first (no tool call).
5. After a tool returns, give its answer to the user as is. Keep the citations and structure. Do not add new facts."""


def _instructions(ctx: RunContextWrapper[AppContext], agent: Agent[AppContext]) -> str:
    names = ctx.context.store.document_names()
    return PLANNER_INSTRUCTIONS + "\n\nUploaded files: " + (", ".join(names) or "none")


def build_planner_agent() -> Agent[AppContext]:
    rag_agent = build_rag_agent()
    summarizer_agent = build_summarizer_agent()

    qa_tool = rag_agent.as_tool(
        tool_name=QA_TOOL,
        tool_description="Answer a specific question using the content of the uploaded PDFs (retrieval-augmented).",
    )
    summary_tool = summarizer_agent.as_tool(
        tool_name=SUMMARY_TOOL,
        tool_description="Summarise one or all uploaded PDFs (map-reduce, works for very long documents).",
    )
    # Non-strict schema = widest compatibility with Groq/OpenRouter models.
    qa_tool.strict_json_schema = False
    summary_tool.strict_json_schema = False

    return Agent[AppContext](
        name="Planner",
        instructions=_instructions,
        model=get_model(),
        model_settings=ModelSettings(temperature=0.1),
        tools=[qa_tool, summary_tool],
        tool_use_behavior="stop_on_first_tool",
    )


async def ask(question: str, history: list[dict], ctx: AppContext) -> tuple[str, list[str]]:
    """Run the planner for one user message.

    history: earlier turns as [{"role": "user"/"assistant", "content": "..."}]
    Returns: (answer text, names of the tools/agents the planner chose)
    """
    planner = build_planner_agent()
    messages = history[-config.MAX_HISTORY_MESSAGES:] + [{"role": "user", "content": question}]

    # Agents SDK: Runner.run loops  LLM -> tool call -> tool result -> LLM ...  until a final answer.
    # max_turns is a safety limit on that loop.
    result = await Runner.run(planner, input=messages, context=ctx, max_turns=8)

    # result.new_items lists everything that happened in this run (messages, tool calls, outputs).
    # We keep the tool calls so the UI can show which agent the planner chose.
    tools_used = [getattr(item.raw_item, "name", "") for item in result.new_items if isinstance(item, ToolCallItem)]
    return str(result.final_output), tools_used
