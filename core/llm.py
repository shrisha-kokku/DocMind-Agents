"""
llm.py - connects the OpenAI Agents SDK to Groq (or OpenRouter, etc.).

The Agents SDK talks to OpenAI by default. Groq/OpenRouter expose an
OpenAI-COMPATIBLE API, so we only need to (1) point an AsyncOpenAI client at
their base_url and (2) wrap it in OpenAIChatCompletionsModel, because those
providers support the Chat Completions API (not OpenAI's newer Responses API).
"""
from agents import OpenAIChatCompletionsModel, set_tracing_disabled
from openai import AsyncOpenAI

import config

# The SDK sends traces to OpenAI's servers by default. We have no OpenAI key,
# so switch that off (otherwise you get "401 tracing" warnings in the logs).
set_tracing_disabled(True)


def get_model() -> OpenAIChatCompletionsModel:
    """Build a model object that any Agent can use via Agent(model=...).

    Not cached on purpose: Streamlit runs each question in a fresh asyncio
    event loop, and an async HTTP client must not outlive its loop.
    """
    if not config.LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is missing. Add it to .env (local) or Streamlit secrets (cloud).")
    client = AsyncOpenAI(
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        max_retries=config.LLM_MAX_RETRIES,  # automatic backoff + retry on 429 rate limits
    )
    return OpenAIChatCompletionsModel(model=config.LLM_MODEL, openai_client=client)
