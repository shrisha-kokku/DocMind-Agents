# EXPLAIN.md - understand every file (interview guide)

## 1. The big picture

```
PDFs --> pdf_loader --> chunker --> embeddings --> FAISS index        (core/, runs once per upload)

User message --> PLANNER agent --(LLM picks a tool)--> RAG agent ------> search_documents tool --> FAISS
                                                   \-> Summarizer agent -> summarize_documents tool (map-reduce)
```
No keyword routing in code: the planner LLM chooses the tool from the *meaning* of the message.

## 2. OpenAI Agents SDK crash course (you have never used it - read this first)

| Concept | One-line meaning | Where used |
|---|---|---|
| **Agent** | An LLM + instructions (system prompt) + tools. | every `build_*_agent()` |
| **Runner** | Runs the agent *loop*: LLM -> tool call -> tool result -> LLM ... until final answer. `await Runner.run(agent, input)`. | `planner_agent.ask`, `summarizer_agent._ask_llm` |
| **function_tool** | Decorator that turns a Python function into a tool. Name = tool name, docstring = description, type hints = argument schema. | `search_documents`, `summarize_documents` |
| **Agent as tool** (`agent.as_tool`) | Wraps a whole agent so another agent can call it like a function. The caller gets back only the final text. | planner uses RAG + summarizer |
| **Context** (`RunContextWrapper`) | Your own Python object passed to `Runner.run(context=...)`; tools read it. Never sent to the LLM. | `AppContext` (vector store + progress callback) |
| **Dynamic instructions** | `instructions=` can be a function, called every run, so the prompt can contain live data. | planner/RAG list uploaded file names |
| **Model provider** | SDK defaults to OpenAI. For Groq we give `OpenAIChatCompletionsModel(openai_client=AsyncOpenAI(base_url=...))`. | `core/llm.py` |
| **ModelSettings** | Temperature etc. per agent. | low temp for factual agents |
| **max_turns** | Safety cap on loop iterations. | `Runner.run(..., max_turns=8)` |
| **RunResult** | `.final_output` = answer, `.new_items` = everything that happened (tool calls...). | routing caption in UI |
| **Tracing** | SDK uploads traces to OpenAI by default; we disable it (no OpenAI key). | `set_tracing_disabled(True)` |

**Handoffs vs agents-as-tools** (likely question): with a *handoff* the new agent takes over the conversation; with *agents-as-tools* the planner stays in control, calls specialists, and composes the final reply. The assignment says "agents instantiated as tools" and "planner decides which tool", so `as_tool` is the exact fit.

**What the LLM actually sees (agent as tool):** the planner gets two functions, `document_qa(input: str)` and `document_summary(input: str)`. It writes a text request, the SDK runs the sub-agent with that text and returns the sub-agent's final answer as the tool result.

## 3. File by file

### config.py
All settings from environment variables (`os.getenv`) with defaults. Why: change model/key without touching code; same code works locally (`.env`) and on Streamlit Cloud (secrets).

### core/pdf_loader.py
- `extract_pages(data: bytes)` - opens the PDF from memory with PyMuPDF, loops over pages, `get_text("text")`, cleans whitespace, skips empty pages, returns `[Page(number, text)]`. Raises if password-protected. Why PyMuPDF: fastest option, page-by-page, handles 10 MB+ files. Limitation: scanned PDFs have no text (needs OCR).

### core/chunker.py
- `_split` - recursive splitter: try paragraph, then line, then sentence, then word boundaries; hard cut only as last resort.
- `split_text` - merges small pieces up to `CHUNK_SIZE` (800 chars) and starts each new chunk with the last 150 chars of the previous one (**overlap**, so an idea cut at a boundary still appears whole in one chunk).
- `chunk_pages` - chunks page by page so every `Chunk` keeps `source` + `page` for citations.
- Why 800/150: small enough for precise retrieval, large enough to carry a full idea.

### core/vector_store.py
- `FastEmbedder` - local ONNX transformer (`bge-small-en-v1.5`), lazy-loaded. `embed_query` uses BGE's query prefix. Vectors are normalised.
- `VectorStore` - one FAISS `IndexFlatIP` for all PDFs. Normalised vectors + inner product = cosine similarity. `self.chunks[i]` is the metadata for index row `i`.
- `add_document` embeds in batches of 64 and reports progress (important for big PDFs).
- `search(query, k, filename)` - embeds the query, returns top-k `Hit(chunk, score)`; optional per-file filter.
- `has_hash`, `unique_name` - skip duplicate uploads, never overwrite same-named different files.
- Why FAISS flat (exact) search: simple, exact, fast enough for hundreds of thousands of chunks. Why not Chroma: needs no persistence here, fewer moving parts.

### core/ingest.py
`ingest_pdf` = the full pipeline (hash -> extract -> chunk -> embed -> index). `file_hash` (SHA-1 of content) detects re-uploads, because Streamlit re-runs the script on every click.

### core/llm.py
`get_model()` builds the Groq-backed model object. Not cached because each Streamlit question runs in a new asyncio event loop, and an async HTTP client must not outlive its loop. `max_retries` handles Groq's 429 rate limits automatically.

### pdf_agents/context.py
`AppContext(store, progress)`. Shared state without globals. `progress` lets tools tell the UI what they are doing. (Folder is `pdf_agents`, not `agents`, which would clash with the SDK package.)

### pdf_agents/rag_agent.py
- `search_documents` tool: embeds the query, gets top-5, drops hits below `MIN_SCORE` (**fallback**: returns `NO_RELEVANT_PASSAGES_FOUND` so the LLM says "not in the documents" instead of inventing), formats passages as `[n] file (page p) | relevance`.
- `RAG_INSTRUCTIONS`: always search first, retry once with new wording, answer only from passages, cite, short answer then bullets (**clarity of output**).
- `build_rag_agent`: temperature 0.1 for factual answers.
- **Context management**: only top-k passages (~1k tokens) enter the prompt, never the whole PDF.

### pdf_agents/summarizer_agent.py
- `_make_groups` joins chunks into ~10k-char blocks. `_evenly_sample` - for huge PDFs keeps max 10 blocks spread evenly (start to end) and tells the user it did so. Trade-off: bounded cost and rate-limit friendly vs. not reading every sentence.
- `_summarize_one`: **MAP** - each block summarised by a tool-less helper agent (150 words). **REDUCE** - partial summaries merged in batches of 5, repeated until one remains (hierarchical, so it scales). Merger prompt removes repetition and keeps order = coherence.
- `summarize_documents` tool: one named file or all files (multi-PDF). Unknown file -> fallback message listing real names.
- The agent then presents the result (overview + bullets).

### pdf_agents/planner_agent.py
- `build_planner_agent`: wraps both agents with `as_tool`. `strict_json_schema=False` = wider provider compatibility.
- Instructions: pick by meaning; make the request self-contained (sub-agents cannot see chat history); greetings answered directly; no docs -> ask for upload.
- `ask()`: builds messages = last 10 turns + new question, `Runner.run(..., context=ctx)`, then reads `result.new_items` to know which tool was chosen.

### app.py
Streamlit only. `st.session_state` keeps each user's store/chat. Sidebar: uploader + progress bar (`st.status`, `st.progress`). Chat: `st.chat_message`, `st.chat_input`. While agents run, `st.status` shows live steps because tools call `ctx.progress`. `asyncio.run` bridges Streamlit (sync) to the async SDK. History stores plain text only (cheap, avoids huge tool outputs).

### tests/test_pipeline.py
Offline tests with a fake embedder: chunk size/overlap, PDF extraction, search + file filter, duplicate/rename handling.

## 4. Large and multiple PDFs
- Multiple: one shared index; every chunk tagged with file + page; filter by file; summaries per file.
- Large (10 MB+): streamed page by page; embeddings in batches with a progress bar; upload limit raised to 200 MB in `.streamlit/config.toml`; summary capped by sampling.

## 5. Likely interview questions
1. **Why RAG instead of sending the PDF to the LLM?** Context limits, cost, and grounding; only relevant chunks are sent.
2. **Why chunk overlap?** Avoid losing meaning at chunk boundaries.
3. **Why cosine / normalised vectors?** Compares meaning not length; inner product on normalised vectors = cosine.
4. **How does the planner decide?** LLM reads tool descriptions + instructions and emits a tool call; no hard-coded rules.
5. **Why `as_tool` not handoff?** Planner keeps control and composes the reply; fits "agents as tools".
6. **What is context in the SDK?** Local object passed to tools, not seen by the LLM.
7. **How do you run Groq with the SDK?** `AsyncOpenAI(base_url)` + `OpenAIChatCompletionsModel`; disable tracing.
8. **How do you reduce hallucination?** Low temperature, "answer only from passages", score threshold, citations, explicit "not found".
9. **Map-reduce vs refine?** Map-reduce is parallelisable/simple but may lose cross-section links; refine is sequential and keeps flow but slower and error-accumulating.
10. **What if the PDF is huge?** Batching, sampling cap, hierarchical reduce.
11. **Rate limits?** `max_retries` backoff; smaller model via `LLM_MODEL`; group size is configurable.
12. **Why FAISS?** Exact, in-memory, no server. For millions of chunks: IVF/HNSW or a managed vector DB.
13. **How is multi-user handled?** Per-session store in `st.session_state`; nothing shared between users.
14. **How to extend?** Add a new agent file + one `as_tool` line in the planner (e.g. table-extraction agent).
15. **Limitations / improvements?** No OCR, in-memory index lost on restart, sampling in huge summaries, add re-ranking, hybrid BM25 search, streaming answers, evaluation set.
