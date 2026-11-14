"""
app.py - Streamlit UI ONLY. All AI logic lives in core/ and pdf_agents/.

What happens here:
  1. Sidebar: upload PDFs -> ingest (extract, chunk, embed, index) with a progress bar.
  2. Main: chat. Each message goes to the Planner agent, which picks RAG or Summarization.
  3. A live status box shows what the agents are doing while they work.
"""
import asyncio
import os

import streamlit as st

st.set_page_config(page_title="DocMind Agents", page_icon="📚", layout="wide")


def _load_streamlit_secrets() -> None:
    """On Streamlit Cloud, keys live in st.secrets. Copy them to env vars so config.py sees them."""
    try:
        for key in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "EMBEDDING_MODEL"):
            if key in st.secrets:
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass  # no secrets file locally - fine, .env is used instead


_load_streamlit_secrets()  # must run BEFORE `import config`

import config  # noqa: E402
from core.ingest import file_hash, ingest_pdf  # noqa: E402
from core.vector_store import FastEmbedder, VectorStore  # noqa: E402
from pdf_agents.context import AppContext  # noqa: E402
from pdf_agents.planner_agent import QA_TOOL, SUMMARY_TOOL, ask  # noqa: E402

ROUTE_LABELS = {QA_TOOL: "🔎 RAG agent", SUMMARY_TOOL: "📝 Summarization agent"}

# ---------- light styling ----------
st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.hero {font-size: 2.1rem; font-weight: 800; background: linear-gradient(90deg,#6366f1,#ec4899);
       -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.sub {color: #6b7280; margin-bottom: 1rem;}
[data-testid="stChatMessage"] {border-radius: 14px; padding: 0.8rem 1rem; margin-bottom: 0.5rem;
       border: 1px solid rgba(128,128,128,0.2);}
.doc-card {border: 1px solid rgba(128,128,128,0.3); border-radius: 10px; padding: 0.5rem 0.7rem;
       margin-bottom: 0.4rem; font-size: 0.85rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_embedder() -> FastEmbedder:
    """Create the embedding model wrapper once per server (the model itself loads lazily)."""
    return FastEmbedder(config.EMBEDDING_MODEL)


# ---------- per-user session state ----------
if "store" not in st.session_state:
    st.session_state.store = VectorStore(get_embedder())
    st.session_state.messages = []   # what is shown in the chat
    st.session_state.history = []    # plain user/assistant text sent to the planner
store: VectorStore = st.session_state.store

# ---------- header ----------
st.markdown('<div class="hero">📚 DocMind Agents</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Chat with your PDFs. Ask a Question or request a Summary</div>', unsafe_allow_html=True)

if not config.LLM_API_KEY:
    st.error("LLM_API_KEY is not set. Add it to your .env file (local) or Streamlit secrets (cloud).")
    st.stop()

# ---------- sidebar: upload + indexing ----------
with st.sidebar:
    st.header("📂 Documents")
    uploads = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)

    for upload in uploads or []:
        data = upload.getvalue()
        if store.has_hash(file_hash(data)):
            continue  # already indexed (Streamlit re-runs the script on every interaction)
        with st.status(f"Indexing {upload.name}", expanded=True) as status:
            bar = st.progress(0.0)

            def on_progress(fraction: float, message: str) -> None:
                bar.progress(min(fraction, 1.0), text=message)

            try:
                info = ingest_pdf(store, upload.name, data, progress=on_progress)
                status.update(label=f"Indexed {info['name']} ({info['pages']} pages, {info['chunks']} chunks)",
                              state="complete", expanded=False)
            except ValueError as error:
                status.update(label=f"Skipped {upload.name}: {error}", state="error")

    if store.docs:
        st.subheader("Indexed")
        for name, doc in store.docs.items():
            st.markdown(f'<div class="doc-card">📄 <b>{name}</b><br>{doc["pages"]} pages · '
                        f'{len(doc["chunk_ids"])} chunks</div>', unsafe_allow_html=True)
        if st.button("🗑️ Reset documents & chat", use_container_width=True):
            for key in ("store", "messages", "history"):
                st.session_state.pop(key, None)
            st.rerun()
    st.caption(f"Model: `{config.LLM_MODEL}`")

# ---------- chat history ----------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("route"):
            st.caption(f"Routed to: {message['route']}")

if not st.session_state.messages:
    st.info("Upload PDFs in the sidebar, then try: **“Summarize the document”** or "
            "**“What are the main requirements?”**")

# ---------- new message ----------
if prompt := st.chat_input("Ask a question or request a summary..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        route = ""
        with st.status("Agents are working...", expanded=True) as status:
            status.write("Planner is reading your request...")
            # Tools call ctx.progress(...) -> shows up live inside this status box
            context = AppContext(store=store, progress=status.write)
            try:
                # Streamlit scripts are synchronous; asyncio.run drives the async Agents SDK.
                answer, tools_used = asyncio.run(ask(prompt, st.session_state.history, context))
                route = ", ".join(ROUTE_LABELS.get(t, t) for t in tools_used) or "Planner (answered directly)"
                status.update(label="Done", state="complete", expanded=False)
            except Exception as error:  # API errors, rate limits, max turns ...
                answer = f"⚠️ Something went wrong: `{error}`\n\nTry again, or switch LLM_MODEL in your settings."
                status.update(label="Failed", state="error")
        st.markdown(answer)
        if route:
            st.caption(f"Routed to: {route}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "route": route})
    # keep only plain text turns for the planner (cheap, avoids huge tool outputs in the prompt)
    st.session_state.history += [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]
    st.session_state.history = st.session_state.history[-config.MAX_HISTORY_MESSAGES:]
