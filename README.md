# 📚 DocMind Agents

Chat with multiple PDFs. A **Planner agent** (OpenAI Agents SDK) reads each message and decides on its own:
question → **RAG agent**, summary request → **Summarization agent**. No mode buttons.

## Run locally
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # Mac/Linux: cp .env.example .env   -> then paste your Groq key in .env
streamlit run app.py
pytest -q                       # offline tests (no key needed)
```
Free key: https://console.groq.com/keys . Change model any time with `LLM_MODEL` in `.env`.
(First run downloads the small embedding model, ~130 MB.)

## Deploy (Streamlit Community Cloud, free)
1. Push to GitHub (below). 2. share.streamlit.io -> New app -> pick repo, main file `app.py`.
3. Advanced settings -> Python 3.11 or 3.12 -> Secrets:
```toml
LLM_API_KEY = "your_groq_key"
LLM_MODEL = "llama-3.3-70b-versatile"
```

## Git: push step by step (history shows the build order)
```bash
git init && git branch -M main
git add config.py core/ requirements.txt .gitignore .env.example && git commit -m "Add config and PDF ingestion pipeline (loader, chunker, FAISS store)"
git add pdf_agents/ && git commit -m "Add RAG, summarization and planner agents (OpenAI Agents SDK)"
git add app.py .streamlit/ && git commit -m "Add Streamlit chat UI"
git add tests/ README.md EXPLAIN.md && git commit -m "Add tests and docs"
git remote add origin https://github.com/<you>/docmind-agents.git
git push -u origin main
```
Never commit `.env` (it is in `.gitignore`).

Read **EXPLAIN.md** to understand every file.
