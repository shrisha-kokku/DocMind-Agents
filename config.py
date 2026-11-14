"""
config.py - every setting lives here, read from environment variables.

"""
import os
from dotenv import load_dotenv

load_dotenv()  

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "6"))  # auto-retry on 429 rate limits

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

CHUNK_SIZE = 800        # max characters per chunk
CHUNK_OVERLAP = 150     # characters shared between neighbouring chunks
TOP_K = 5               # passages returned per search
MIN_SCORE = 0.30        # below this cosine score we treat a passage as irrelevant

SUMMARY_GROUP_CHARS = 10000   # size of one "map" input
SUMMARY_MAX_GROUPS = 10       # cap on map calls per document (keeps big PDFs affordable)
SUMMARY_REDUCE_BATCH = 5      # how many partial summaries are merged per reduce call

MAX_HISTORY_MESSAGES = 10     # how many past messages the planner sees
