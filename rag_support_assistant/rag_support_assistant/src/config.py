# ═══════════════════════════════════════════════════════════════
# config.py — Central configuration for RAG Support Assistant
# All hyperparameters live here. Tune without touching logic.
# ═══════════════════════════════════════════════════════════════

import os
from dotenv import load_dotenv

load_dotenv()

# ── API ─────────────────────────────────────────────────────────
GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL         = "llama3-8b-8192"

# ── PATHS ───────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.dirname(__file__))
DATA_DIR           = os.path.join(BASE_DIR, "data")
LOGS_DIR           = os.path.join(BASE_DIR, "logs")
CHROMA_DIR         = os.path.join(BASE_DIR, "chroma_db")
PDF_PATH           = os.path.join(DATA_DIR, "knowledge_base.pdf")
ESCALATION_LOG     = os.path.join(LOGS_DIR, "escalations.log")

# ── RAG HYPERPARAMETERS ─────────────────────────────────────────
CHUNK_SIZE           = 1000   # ~150-200 words per chunk
CHUNK_OVERLAP        = 200    # 20% overlap prevents boundary gaps
TOP_K                = 3      # Top-k chunks retrieved per query

# ── ROUTING ─────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.35   # Below this → escalate to human

ESCALATION_KEYWORDS  = [
    "urgent", "complaint", "frustrated", "angry",
    "human", "agent", "escalate", "manager",
    "legal", "lawsuit", "unacceptable", "terrible"
]

# ── LLM ─────────────────────────────────────────────────────────
LLM_TEMPERATURE    = 0.1
LLM_MAX_TOKENS     = 512
