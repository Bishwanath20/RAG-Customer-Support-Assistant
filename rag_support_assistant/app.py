#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════
# app.py — RAG Customer Support Assistant (Streamlit Web UI)
#
# Run locally:  streamlit run app.py
# Deploy:       push to GitHub → share.streamlit.io → deploy
# ═══════════════════════════════════════════════════════════════

import os
import json
import time

import streamlit as st
from langchain_groq import ChatGroq

from src.config import (
    GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    CHROMA_DIR, PDF_PATH, ESCALATION_LOG, CONFIDENCE_THRESHOLD,
    DATA_DIR, LOGS_DIR
)
from src.kb_builder import create_knowledge_base
from src.ingestion  import build_index, load_index
from src.graph      import build_graph, run_query


# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Helix — Support Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────
# Direction: "Resolution Desk" — a support system feels like a control
# surface, not a chat toy. Ink-and-signal palette: deep slate ground,
# warm paper panels, a single confident teal as the "resolved" signal
# and amber as the "needs a human" signal. Type: Fraunces (display,
# editorial confidence) + Inter (UI clarity) + JetBrains Mono (data).
# The signature: a live "decision trace" rail that shows the query
# moving through the pipeline — the system reasons in the open.

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --ground: #14171C;
    --ground-soft: #1B1F26;
    --panel: #FBF8F1;
    --panel-line: #E6DFD1;
    --ink: #14171C;
    --ink-soft: #5A5F68;
    --signal: #0E8C7F;          /* resolved / answered */
    --signal-soft: #D6EBE7;
    --alert: #C7763A;           /* escalated to human */
    --alert-soft: #F2E2D2;
    --hair: rgba(255,255,255,0.08);
}

/* ── reset streamlit chrome ── */
#MainMenu, header, footer {visibility: hidden;}
.stDeployButton {display: none;}
.block-container {padding: 0 !important; max-width: 100% !important;}
section.main > div {padding: 0 !important;}
.stApp {background: var(--ground);}

/* ── hero ── */
.hero {
    background:
        radial-gradient(120% 140% at 12% -10%, #232831 0%, var(--ground) 55%);
    padding: 5.5rem 8vw 4rem;
    border-bottom: 1px solid var(--hair);
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    right: -8%; top: -30%;
    width: 480px; height: 480px;
    background: radial-gradient(circle, rgba(14,140,127,0.16) 0%, transparent 65%);
    filter: blur(8px);
}
.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: var(--signal);
    margin-bottom: 1.4rem;
    display: flex; align-items: center; gap: 0.7rem;
}
.eyebrow::before {
    content: ""; width: 28px; height: 1px;
    background: var(--signal); display: inline-block;
}
.hero h1 {
    font-family: 'Fraunces', serif;
    font-size: clamp(2.6rem, 5.5vw, 4.6rem);
    line-height: 1.02;
    font-weight: 600;
    color: #F7F4EC;
    margin: 0 0 1.3rem;
    letter-spacing: -0.02em;
    max-width: 16ch;
}
.hero h1 em {
    font-style: italic;
    color: var(--signal);
    font-weight: 500;
}
.hero p {
    font-family: 'Inter', sans-serif;
    font-size: 1.08rem;
    line-height: 1.65;
    color: #A8AEB8;
    max-width: 54ch;
    margin: 0;
}
.hero-meta {
    display: flex; gap: 2.8rem; margin-top: 3rem;
    flex-wrap: wrap;
}
.hero-meta .stat {
    font-family: 'Inter', sans-serif;
}
.hero-meta .stat .num {
    font-family: 'Fraunces', serif;
    font-size: 1.9rem; font-weight: 600;
    color: #F7F4EC; display: block; line-height: 1;
}
.hero-meta .stat .lbl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.66rem; letter-spacing: 0.16em;
    text-transform: uppercase; color: var(--ink-soft);
    margin-top: 0.5rem; display: block;
}

/* ── work surface ── */
.surface {
    background: var(--panel);
    padding: 3.5rem 8vw 5rem;
    min-height: 50vh;
}
.surface-head {
    display: flex; align-items: baseline;
    justify-content: space-between;
    border-bottom: 2px solid var(--ink);
    padding-bottom: 0.9rem; margin-bottom: 2.2rem;
}
.surface-head h2 {
    font-family: 'Fraunces', serif;
    font-size: 1.5rem; font-weight: 600;
    color: var(--ink); margin: 0;
}
.surface-head .ctx {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; letter-spacing: 0.14em;
    color: var(--ink-soft); text-transform: uppercase;
}

/* ── input ── */
.stTextInput > div > div > input {
    font-family: 'Inter', sans-serif !important;
    font-size: 1.05rem !important;
    background: #FFFFFF !important;
    border: 1px solid var(--panel-line) !important;
    border-radius: 0 !important;
    padding: 1.1rem 1.3rem !important;
    color: var(--ink) !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--signal) !important;
    box-shadow: 0 0 0 3px var(--signal-soft) !important;
}
.stButton > button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    background: var(--ink) !important;
    color: var(--panel) !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 1.05rem 1.8rem !important;
    transition: background 0.15s ease !important;
}
.stButton > button:hover {
    background: var(--signal) !important;
    color: #FFFFFF !important;
}

/* ── result card ── */
.result {
    background: #FFFFFF;
    border: 1px solid var(--panel-line);
    border-left: 4px solid var(--signal);
    padding: 2rem 2.2rem;
    margin-top: 1.8rem;
}
.result.escalated { border-left-color: var(--alert); }
.result .verdict {
    display: flex; align-items: center; gap: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem; letter-spacing: 0.18em;
    text-transform: uppercase; margin-bottom: 1.3rem;
}
.result .verdict .dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--signal);
}
.result.escalated .verdict .dot { background: var(--alert); }
.result .verdict .tag { color: var(--signal); font-weight: 500; }
.result.escalated .verdict .tag { color: var(--alert); }
.result .answer {
    font-family: 'Inter', sans-serif;
    font-size: 1.12rem; line-height: 1.7;
    color: var(--ink);
}

/* ── trace rail (signature element) ── */
.trace {
    margin-top: 1.6rem; padding-top: 1.5rem;
    border-top: 1px dashed var(--panel-line);
}
.trace-row {
    display: flex; align-items: center; gap: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem; color: var(--ink-soft);
    padding: 0.4rem 0;
}
.trace-row .node {
    color: var(--ink); font-weight: 500;
    min-width: 168px;
}
.trace-row .val { color: var(--signal); }
.trace-row .bar {
    flex: 1; height: 1px; background: var(--panel-line);
}

/* ── chips ── */
.chips { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 1rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# SYSTEM INITIALIZATION (cached — runs once)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_system():
    """Initialize knowledge base, ChromaDB index, LLM, and graph once."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    if not os.path.exists(PDF_PATH):
        create_knowledge_base(PDF_PATH)

    if os.path.exists(CHROMA_DIR):
        try:
            collection = load_index(CHROMA_DIR)
        except Exception:
            collection = build_index(PDF_PATH, CHROMA_DIR)
    else:
        collection = build_index(PDF_PATH, CHROMA_DIR)

    llm = ChatGroq(
        api_key=GROQ_API_KEY, model=GROQ_MODEL,
        temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS
    )
    graph = build_graph(collection, llm)
    return collection, graph


# ─────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="eyebrow">Retrieval-Augmented Support Engine</div>
    <h1>Every question gets a <em>grounded</em> answer — or a human.</h1>
    <p>Helix reads your knowledge base, answers from verified source material,
    and knows when to step back. No invented policies, no dead ends —
    uncertain cases route straight to a person.</p>
    <div class="hero-meta">
        <div class="stat"><span class="num">&lt;500ms</span><span class="lbl">Response time</span></div>
        <div class="stat"><span class="num">100%</span><span class="lbl">Source-grounded</span></div>
        <div class="stat"><span class="num">Zero</span><span class="lbl">Hallucination policy</span></div>
        <div class="stat"><span class="num">Auto</span><span class="lbl">Human escalation</span></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# WORK SURFACE
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="surface">', unsafe_allow_html=True)
st.markdown("""
<div class="surface-head">
    <h2>Ask the desk</h2>
    <span class="ctx">Knowledge base · 8 sections · returns, shipping, payments, warranty</span>
</div>
""", unsafe_allow_html=True)

# API key guard
if not GROQ_API_KEY or GROQ_API_KEY.startswith("gsk_PASTE"):
    st.error("Set GROQ_API_KEY in your environment or Streamlit secrets to run the assistant. "
             "Get a free key at console.groq.com.")
    st.stop()

try:
    collection, graph = init_system()
except Exception as e:
    st.error(f"System initialization failed: {e}")
    st.stop()

# Query input
col_in, col_btn = st.columns([5, 1])
with col_in:
    query = st.text_input(
        "query", placeholder="e.g.  How long do refunds take to process?",
        label_visibility="collapsed",
        key="query_input"
    )
with col_btn:
    submitted = st.button("Resolve", use_container_width=True)

# Example chips
st.markdown("""
<div class="chips">
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
    color:#8A8F98;letter-spacing:0.1em;">TRY:</span>
</div>
""", unsafe_allow_html=True)

examples = [
    "What is the return policy?",
    "How do I reset my password?",
    "I want to speak to a manager",
    "Who won the World Cup?",
]
ex_cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    with ex_cols[i]:
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            query = ex
            submitted = True


# ─────────────────────────────────────────────────────────────────
# RESULT
# ─────────────────────────────────────────────────────────────────
if submitted and query:
    t0 = time.time()
    with st.spinner(""):
        state = run_query(graph, query)
    elapsed = (time.time() - t0) * 1000

    is_answer = state["route"] == "answer"
    conf = state["confidence_score"]

    card_class = "result" if is_answer else "result escalated"
    verdict_tag = "Answered from source" if is_answer else "Routed to human agent"
    body = state["answer"] if is_answer else state["human_response"]

    # Decision trace (signature element)
    kw = "yes" if state["requires_escalation"] else "no"
    docs_n = len(state["retrieved_docs"])
    reason = state["escalation_reason"] or "high confidence"

    st.markdown(f"""
    <div class="{card_class}">
        <div class="verdict">
            <span class="dot"></span>
            <span class="tag">{verdict_tag}</span>
            <span style="color:#A8AEB8;">·  confidence {conf:.2f}  ·  {elapsed:.0f}ms</span>
        </div>
        <div class="answer">{body}</div>
        <div class="trace">
            <div class="trace-row"><span class="node">01 · query_processor</span>
                <span class="bar"></span><span class="val">keyword: {kw}</span></div>
            <div class="trace-row"><span class="node">02 · retriever</span>
                <span class="bar"></span><span class="val">{docs_n} chunks · conf {conf:.2f}</span></div>
            <div class="trace-row"><span class="node">03 · router</span>
                <span class="bar"></span><span class="val">{reason}</span></div>
            <div class="trace-row"><span class="node">04 · {'answer_generator' if is_answer else 'hitl_escalator'}</span>
                <span class="bar"></span><span class="val">{'groq llama-3' if is_answer else 'logged + notified'}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
