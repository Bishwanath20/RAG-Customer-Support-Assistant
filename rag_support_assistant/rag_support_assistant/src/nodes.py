# ═══════════════════════════════════════════════════════════════
# nodes.py — GraphState definition + all 5 LangGraph node functions
# Each node: receives GraphState → one job → returns updated state
# ═══════════════════════════════════════════════════════════════

import json
import datetime
import os
from typing import TypedDict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from src.config import (
    GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    CONFIDENCE_THRESHOLD, ESCALATION_KEYWORDS,
    ESCALATION_LOG, LOGS_DIR, PDF_PATH, TOP_K
)
from src.ingestion import retrieve, compute_confidence


# ── GraphState ──────────────────────────────────────────────────────
class GraphState(TypedDict):
    """
    Shared state TypedDict that flows through the entire LangGraph pipeline.

    Populated progressively by each node:
      query_processor   → cleaned_query, requires_escalation
      retriever_node    → retrieved_docs, confidence_score
      router            → route, escalation_reason
      answer_generator  → answer
      hitl_escalator    → human_response
    """
    query               : str
    cleaned_query       : str
    retrieved_docs      : List[Document]
    confidence_score    : float
    route               : str           # 'answer' | 'escalate'
    answer              : str
    escalation_reason   : str
    human_response      : str
    requires_escalation : bool
    error               : Optional[str]


# ── LLM singleton ───────────────────────────────────────────────────
def get_llm() -> ChatGroq:
    return ChatGroq(
        api_key     = GROQ_API_KEY,
        model       = GROQ_MODEL,
        temperature = LLM_TEMPERATURE,
        max_tokens  = LLM_MAX_TOKENS
    )


# ── NODE 1: Query Processor ─────────────────────────────────────────
def query_processor(state: GraphState) -> GraphState:
    """
    Entry node. Two responsibilities:
      1. Normalize raw query (strip + lowercase)
      2. Scan for escalation keywords → sets requires_escalation flag

    Keyword escalation is separate from confidence scoring so that
    emotionally charged complaints escalate even when the KB has
    a relevant policy answer.
    """
    query   = state["query"].strip()
    cleaned = query.lower()
    kw_hit  = any(kw in cleaned for kw in ESCALATION_KEYWORDS)

    print(f"\n╔══ Node 1 — QueryProcessor")
    print(f"  Query   : \"{query[:70]}{'...' if len(query) > 70 else ''}\"")
    print(f"  Keyword : {'⚠️  YES — escalation keyword detected' if kw_hit else '✅  NO'}")

    return {
        **state,
        "cleaned_query"       : cleaned,
        "requires_escalation" : kw_hit,
        "retrieved_docs"      : [],
        "confidence_score"    : 0.0,
        "route"               : "",
        "answer"              : "",
        "escalation_reason"   : "",
        "human_response"      : "",
        "error"               : None,
    }


# ── NODE 2: Retriever ───────────────────────────────────────────────
def make_retriever_node(collection):
    """
    Factory that binds the ChromaDB collection to the retriever node.
    Returns a LangGraph-compatible node function.
    """
    def retriever_node(state: GraphState) -> GraphState:
        """
        Queries ChromaDB for semantically similar chunks.
        Uses actual cosine distances (not doc count) for confidence scoring.

        distance = 0.0 → perfect match  → similarity = 1.0
        distance = 1.0 → orthogonal     → similarity = 0.0
        confidence = mean(similarities) across top-k chunks
        """
        try:
            texts, metas, distances = retrieve(collection, state["cleaned_query"])
            docs = [
                Document(page_content=t, metadata=m)
                for t, m in zip(texts, metas)
            ]
            confidence = compute_confidence(distances)
        except Exception as e:
            print(f"  ⚠️  Retriever error: {e}")
            return {**state, "retrieved_docs": [], "confidence_score": 0.0, "error": str(e)}

        print(f"\n╔══ Node 2 — Retriever")
        print(f"  Chunks      : {len(docs)}/{TOP_K}")
        print(f"  Distances   : {[round(d, 3) for d in distances]}")
        print(f"  Confidence  : {confidence:.3f}  (threshold: {CONFIDENCE_THRESHOLD})")

        return {**state, "retrieved_docs": docs, "confidence_score": confidence}

    return retriever_node


# ── NODE 3: Router ──────────────────────────────────────────────────
def router(state: GraphState) -> GraphState:
    """
    Decision node — core routing intelligence.

    Three independent escalation triggers (ANY one is sufficient):
      1. Keyword match  — explicit human/urgent/complaint signal
      2. Low confidence — retrieved chunks not semantically relevant
      3. No docs        — retrieval failed entirely
    """
    kw_esc   = state["requires_escalation"]
    low_conf = state["confidence_score"] < CONFIDENCE_THRESHOLD
    no_docs  = len(state["retrieved_docs"]) == 0

    escalate = kw_esc or low_conf or no_docs
    route    = "escalate" if escalate else "answer"
    reason   = (
        "escalation_keyword"   if kw_esc   else
        "no_context_retrieved" if no_docs  else
        "low_confidence"       if low_conf else ""
    )

    print(f"\n╔══ Node 3 — Router")
    print(f"  Keyword trigger : {kw_esc}")
    print(f"  Low confidence  : {low_conf}  ({state['confidence_score']:.3f} < {CONFIDENCE_THRESHOLD})")
    print(f"  No docs         : {no_docs}")
    print(f"  Decision        : {route.upper()} {'🤖' if route == 'answer' else '🚨'}")

    return {**state, "route": route, "escalation_reason": reason}


# ── NODE 4: Answer Generator ────────────────────────────────────────
def make_answer_node(llm: ChatGroq):
    """Factory that binds the LLM to the answer generator node."""

    def answer_generator(state: GraphState) -> GraphState:
        """
        RAG answer generation. Builds a structured prompt with:
          - System prompt: strict grounding instruction
          - Context: retrieved chunks as numbered sections
          - User turn: original customer query

        The 'answer ONLY from context' instruction is the primary
        hallucination prevention mechanism.
        """
        context = "\n\n".join(
            f"[Section {i+1} | Page {doc.metadata.get('page', '?')}]\n"
            f"{doc.page_content.strip()}"
            for i, doc in enumerate(state["retrieved_docs"])
        )

        messages = [
            SystemMessage(content=(
                "You are a professional customer support assistant for an e-commerce company. "
                "Answer the customer's question ONLY using the information in the context sections. "
                "Do NOT use any outside knowledge. "
                "If the context does not contain enough information, say: "
                "'I don't have complete information on that. Please contact our support team "
                "at support@shopify-demo.com or call 1800-000-0000.' "
                "Be concise, warm, and professional. Maximum 3 sentences."
            )),
            HumanMessage(content=(
                f"Context:\n{context}\n\n"
                f"Customer Question: {state['query']}\n\n"
                f"Answer:"
            ))
        ]

        try:
            response = llm.invoke(messages)
            answer   = response.content.strip()
        except Exception as e:
            print(f"  ⚠️  LLM error: {e}")
            answer = (
                "I'm experiencing a technical issue. "
                "Please contact support@shopify-demo.com or call 1800-000-0000."
            )

        print(f"\n╔══ Node 4 — AnswerGenerator")
        print(f"  Context chunks : {len(state['retrieved_docs'])}")
        print(f"  Answer length  : {len(answer)} chars")

        return {**state, "answer": answer}

    return answer_generator


# ── NODE 5: HITL Escalator ──────────────────────────────────────────
def hitl_escalator(state: GraphState) -> GraphState:
    """
    Human-in-the-Loop escalation node.

    Actions:
      - Logs structured escalation record to escalations.log (JSON Lines)
      - Returns professional customer-facing acknowledgement

    Production extensions:
      - POST to Slack webhook / create Zendesk ticket
      - Logged data feeds future intent classifier fine-tuning
    """
    os.makedirs(LOGS_DIR, exist_ok=True)

    record = {
        "timestamp"          : datetime.datetime.now().isoformat(),
        "query"              : state["query"],
        "confidence_score"   : state["confidence_score"],
        "escalation_reason"  : state["escalation_reason"],
        "requires_escalation": state["requires_escalation"],
    }

    with open(ESCALATION_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"\n╔══ Node 5 — HITL Escalator")
    print(f"  🚨 ESCALATION TRIGGERED")
    print(f"  Reason     : {state['escalation_reason']}")
    print(f"  Confidence : {state['confidence_score']:.3f}")
    print(f"  ✅ Logged to {ESCALATION_LOG}")

    return {
        **state,
        "human_response": (
            "Thank you for reaching out. Your query has been escalated to our dedicated "
            "support team. A human agent will contact you within 24 hours via email. "
            "We apologize for any inconvenience and appreciate your patience."
        )
    }


# ── Conditional edge selector ───────────────────────────────────────
def route_decision(state: GraphState) -> str:
    """Returns 'answer' or 'escalate' — used by LangGraph conditional edge."""
    return state["route"]
