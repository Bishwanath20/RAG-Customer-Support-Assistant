#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════
# main.py — RAG Customer Support Assistant Entry Point
#
# Usage:
#   python main.py              → Interactive chat mode
#   python main.py --rebuild    → Force rebuild ChromaDB index
#   python main.py --test       → Run all test cases
#   python main.py --logs       → View escalation log
#
# First time setup:
#   1. pip install -r requirements.txt
#   2. Add GROQ_API_KEY to .env
#   3. python main.py --rebuild
#   4. python main.py
# ═══════════════════════════════════════════════════════════════

import os
import sys
import json
import argparse
import textwrap

from langchain_groq import ChatGroq

from src.config import (
    GROQ_API_KEY, GROQ_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    CHROMA_DIR, PDF_PATH, ESCALATION_LOG, DATA_DIR, LOGS_DIR
)
from src.kb_builder  import create_knowledge_base
from src.ingestion   import build_index, load_index
from src.graph       import build_graph, run_query


# ── Display helpers ─────────────────────────────────────────────────

def print_banner():
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   🤖  RAG-Based Customer Support Assistant                  ║")
    print("║   LangGraph · ChromaDB · Groq LLaMA-3 · HITL Escalation    ║")
    print("║   Innomatics Research Labs — GenAI Internship 2026          ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()


def display_result(state: dict):
    """Print a formatted result card for a query."""
    route = state["route"]
    icon  = "🤖" if route == "answer" else "🚨"
    W     = 63

    print()
    print("╔" + "═" * W + "╗")
    q = state["query"]
    print(f"║  {icon}  Query      : {q[:W-16]:<{W-16}}  ║")
    print(f"║  📊  Route      : {route.upper():<{W-16}}  ║")
    print(f"║  📈  Confidence : {state['confidence_score']:.3f}{'':>{W-23}}  ║")
    print("╠" + "═" * W + "╣")

    if route == "answer":
        print(f"║  💬  Response:{'':>{W-14}}  ║")
        wrapped = textwrap.wrap(state["answer"], width=W - 5)
        for line in wrapped:
            print(f"║     {line:<{W-5}}  ║")
    else:
        print(f"║  🚨  ESCALATED TO HUMAN AGENT{'':>{W-30}}  ║")
        print(f"║  📋  Reason : {state['escalation_reason']:<{W-14}}  ║")
        wrapped = textwrap.wrap(state["human_response"], width=W - 5)
        print(f"║  📨  Agent Response:{'':>{W-20}}  ║")
        for line in wrapped:
            print(f"║     {line:<{W-5}}  ║")

    print("╚" + "═" * W + "╝")


# ── Modes ───────────────────────────────────────────────────────────

def run_tests(graph):
    """Run all predefined test cases and print a pass/fail summary."""
    test_cases = [
        ("What is the return policy?",                       "ANSWER"),
        ("I am very frustrated and need a human agent now",  "ESCALATE"),
        ("What is the capital of France?",                   "ESCALATE"),
        ("How do I reset my password?",                      "ANSWER"),
        ("What payment methods do you accept?",              "ANSWER"),
        ("This is urgent, I need a refund now!",             "ESCALATE"),
        ("How do I cancel my order?",                        "ANSWER"),
        ("I want to speak to a manager",                     "ESCALATE"),
    ]

    print("\n🧪 RUNNING TEST SUITE")
    print("─" * 84)
    print(f"  {'#':<3} {'Query':<45} {'Expected':<10} {'Got':<10} {'Conf':<7} Result")
    print("─" * 84)

    passed = 0
    results = []

    for i, (query, expected) in enumerate(test_cases, 1):
        state = run_query(graph, query)
        got   = state["route"].upper()
        conf  = state["confidence_score"]
        ok    = got == expected
        icon  = "✅ PASS" if ok else "❌ FAIL"
        if ok:
            passed += 1
        results.append((query, expected, got, conf, ok))
        print(f"  {i:<3} {query[:44]:<45} {expected:<10} {got:<10} {conf:<7.3f} {icon}")

    print("─" * 84)
    accuracy = passed / len(test_cases) * 100
    print(f"\n  ✅  Passed   : {passed}/{len(test_cases)}")
    print(f"  📈  Accuracy : {accuracy:.1f}%\n")

    if accuracy == 100.0:
        print("  🏆  Perfect score — all routing decisions correct!")
    else:
        print(f"  ⚠️   {len(test_cases) - passed} test(s) failed — adjust CONFIDENCE_THRESHOLD in config.py")


def view_logs():
    """Display the escalation log."""
    print("\n📋 ESCALATION LOG")
    print("=" * 65)

    if not os.path.exists(ESCALATION_LOG):
        print("  No escalations logged yet.")
        return

    with open(ESCALATION_LOG) as f:
        lines = f.readlines()

    print(f"  Total escalations: {len(lines)}\n")
    for i, line in enumerate(lines, 1):
        rec = json.loads(line.strip())
        print(f"  [{i}]  {rec['timestamp'][:19]}")
        print(f"        Query      : {rec['query']}")
        print(f"        Reason     : {rec['escalation_reason']}")
        print(f"        Confidence : {rec['confidence_score']:.3f}")
        print()

    print("=" * 65)


def interactive_mode(graph):
    """Live interactive chat mode."""
    print_banner()
    print("  Type your customer query below.")
    print("  Commands: 'exit' to quit | 'logs' to view escalation log")
    print("  ─────────────────────────────────────────────────────────")
    print()
    print("  Try these:")
    print("    → 'What is the return policy?'")
    print("    → 'How long does shipping take?'")
    print("    → 'I want to speak to a manager'   (triggers HITL)")
    print("    → 'Who won the FIFA World Cup?'     (out-of-scope)")
    print()

    session = 0
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Session ended.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye", "q"):
            print(f"  👋 Session ended after {session} queries. Goodbye!")
            break

        if user_input.lower() == "logs":
            view_logs()
            continue

        session += 1
        state = run_query(graph, user_input)
        display_result(state)
        print()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RAG-Based Customer Support Assistant"
    )
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild the ChromaDB index from PDF")
    parser.add_argument("--test",    action="store_true",
                        help="Run predefined test suite")
    parser.add_argument("--logs",    action="store_true",
                        help="View escalation log")
    args = parser.parse_args()

    # ── Validate API key ──────────────────────────────────────────
    if not GROQ_API_KEY or GROQ_API_KEY.startswith("gsk_PASTE"):
        print("❌  GROQ_API_KEY not set.")
        print("    1. Get your free key at https://console.groq.com")
        print("    2. Add it to .env: GROQ_API_KEY=gsk_your_key_here")
        sys.exit(1)

    # ── Build knowledge base PDF if missing ───────────────────────
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    if not os.path.exists(PDF_PATH):
        print("📄  Knowledge base PDF not found — generating...")
        create_knowledge_base(PDF_PATH)

    # ── Build or load ChromaDB index ──────────────────────────────
    if args.rebuild or not os.path.exists(CHROMA_DIR):
        print("🔄  Building ChromaDB index...")
        collection = build_index(PDF_PATH, CHROMA_DIR)
    else:
        print("✅  Loading existing ChromaDB index...")
        collection = load_index(CHROMA_DIR)
        print(f"   Chunks in index: {collection.count()}\n")

    # ── Initialize LLM ────────────────────────────────────────────
    print("🧠  Initializing Groq LLM...")
    llm = ChatGroq(
        api_key     = GROQ_API_KEY,
        model       = GROQ_MODEL,
        temperature = LLM_TEMPERATURE,
        max_tokens  = LLM_MAX_TOKENS
    )
    print(f"   Model: {GROQ_MODEL}  |  Temp: {LLM_TEMPERATURE}\n")

    # ── Compile graph ─────────────────────────────────────────────
    print("🕸️   Compiling LangGraph...")
    graph = build_graph(collection, llm)
    print("   Graph ready: 5 nodes | conditional routing | HITL enabled\n")

    # ── Dispatch mode ─────────────────────────────────────────────
    if args.logs:
        view_logs()
    elif args.test:
        run_tests(graph)
    else:
        interactive_mode(graph)


if __name__ == "__main__":
    main()
