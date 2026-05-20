# ═══════════════════════════════════════════════════════════════
# graph.py — Builds and compiles the LangGraph StateGraph
# ═══════════════════════════════════════════════════════════════

import chromadb
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from src.nodes import (
    GraphState,
    query_processor,
    make_retriever_node,
    router,
    make_answer_node,
    hitl_escalator,
    route_decision,
)


def build_graph(collection: chromadb.Collection, llm: ChatGroq):
    """
    Assemble and compile the full LangGraph StateGraph.

    Topology
    --------
    START
      → query_processor   (Node 1: normalize + keyword detection)
      → retriever_node    (Node 2: ChromaDB retrieval + confidence)
      → router            (Node 3: conditional branch point)
          ├── 'answer'   → answer_generator  (Node 4: Groq LLM)  → END
          └── 'escalate' → hitl_escalator    (Node 5: HITL log)  → END

    Parameters
    ----------
    collection : ChromaDB collection (already indexed)
    llm        : Initialized ChatGroq instance

    Returns
    -------
    Compiled LangGraph executable
    """
    retriever_node   = make_retriever_node(collection)
    answer_generator = make_answer_node(llm)

    workflow = StateGraph(GraphState)

    # Register all nodes
    workflow.add_node("query_processor",  query_processor)
    workflow.add_node("retriever_node",   retriever_node)
    workflow.add_node("router",           router)
    workflow.add_node("answer_generator", answer_generator)
    workflow.add_node("hitl_escalator",   hitl_escalator)

    # Entry point
    workflow.set_entry_point("query_processor")

    # Unconditional sequential edges
    workflow.add_edge("query_processor", "retriever_node")
    workflow.add_edge("retriever_node",  "router")

    # Conditional branch at router
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "answer"  : "answer_generator",
            "escalate": "hitl_escalator",
        }
    )

    # Both terminal nodes → END
    workflow.add_edge("answer_generator", END)
    workflow.add_edge("hitl_escalator",   END)

    return workflow.compile()


def run_query(graph, query: str) -> GraphState:
    """
    Execute one query through the full RAG pipeline.

    Parameters
    ----------
    graph : Compiled LangGraph
    query : Customer's natural language question

    Returns
    -------
    GraphState : Complete final state with answer or human_response
    """
    initial_state: GraphState = {
        "query"               : query,
        "cleaned_query"       : "",
        "retrieved_docs"      : [],
        "confidence_score"    : 0.0,
        "route"               : "",
        "answer"              : "",
        "escalation_reason"   : "",
        "human_response"      : "",
        "requires_escalation" : False,
        "error"               : None,
    }
    return graph.invoke(initial_state)
