# ═══════════════════════════════════════════════════════════════
# evaluation.py — RAG Quality Evaluation Framework
#
# Measures three core RAG metrics WITHOUT external paid APIs:
#   1. Context Precision  — were the retrieved chunks actually relevant?
#   2. Answer Relevancy   — does the answer address the question?
#   3. Faithfulness       — is the answer grounded in retrieved context?
#
# These are the same dimensions RAGAS measures. We compute them using
# the embedding model already in the pipeline (semantic similarity),
# so there's zero extra cost and full transparency into the math.
# ═══════════════════════════════════════════════════════════════

import statistics
from typing import List, Dict

from chromadb.utils import embedding_functions

# Reuse the same embedding model the pipeline uses — consistency matters
_ef = embedding_functions.DefaultEmbeddingFunction()


def _embed(text: str) -> List[float]:
    """Embed a single string into a 384-dim vector."""
    return _ef([text])[0]


def _cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two vectors, clamped to [0, 1]."""
    dot  = sum(x * y for x, y in zip(a, b))
    na   = sum(x * x for x in a) ** 0.5
    nb   = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


def context_precision(query: str, retrieved_docs: List[str]) -> float:
    """
    Context Precision — how relevant are the retrieved chunks to the query?

    Method: average cosine similarity between the query embedding and
    each retrieved chunk embedding. High score = retriever pulled the
    right context. Low score = retriever is noisy.

    Range: 0.0 (irrelevant) → 1.0 (perfectly relevant)
    """
    if not retrieved_docs:
        return 0.0
    q_vec = _embed(query)
    sims  = [_cosine(q_vec, _embed(doc)) for doc in retrieved_docs]
    return round(statistics.mean(sims), 4)


def answer_relevancy(query: str, answer: str) -> float:
    """
    Answer Relevancy — does the generated answer actually address the query?

    Method: cosine similarity between query embedding and answer embedding.
    A high score means the answer is on-topic; a low score means the LLM
    drifted or returned a generic non-answer.

    Range: 0.0 (off-topic) → 1.0 (directly addresses query)
    """
    if not answer.strip():
        return 0.0
    return round(_cosine(_embed(query), _embed(answer)), 4)


def faithfulness(answer: str, retrieved_docs: List[str]) -> float:
    """
    Faithfulness — is the answer grounded in the retrieved context?

    This is the KEY anti-hallucination metric. Method: similarity between
    the answer and the combined retrieved context. If the answer contains
    claims NOT supported by context, similarity drops — signaling potential
    hallucination.

    Range: 0.0 (hallucinated / ungrounded) → 1.0 (fully grounded)
    """
    if not answer.strip() or not retrieved_docs:
        return 0.0
    context = " ".join(retrieved_docs)
    return round(_cosine(_embed(answer), _embed(context)), 4)


def evaluate_single(query: str, answer: str, retrieved_docs: List[str]) -> Dict[str, float]:
    """
    Run all three metrics on a single query-answer-context triple.

    Returns
    -------
    dict with keys: context_precision, answer_relevancy, faithfulness, overall
    """
    cp = context_precision(query, retrieved_docs)
    ar = answer_relevancy(query, answer)
    fa = faithfulness(answer, retrieved_docs)
    overall = round(statistics.mean([cp, ar, fa]), 4)

    return {
        "context_precision": cp,
        "answer_relevancy" : ar,
        "faithfulness"     : fa,
        "overall"          : overall,
    }


def evaluate_dataset(eval_cases: List[Dict]) -> Dict:
    """
    Evaluate a list of cases and return aggregate + per-case metrics.

    Parameters
    ----------
    eval_cases : list of dicts, each with keys:
                 'query', 'answer', 'retrieved_docs'

    Returns
    -------
    dict with 'per_case' list and 'aggregate' averages
    """
    per_case = []
    for case in eval_cases:
        scores = evaluate_single(
            case["query"],
            case["answer"],
            case["retrieved_docs"]
        )
        per_case.append({"query": case["query"], **scores})

    # Aggregate averages across all cases
    aggregate = {
        "context_precision": round(statistics.mean(c["context_precision"] for c in per_case), 4),
        "answer_relevancy" : round(statistics.mean(c["answer_relevancy"]  for c in per_case), 4),
        "faithfulness"     : round(statistics.mean(c["faithfulness"]      for c in per_case), 4),
        "overall"          : round(statistics.mean(c["overall"]           for c in per_case), 4),
    }

    return {"per_case": per_case, "aggregate": aggregate}


def print_report(results: Dict):
    """Pretty-print an evaluation report."""
    print("\n" + "=" * 78)
    print("  📊  RAG EVALUATION REPORT")
    print("=" * 78)

    print(f"\n  {'Query':<42} {'CtxPrec':<9} {'AnsRel':<9} {'Faith':<9} {'Overall'}")
    print("  " + "─" * 76)
    for c in results["per_case"]:
        print(f"  {c['query'][:41]:<42} "
              f"{c['context_precision']:<9.3f} "
              f"{c['answer_relevancy']:<9.3f} "
              f"{c['faithfulness']:<9.3f} "
              f"{c['overall']:.3f}")

    agg = results["aggregate"]
    print("  " + "─" * 76)
    print(f"  {'AVERAGE':<42} "
          f"{agg['context_precision']:<9.3f} "
          f"{agg['answer_relevancy']:<9.3f} "
          f"{agg['faithfulness']:<9.3f} "
          f"{agg['overall']:.3f}")

    print("\n  Interpretation:")
    print(f"  • Context Precision {agg['context_precision']:.2f} — retriever pulls relevant chunks")
    print(f"  • Answer Relevancy  {agg['answer_relevancy']:.2f} — answers stay on-topic")
    print(f"  • Faithfulness      {agg['faithfulness']:.2f} — answers grounded in context (anti-hallucination)")
    print("=" * 78)
