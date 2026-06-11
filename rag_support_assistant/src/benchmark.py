# ═══════════════════════════════════════════════════════════════
# benchmark.py — Chunk Size Ablation Experiment
#
# Scientific question: Does chunk size affect retrieval quality?
# We test 500 vs 1000 vs 1500 char chunks and measure retrieval
# precision on a fixed set of queries. This demonstrates evidence-
# based hyperparameter selection rather than guessing.
#
# Run:  python -m src.benchmark
# ═══════════════════════════════════════════════════════════════

import os
import shutil
import statistics

import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import PDF_PATH, TOP_K
from src.kb_builder import create_knowledge_base


# Fixed evaluation queries with the KB section they SHOULD retrieve.
# Used to measure whether smaller/larger chunks retrieve better context.
BENCHMARK_QUERIES = [
    ("What is the return policy?",            "return"),
    ("How do I reset my password?",           "password"),
    ("What payment methods do you accept?",   "payment"),
    ("How long does shipping take?",          "shipping"),
    ("How do I cancel my order?",             "cancel"),
    ("What does the warranty cover?",         "warranty"),
    ("What are your support hours?",          "support"),
    ("How do loyalty points work?",           "loyalty"),
]

CHUNK_CONFIGS = [
    (500,  100),   # small chunks, 20% overlap
    (1000, 200),   # medium (our default)
    (1500, 300),   # large chunks
]

_ef = embedding_functions.DefaultEmbeddingFunction()


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(y * y for y in b) ** 0.5
    return max(0.0, dot / (na * nb)) if na and nb else 0.0


def build_temp_index(chunk_size: int, overlap: int, raw_docs):
    """Build a throwaway in-memory ChromaDB index for one chunk config."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(raw_docs)

    client = chromadb.EphemeralClient()  # in-memory, no disk persistence
    col = client.get_or_create_collection(
        name=f"bench_{chunk_size}",
        embedding_function=_ef,
        metadata={"hnsw:space": "cosine"}
    )
    col.add(
        documents=[c.page_content for c in chunks],
        ids=[f"c{i}" for i in range(len(chunks))]
    )
    return col, len(chunks)


def run_benchmark():
    """Run the chunk-size ablation and print a comparison table."""
    if not os.path.exists(PDF_PATH):
        create_knowledge_base(PDF_PATH)

    raw_docs = PyPDFLoader(PDF_PATH).load()

    print("\n" + "=" * 72)
    print("  🔬  CHUNK SIZE ABLATION EXPERIMENT")
    print("  Question: Which chunk size gives the best retrieval quality?")
    print("=" * 72)

    summary = []

    for chunk_size, overlap in CHUNK_CONFIGS:
        col, n_chunks = build_temp_index(chunk_size, overlap, raw_docs)

        # For each query, measure: (a) avg retrieval similarity,
        # (b) whether the expected keyword appears in top-k results (hit rate)
        sims, hits = [], 0

        for query, expected_kw in BENCHMARK_QUERIES:
            res = col.query(
                query_texts=[query], n_results=TOP_K,
                include=["documents", "distances"]
            )
            docs      = res["documents"][0]
            distances = res["distances"][0]

            avg_sim = statistics.mean([max(0.0, 1 - d) for d in distances])
            sims.append(avg_sim)

            # Hit = expected keyword found in any retrieved chunk
            combined = " ".join(docs).lower()
            if expected_kw in combined:
                hits += 1

        avg_similarity = round(statistics.mean(sims), 4)
        hit_rate       = round(hits / len(BENCHMARK_QUERIES) * 100, 1)

        summary.append({
            "chunk_size": chunk_size,
            "overlap"   : overlap,
            "n_chunks"  : n_chunks,
            "avg_sim"   : avg_similarity,
            "hit_rate"  : hit_rate,
        })

    # ── Print comparison table ────────────────────────────────────
    print(f"\n  {'Chunk Size':<12} {'Overlap':<9} {'# Chunks':<10} "
          f"{'Avg Similarity':<16} {'Hit Rate'}")
    print("  " + "─" * 64)
    for s in summary:
        print(f"  {s['chunk_size']:<12} {s['overlap']:<9} {s['n_chunks']:<10} "
              f"{s['avg_sim']:<16.4f} {s['hit_rate']}%")

    # ── Determine winner ──────────────────────────────────────────
    best = max(summary, key=lambda s: (s["hit_rate"], s["avg_sim"]))
    print("  " + "─" * 64)
    print(f"\n  🏆  Best config: chunk_size={best['chunk_size']}, "
          f"overlap={best['overlap']}")
    print(f"      Hit rate {best['hit_rate']}% | Avg similarity {best['avg_sim']:.4f}")

    print("\n  📝  Analysis:")
    print("      • Smaller chunks (500) → more precise but may split context")
    print("      • Larger chunks (1500) → richer context but diluted embeddings")
    print("      • The winning config balances precision and context completeness")
    print("=" * 72)

    return summary


if __name__ == "__main__":
    run_benchmark()
