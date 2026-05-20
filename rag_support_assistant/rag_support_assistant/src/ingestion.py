# ═══════════════════════════════════════════════════════════════
# ingestion.py — PDF → Chunks → Embeddings → ChromaDB
# Offline phase: run once per document update
# ═══════════════════════════════════════════════════════════════

import os
import shutil
import statistics

import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHROMA_DIR, PDF_PATH,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K
)


def build_index(pdf_path: str = PDF_PATH, persist_dir: str = CHROMA_DIR) -> chromadb.Collection:
    """
    Full ingestion pipeline: PDF → chunks → embeddings → ChromaDB.

    Steps
    -----
    1. Load PDF pages with PyPDFLoader
    2. Split into overlapping chunks with RecursiveCharacterTextSplitter
    3. Initialize ChromaDB with DefaultEmbeddingFunction (ONNX, 384-dim)
    4. Batch insert all chunks into cosine-indexed collection

    Parameters
    ----------
    pdf_path    : Path to the source PDF knowledge base
    persist_dir : Directory to persist ChromaDB index

    Returns
    -------
    chromadb.Collection : Ready-to-query collection
    """
    print("=" * 60)
    print("  📂  INGESTION PIPELINE")
    print("=" * 60)

    # ── Step 1: Load PDF ────────────────────────────────────────
    print("\n  STEP 1: Loading PDF...")
    loader   = PyPDFLoader(pdf_path)
    raw_docs = loader.load()
    total_chars = sum(len(d.page_content) for d in raw_docs)
    print(f"  ✅ {len(raw_docs)} page(s) | {total_chars:,} total characters")

    # ── Step 2: Chunk ───────────────────────────────────────────
    print("\n  STEP 2: Splitting into overlapping chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators    = ["\n\n", "\n", ". ", " ", ""]
    )
    chunks  = splitter.split_documents(raw_docs)
    avg_len = sum(len(c.page_content) for c in chunks) // len(chunks)
    print(f"  ✅ {len(chunks)} chunks | avg {avg_len} chars each")

    # ── Step 3: Embedding model ─────────────────────────────────
    print("\n  STEP 3: Loading embedding model...")
    ef = embedding_functions.DefaultEmbeddingFunction()
    print("  ✅ DefaultEmbeddingFunction (all-MiniLM-L6-v2, ONNX, 384-dim)")

    # ── Step 4: ChromaDB ────────────────────────────────────────
    print("\n  STEP 4: Building ChromaDB vector store...")
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    client     = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name               = "customer_support_kb",
        embedding_function = ef,
        metadata           = {"hnsw:space": "cosine"}
    )

    collection.add(
        documents = [c.page_content for c in chunks],
        ids       = [f"chunk_{i:04d}" for i in range(len(chunks))],
        metadatas = [
            {
                "page"     : int(c.metadata.get("page", 0)),
                "source"   : pdf_path,
                "chunk_id" : i,
                "length"   : len(c.page_content)
            }
            for i, c in enumerate(chunks)
        ]
    )

    print(f"\n{'=' * 60}")
    print(f"  ✅  INGESTION COMPLETE")
    print(f"     Chunks indexed  : {collection.count()}")
    print(f"     Persist path    : {persist_dir}/")
    print(f"     Similarity      : cosine")
    print(f"{'=' * 60}\n")

    return collection


def load_index(persist_dir: str = CHROMA_DIR) -> chromadb.Collection:
    """
    Load an existing ChromaDB index from disk.
    Call this instead of build_index() if the index already exists.
    """
    ef     = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_collection(
        name               = "customer_support_kb",
        embedding_function = ef
    )


def retrieve(collection: chromadb.Collection, query: str, top_k: int = TOP_K):
    """
    Query ChromaDB for the top-k most similar chunks.

    Returns
    -------
    tuple : (List[str] texts, List[dict] metadatas, List[float] distances)
    """
    results = collection.query(
        query_texts = [query],
        n_results   = top_k,
        include     = ["documents", "metadatas", "distances"]
    )
    return (
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )


def compute_confidence(distances: list) -> float:
    """
    Convert ChromaDB cosine distances to a confidence score [0.0, 1.0].

    Formula: confidence = mean(1 - distance_i)
    Cosine distance semantics:
        0.0 → identical   → similarity 1.0
        0.5 → partial     → similarity 0.5
        1.0 → orthogonal  → similarity 0.0
        >1.0 → opposite   → clipped to 0.0
    """
    if not distances:
        return 0.0
    similarities = [max(0.0, 1.0 - d) for d in distances]
    return round(statistics.mean(similarities), 4)
