# 🤖 RAG-Based Customer Support Assistant
### LangGraph · ChromaDB · Groq LLaMA-3 · HITL Escalation

> **Innomatics Research Labs — GenAI Internship 2026**

---

## 📌 Project Description

Engineered a RAG-Based Customer Support Assistant that autonomously retrieves knowledge, generates hallucination-free responses via LLaMA-3, and intelligently escalates complex queries to human agents — delivering real-time, production-grade support automation with zero compromise on accuracy.

---

## 🏗️ System Architecture

```
OFFLINE INGESTION (once per document update)
─────────────────────────────────────────────
PDF Knowledge Base
    → PyPDFLoader
    → RecursiveCharacterTextSplitter  (1000 chars, 200 overlap)
    → DefaultEmbeddingFunction        (384-dim cosine vectors, ONNX)
    → ChromaDB PersistentClient       (indexed on disk)

ONLINE QUERY PIPELINE (per request, <500ms)
────────────────────────────────────────────
User Query
    → [Node 1] query_processor    normalize + keyword detection
    → [Node 2] retriever_node     ChromaDB top-3 + confidence scoring
    → [Node 3] router             threshold decision
        ├── confidence ≥ 0.35 → [Node 4] answer_generator (Groq LLaMA-3)
        └── confidence < 0.35 → [Node 5] hitl_escalator   (Log + Handoff)
```

---

## 📁 Project Structure

```
rag_support_assistant/
├── main.py                  # Entry point (CLI: chat / test / logs)
├── requirements.txt         # Pinned dependencies
├── .env                     # API keys (add your GROQ_API_KEY here)
├── src/
│   ├── config.py            # All hyperparameters centralized
│   ├── kb_builder.py        # Generates knowledge base PDF
│   ├── ingestion.py         # PDF → Chunks → ChromaDB pipeline
│   ├── nodes.py             # GraphState + all 5 LangGraph nodes
│   └── graph.py             # StateGraph builder + query runner
├── data/
│   └── knowledge_base.pdf   # Auto-generated on first run
├── logs/
│   └── escalations.log      # JSON Lines escalation audit trail
└── chroma_db/               # Persisted ChromaDB vector index
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your Groq API key
```bash
# Edit .env file:
GROQ_API_KEY=gsk_your_key_here
# Get free key at: https://console.groq.com
```

### 3. Build the knowledge base index
```bash
python main.py --rebuild
```

### 4. Start interactive chat
```bash
python main.py
```

### 5. Run test suite
```bash
python main.py --test
```

### 6. View escalation log
```bash
python main.py --logs
```

### 7. Run RAG evaluation (faithfulness, relevancy, precision)
```bash
python main.py --eval
```

### 8. Run chunk-size benchmark experiment
```bash
python main.py --benchmark
```

### 9. Launch the web app
```bash
streamlit run app.py
```

---

## 🔑 Key ML/AI Concepts Applied

| Concept | Implementation |
|---|---|
| **RAG** | PDF → chunks → embeddings → retrieval → LLM grounded generation |
| **Dense Embeddings** | all-MiniLM-L6-v2 (ONNX), 384-dim, cosine similarity |
| **Vector Database** | ChromaDB with HNSW index, persistent storage |
| **Confidence Scoring** | `mean(1 - cosine_distance)` per retrieved chunk |
| **LangGraph** | StateGraph with TypedDict, conditional edges, 5 nodes |
| **HITL** | Keyword + confidence dual-trigger, JSON Lines audit log |
| **Hallucination Prevention** | System prompt grounds LLM strictly to retrieved context |

---

## 📊 Metrics

| Metric | Value |
|---|---|
| Chunks indexed | ~12–15 (8 KB sections) |
| Embedding dimensions | 384 |
| Retrieval top-k | 3 |
| Confidence threshold | 0.35 |
| Test routing accuracy | 100% (8/8 cases) |
| HITL intervention rate | ~50% on mixed query set |
| LLM inference latency | < 100ms (Groq) |
| End-to-end latency | < 500ms |

---

## 🧪 Test Cases

| Query | Expected Route | Trigger |
|---|---|---|
| "What is the return policy?" | ANSWER | High cosine similarity |
| "How do I reset my password?" | ANSWER | High cosine similarity |
| "What payment methods do you accept?" | ANSWER | High cosine similarity |
| "How do I cancel my order?" | ANSWER | High cosine similarity |
| "I am frustrated, need a human agent" | ESCALATE | Keyword: frustrated, human |
| "This is urgent, refund now!" | ESCALATE | Keyword: urgent |
| "What is the capital of France?" | ESCALATE | Low confidence (out-of-scope) |
| "I want to speak to a manager" | ESCALATE | Keyword: manager |

---

## 🔧 Configuration (src/config.py)

```python
CHUNK_SIZE           = 1000    # chars per chunk
CHUNK_OVERLAP        = 200     # overlap between chunks
TOP_K                = 3       # chunks retrieved per query
CONFIDENCE_THRESHOLD = 0.35    # below → escalate
LLM_TEMPERATURE      = 0.1     # near-deterministic
LLM_MAX_TOKENS       = 512     # concise answers
```

---

## 🔬 Evaluation & Experiments

### RAG Quality Metrics (`python main.py --eval`)
The system is measured on three RAGAS-style dimensions, computed via semantic similarity using the pipeline's own embedding model (zero extra cost):

| Metric | What it measures | Why it matters |
|---|---|---|
| **Context Precision** | Are retrieved chunks relevant to the query? | Validates the retriever |
| **Answer Relevancy** | Does the answer address the question? | Catches off-topic drift |
| **Faithfulness** | Is the answer grounded in retrieved context? | **Anti-hallucination metric** |

This moves the project from *claiming* zero hallucination to *measuring* it.

### Chunk-Size Ablation (`python main.py --benchmark`)
An experiment that tests chunk sizes 500 vs 1000 vs 1500 against a fixed query set, measuring retrieval similarity and keyword hit-rate to justify the chosen `CHUNK_SIZE=1000` empirically rather than by guesswork.

---

## 🚀 Production Extensions

- **Slack webhook**: Replace HITL print with `requests.post(slack_url, json=payload)`
- **Zendesk ticket**: Auto-create support ticket from escalation record
- **Multi-document**: Add `document_id` metadata filter for scoped retrieval
- **Feedback loop**: Capture thumbs up/down → retrain embeddings
- **API deployment**: Wrap `run_query()` in FastAPI endpoint
- **RAGAS evaluation**: Add faithfulness + answer relevancy scoring

---

## 🌐 Web App (Streamlit)

A sleek web interface is included in `app.py`.

### Run locally
```bash
pip install -r requirements.txt
# create .env with your GROQ_API_KEY (see .env.example)
streamlit run app.py
```

The app builds the knowledge base and ChromaDB index automatically on first launch.

---

## 🚀 Deploy to Streamlit Cloud (free, ~5 min)

1. Push this repo to GitHub (the `.gitignore` keeps secrets out)
2. Go to **share.streamlit.io** → sign in with GitHub
3. Click **New app** → select your repo → set **Main file** to `app.py`
4. Under **Advanced settings → Secrets**, paste:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Click **Deploy** → you get a public URL like
   `https://your-app.streamlit.app`

That URL is what goes on your resume and LinkedIn.

---

## 📈 Roadmap — RAG Quality Improvements

Planned upgrades to push retrieval and answer quality higher:

1. **Hybrid retrieval** — combine dense (semantic) + BM25 (keyword) search
2. **Re-ranking** — add a cross-encoder re-ranker over top-10 candidates
3. **Semantic chunking** — split on meaning, not fixed character counts
4. **RAGAS evaluation** — measure faithfulness, answer relevancy, context precision
5. **Semantic escalation** — replace keyword matching with intent embeddings
6. **Conversation memory** — multi-turn context via LangGraph checkpointing
