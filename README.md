# RAG System with History-Based Self-Learning

A complete **Retrieval-Augmented Generation (RAG)** pipeline that answers questions over custom documents — with a self-learning layer that improves retrieval over time based on past interactions.

## Demo

> Run locally with Streamlit — see [Setup](#setup) below.

---

## How It Works

```
User Query
    │
    ▼
ChromaDB Vector Store  ──(BGE Embeddings)──▶  Top-K Relevant Chunks
    │
    ▼
Self-Learning Layer  ──(boost docs that helped before)──▶  Re-ranked Chunks
    │
    ▼
Gemini 2.5 Flash LLM  ──▶  Final Answer
    │
    ▼
Streamlit UI  ──▶  Answer + Retrieval Details
```

---

## Features

- **Vector Search** — Documents are embedded using `BAAI/bge-small-en-v1.5` (HuggingFace) and stored in ChromaDB for fast semantic retrieval
- **LLM Generation** — Google Gemini 2.5 Flash generates answers grounded in retrieved context
- **Self-Learning Layer** — Detects similar past queries (SequenceMatcher > 0.8 similarity) and boosts relevance scores (×1.5) of document chunks that succeeded historically — persisted as `rag_history.json`
- **Streamlit UI** — Interactive chat interface with expandable retrieval details and self-learning notifications

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestration | LlamaIndex |
| Vector Store | ChromaDB (persistent) |
| Embeddings | BAAI/bge-small-en-v1.5 (HuggingFace) |
| LLM | Google Gemini 2.5 Flash |
| UI | Streamlit |
| Self-Learning | SequenceMatcher + JSON history |

---

## Project Structure

```
Rag_System/
├── rag_system.py       # Core RAG logic + Streamlit UI
├── app.py              # App entry point
├── rag_history.json    # Persisted interaction history
├── data/
│   └── documents.txt   # Source documents for the vector DB
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install llama-index llama-index-vector-stores-chroma \
            llama-index-embeddings-huggingface \
            llama-index-llms-gemini \
            sentence-transformers chromadb streamlit
```

### 2. Set your Gemini API key

Open `rag_system.py` and replace:
```python
GEMINI_API_KEY = "PUT YOUR GEMINI API KEY"
```

Or set it as an environment variable:
```bash
export GEMINI_API_KEY="your_key_here"
```

### 3. Add your documents

Place your `.txt` files inside the `data/` folder.

### 4. Run

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`. On first run, ChromaDB will build the vector index automatically.

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `GEMINI_MODEL_NAME` | `models/gemini-2.5-flash` | LLM model |
| `chunk_size` | 512 tokens | Document chunk size |
| `chunk_overlap` | 50 tokens | Overlap between chunks |
| `similarity_top_k` | 5 | Number of chunks retrieved per query |
| Self-learning threshold | 0.8 | Minimum similarity to trigger boost |
