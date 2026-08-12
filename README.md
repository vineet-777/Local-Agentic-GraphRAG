# 🧠 Local GraphRAG

A local GraphRAG multi-agent system. Upload your PDFs/DOCX files and get instant RAG + Knowledge Graph powered answers — **100% local**, powered by LM Studio.

## How It Works

```
Upload PDF/DOCX → Extract Text → Chunk → Embed (ChromaDB) + Build Graph (graph.json)
                                              ↓
              Query → [LangGraph Router] → Vector DB or Knowledge Graph
                                         → LLM (LM Studio) → Answer + Reasoning Trace
```

## Architecture

| Component | Technology | Model |
|---|---|---|
| Router Agent | LM Studio | gemma-4-e4b-it |
| Synthesiser | LM Studio | gemma-4-e4b-it |
| Embeddings | LM Studio | nomic-embed-text-v1.5 |
| Vector Store | ChromaDB | — |
| Knowledge Graph | JSON (graph.json) | — |
| Orchestration | LangGraph | — |
| Backend API | FastAPI | — |
| Frontend | Streamlit | — |

## Prerequisites

1. **Python 3.11+**
2. **LM Studio** installed and running on `http://localhost:1234`
3. Load these models in LM Studio:
   - `gemma-4-e4b-it` — Router + Synthesiser (Q4_K_M GGUF recommended)
   - `nomic-embed-text-v1.5` — Embeddings

> **Hardware note:** Tuned for AMD Radeon RX 6500M (4GB VRAM). Context length is set to 8192 (Gemma 4 E4B supports 128K, capped for VRAM safety).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Terminal 1: Start the API server
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000

# Terminal 2: Start the Streamlit UI
streamlit run ui/app.py
```

Then:
1. Open `http://localhost:8501` in your browser
2. Upload PDF/DOCX files via the sidebar
3. Wait for processing (progress bar shown)
4. Start chatting!

## Project Structure

```
├── core/
│   ├── config.py          # Centralized settings — single source of truth for all models
│   ├── ingest.py          # PDF/DOCX extraction → chunking → ChromaDB
│   ├── graph_builder.py   # LLM triple extraction → graph.json
│   ├── retrievers.py      # Vector + Graph retriever logic
│   └── engine.py          # LangGraph state machine
├── api/
│   └── server.py          # FastAPI (upload, query, health, status)
├── ui/
│   └── app.py             # Streamlit frontend
├── data/                  # Raw files (optional storage)
├── db/                    # Persisted ChromaDB + graph.json
│   ├── chromadb/
│   └── graph.json
├── uploads/               # Temporary upload staging (auto-cleaned)
├── requirements.txt
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | LM Studio + DB status |
| `/api/upload` | POST | Upload PDF/DOCX files for processing |
| `/api/upload-status` | GET | Poll background processing progress |
| `/api/query` | POST | Ask a question, get answer + reasoning trace |

## Features

- **Dynamic Ingestion**: Upload files anytime — data is additive, never overwritten
- **Dual Retrieval**: Automatic routing between vector search and knowledge graph
- **Reasoning Trace**: See exactly which path the AI took and what sources it used
- **Progress Tracking**: Real-time progress bar during file processing
- **Cold Start Protection**: Clear error messages for missing services or data
- **Persistent Storage**: All data survives app restarts (stored in `/db/`)
- **100% Local**: All inference via LM Studio — no cloud, no API keys, no data leaves your machine
