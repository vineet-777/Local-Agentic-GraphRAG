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
2. Create an account or sign in on the welcome screen
3. Ask questions about indexed data in Query Studio
4. Explore the persisted knowledge graph in Knowledge Graph Explorer

The Streamlit UI does not include document upload or ingestion monitoring. The authenticated upload and status API endpoints remain available to API clients.

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
| `/api/conversations` | GET | List the signed-in user's conversations (bearer token required) |
| `/api/conversations/{id}` | GET | Load an owned conversation with message traces (bearer token required) |
| `/api/conversations/{id}` | PATCH | Rename an owned conversation (bearer token required) |
| `/api/conversations/{id}` | DELETE | Delete an owned conversation (bearer token required) |
| `/api/auth/signup` | POST | Create an account and return a session token |
| `/api/auth/signin` | POST | Authenticate and return a session token |
| `/api/auth/me` | GET | Return the signed-in user (Bearer token required) |
| `/api/auth/logout` | POST | Revoke the current session |
| `/api/health` | GET | LM Studio + DB status |
| `/api/upload` | POST | Upload PDF/DOCX files for processing (Bearer token required) |
| `/api/upload-status` | GET | Poll processing progress (Bearer token required) |
| `/api/query` | POST | Ask a question (Bearer token required) |

Signup and sign-in return an opaque bearer token that expires after seven days. Send it as `Authorization: Bearer <token>` to protected endpoints. Passwords are stored as Argon2 hashes and account/session data is persisted locally in `db/auth.sqlite3`.

## Features

- **API-Based Ingestion**: Authenticated upload endpoints add data without overwriting existing records
- **Persistent Chat History**: Per-user conversations with searchable sidebar history, rename, delete, and restored provenance traces
- **Dual Retrieval**: Automatic routing between vector search and knowledge graph
- **Reasoning Trace**: See exactly which path the AI took and what sources it used
- **Processing Status API**: Poll background file-processing progress
- **Cold Start Protection**: Clear error messages for missing services or data
- **Persistent Storage**: All data survives app restarts (stored in `/db/`)
- **Local Authentication**: SQLite accounts, Argon2 password hashing, expiring and revocable sessions
- **100% Local**: All inference via LM Studio — no cloud, no API keys, no data leaves your machine

> Accounts authenticate access to the local GraphRAG API. The knowledge base is shared by accounts on this local installation; it is not partitioned per user.
