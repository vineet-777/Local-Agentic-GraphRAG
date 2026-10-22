# Local GraphRAG â€” Complete Project Documentation

**Project Title:** Local GraphRAG â€” A Multi-Agent Graph-Augmented Retrieval System  
**Tagline:** Upload documents, ask questions â€” 100% local, powered by GraphRAG + LM Studio  
**Current LLM:** Gemma 4 E4B (Q4_K_M GGUF) via LM Studio  

---

## 1. Project Overview

Local GraphRAG is a fully local, multi-agent Retrieval-Augmented Generation (RAG) system that combines traditional vector similarity search with a knowledge graph for dual-path retrieval. Authenticated API clients can submit PDF or DOCX documents; the system extracts text, chunks it, embeds the chunks into ChromaDB, and simultaneously constructs a JSON-based knowledge graph by extracting entity-relationship triples using an LLM. The Streamlit interface provides Query Studio and Knowledge Graph Explorer workspaces for querying and exploring indexed data. When a user asks a question, a LangGraph-orchestrated pipeline classifies the query, routes it to the appropriate retrieval backend (vector DB or knowledge graph), retrieves relevant context, and generates a grounded answer â€” all using locally-hosted models via LM Studio's OpenAI-compatible API.

### Core Problem

Standard RAG systems retrieve context only through vector similarity, which works well for factual look-ups but fails for relational or comparative queries (e.g., "How does X relate to Y?"). Local GraphRAG solves this by adding a second retrieval path â€” a knowledge graph built from extracted triples â€” and an intelligent router that selects the best path per query.

### Key Differentiators

- **Dual retrieval:** Vector DB for factual queries, Knowledge Graph for relational queries
- **LLM-powered routing:** A dedicated router agent classifies each query before retrieval
- **Reasoning trace:** Every step (classify â†’ retrieve â†’ generate) is logged with latency and metadata, displayed with the corresponding query response
- **Multi-agent orchestration:** LangGraph state machine coordinates four distinct processing nodes
- **100% local:** All inference runs on LM Studio â€” no cloud APIs, no data leaves the machine
- **Dynamic ingestion:** Authenticated API clients can upload files at any time; data is additive and deduplicated

---

## 2. System Architecture

### 2.1 Three-Layer Architecture

```
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚  FRONTEND (Streamlit)         ui/app.py                 â”‚
â”‚  - Query Studio               - Knowledge Graph Explorerâ”‚
â”‚  - System health display      - Reasoning trace         â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  BACKEND (FastAPI)            api/server.py              â”‚
â”‚  - REST API endpoints         - Background task runner   â”‚
â”‚  - Health checks              - File validation          â”‚
â”‚  - CORS middleware                                       â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  CORE ENGINE                  core/                      â”‚
â”‚  - config.py     â†’ Settings, prompts, health checks      â”‚
â”‚  - engine.py     â†’ LangGraph state machine               â”‚
â”‚  - ingest.py     â†’ Text extraction, chunking, embedding  â”‚
â”‚  - graph_builder.py â†’ LLM triple extraction, graph merge â”‚
â”‚  - retrievers.py â†’ Vector + Graph retrieval classes      â”‚
â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
â”‚  STORAGE                                                 â”‚
â”‚  - db/chromadb/  â†’ Persisted vector embeddings           â”‚
â”‚  - db/graph.json â†’ Knowledge graph (nodes + edges)       â”‚
â”‚  - uploads/      â†’ Temporary file staging (auto-cleaned) â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
â”‚  EXTERNAL                                                â”‚
â”‚  - LM Studio (localhost:1234) â†’ Gemma 4 E4B + nomic     â”‚
â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### 2.2 Ingestion Workflow

```
An authenticated API client uploads PDF/DOCX
    â”‚
    â–¼
POST /api/upload  (FastAPI validates file type + size)
    â”‚
    â–¼
Background task: _process_uploaded_files()
    â”‚
    â”œâ”€â”€ For each file:
    â”‚   â”œâ”€â”€ Stage 1: extract_text()        â†’ list[Document]
    â”‚   â”œâ”€â”€ Stage 2: chunk_documents()     â†’ list[Document] with chunk_ids
    â”‚   â””â”€â”€ Stage 3: embed_chunks()        â†’ ChromaDB (deduplicated by content hash)
    â”‚
    â””â”€â”€ For each file's chunks:
        â””â”€â”€ Stage 4: build_graph_from_chunks()
            â”œâ”€â”€ extract_triples_from_chunk() â†’ LLM call (Gemma 4, system+user role)
            â”‚   â””â”€â”€ JSON regex extraction from LLM response
            â”œâ”€â”€ merge_triples_into_graph()  â†’ Deduplicate nodes/edges
            â””â”€â”€ save_graph()               â†’ db/graph.json

Processing status is available through GET /api/upload-status for API clients
```

**Stages in detail:**

1. **Text Extraction** (`core/ingest.py`): PDF pages extracted via `pypdf.PdfReader`, DOCX paragraphs grouped into logical sections via `python-docx`. Each page/section becomes a `langchain_core.documents.Document` with metadata (source filename, page number, file type).

2. **Chunking** (`core/ingest.py`): `RecursiveCharacterTextSplitter` splits documents into overlapping chunks (default 1000 chars, 200 overlap). Each chunk gets a deterministic `chunk_id` based on MD5 hash of content â€” this enables deduplication on re-upload.

3. **Embedding** (`core/ingest.py`): Chunks are embedded via LM Studio's `nomic-embed-text-v1.5` model through a custom `LMStudioEmbeddings` wrapper class. Stored in a persisted ChromaDB collection named `local_graphrag`. Existing chunk IDs are checked before embedding to avoid duplicates.

4. **Graph Building** (`core/graph_builder.py`): Each chunk is sent to Gemma 4 E4B with a system prompt instructing triple extraction. The LLM returns a JSON array of `{subject, relation, object}` triples. A regex (`re.search(r'\[.*\]', raw, re.DOTALL)`) extracts JSON from anywhere in the response (handles thinking tokens). Triples are merged into `graph.json` with deduplication by node ID (MD5 of normalized label) and edge tuple `(source_id, relation, target_id)`.

### 2.3 Query Workflow

```
User types question in Streamlit chat
    â”‚
    â–¼
POST /api/query  (FastAPI pre-flight: check LM Studio + DB)
    â”‚
    â–¼
run_query() â†’ LangGraph state machine
    â”‚
    â–¼
Node 1: classify_query
    â”œâ”€â”€ LLM call: Gemma 4 (system=ROUTER_SYSTEM_PROMPT, user=query)
    â”œâ”€â”€ max_tokens=128, temperature=0.2
    â”œâ”€â”€ Response parsed: re.findall(r'\b(vector|graph)\b') â†’ take LAST match
    â””â”€â”€ Output: route="vector" or "graph"
    â”‚
    â–¼
Conditional Edge: route_decision()
    â”œâ”€â”€ if route == "graph" â†’ retrieve_graph
    â””â”€â”€ else â†’ retrieve_vector
    â”‚
    â–¼ (one of two paths)
Node 2a: retrieve_vector                    Node 2b: retrieve_graph
â”œâ”€â”€ VectorRetriever.query()                 â”œâ”€â”€ GraphRetriever.query()
â”œâ”€â”€ ChromaDB similarity_search (top_k=5)    â”œâ”€â”€ Keyword matching on node labels
â”œâ”€â”€ Returns scored document chunks           â”œâ”€â”€ BFS traversal (max 2 hops)
â””â”€â”€ format_context() â†’ string               â””â”€â”€ format_context() â†’ string
    â”‚                                            â”‚
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                     â–¼
Node 3: generate_answer
    â”œâ”€â”€ LLM call: Gemma 4 (system=GENERATOR_SYSTEM_PROMPT+context, user=query)
    â”œâ”€â”€ max_tokens=2048, temperature=0.2
    â””â”€â”€ Output: response=answer string

    â–¼
Answer chunks stream from LM Studio through the existing endpoint to Streamlit
Final route + reasoning_trace metadata is sent when generation completes
```

### 2.4 Gemma 4 Integration in the Pipeline

All three LLM agent roles use Gemma 4 E4B's native `system` role in the messages array:

- **Router Agent:** System message instructs classification with reasoning. The model may emit thinking tokens before the final `vector`/`graph` word. The parser uses `re.findall()` to extract the **last** occurrence.
- **Triple Extractor:** System message instructs step-by-step reasoning before JSON output. The JSON regex scans the **entire** response string, so thinking tokens before the array are harmlessly skipped.
- **Answer Synthesiser:** System message includes the retrieved context and instructs the model to reason through it before answering. The response is used directly (no stripping needed since the answer itself is the output).

---

## 3. Module-by-Module Breakdown

### 3.1 `core/config.py` â€” Centralized Configuration

**Purpose:** Single source of truth for all model names, API endpoints, file paths, prompt templates, inference settings, processing status tracking, and health check utilities.

#### Constants

| Variable | Value | Purpose |
|---|---|---|
| `PROJECT_ROOT` | Auto-detected | Root directory of the project |
| `DATA_DIR` | `PROJECT_ROOT/data` | Raw file storage |
| `DB_DIR` | `PROJECT_ROOT/db` | Database directory |
| `CHROMA_DIR` | `DB_DIR/chromadb` | ChromaDB persistence |
| `GRAPH_PATH` | `DB_DIR/graph.json` | Knowledge graph file |
| `UPLOADS_DIR` | `PROJECT_ROOT/uploads` | Temporary upload staging |
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | LM Studio API endpoint |
| `LM_STUDIO_API_KEY` | `lm-studio` | API key (LM Studio default) |
| `ROUTER_MODEL` | `google/gemma-4-e4b` | Model for routing + extraction |
| `SYNTHESISER_MODEL` | `google/gemma-4-e4b` | Model for answer generation |
| `EMBEDDING_MODEL` | `text-embedding-nomic-embed-text-v1.5` | Embedding model |
| `ROUTER_MAX_TOKENS` | `128` | Max tokens for router |
| `SYNTHESIS_MAX_TOKENS` | `2048` | Max tokens for answer generation |
| `EXTRACTION_MAX_TOKENS` | `512` | Max tokens for triple extraction |
| `CONTEXT_LENGTH` | `4096` | Context window (VRAM-safe) |
| `TEMPERATURE` | `0.2` | LLM sampling temperature |
| `TOP_K` | `5` (env overridable) | Number of retrieval results |
| `CHROMA_COLLECTION` | `local_graphrag` (env overridable) | ChromaDB collection name |
| `CHUNK_SIZE` | `1000` (env overridable) | Characters per chunk |
| `CHUNK_OVERLAP` | `200` (env overridable) | Overlap between chunks |

#### Prompt Templates

- `ROUTER_SYSTEM_PROMPT` â€” Instructs Gemma 4 to classify queries as `vector` or `graph` with reasoning
- `GENERATOR_SYSTEM_PROMPT` â€” Instructs answer generation from context with `{context}` placeholder
- `TRIPLE_EXTRACTION_SYSTEM_PROMPT` â€” Instructs knowledge graph triple extraction with JSON output
- `TRIPLE_EXTRACTION_USER_PROMPT` â€” User message template with `{text}` placeholder

#### Classes

| Class | Methods | Purpose |
|---|---|---|
| `ProcessingStatus` | `get()`, `update(**kwargs)`, `start(total_files)`, `finish()`, `fail(error)` | Thread-safe singleton tracking background ingestion progress |

#### Functions

| Function | Signature | Returns | Purpose |
|---|---|---|---|
| `is_lm_studio_running` | `() -> dict` | `{running, models_available, llm_ready, embed_ready, router_ready}` | Health check â€” queries LM Studio `/models` endpoint |
| `is_db_ready` | `() -> dict` | `{ready, chromadb_exists, graph_json_exists, graph_nodes, graph_edges, ...}` | Checks if ChromaDB and/or graph.json exist on disk |

#### Global Objects

- `client` â€” `OpenAI` instance configured for LM Studio
- `processing_status` â€” Global `ProcessingStatus` singleton

---

### 3.2 `core/ingest.py` â€” Document Ingestion Pipeline

**Purpose:** Extracts text from PDF/DOCX files, splits into overlapping chunks, and embeds into ChromaDB with content-hash deduplication.

#### Functions

| Function | Signature | Returns | Purpose |
|---|---|---|---|
| `extract_text_from_pdf` | `(file_path: Path) -> list[Document]` | One `Document` per page | Uses `pypdf.PdfReader`; metadata includes `source`, `page`, `total_pages`, `file_type` |
| `extract_text_from_docx` | `(file_path: Path) -> list[Document]` | One `Document` per paragraph group | Uses `python-docx`; groups paragraphs separated by empty lines into sections |
| `extract_text` | `(file_path: Path) -> list[Document]` | Dispatched result | Routes to PDF or DOCX extractor based on file extension |
| `chunk_documents` | `(documents: list[Document]) -> list[Document]` | Chunked documents | `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ". ", " ", ""]`; adds MD5-based `chunk_id` to metadata |
| `embed_chunks` | `(chunks: list[Document]) -> int` | Count of new chunks added | Creates/opens ChromaDB collection, checks existing IDs, embeds only new chunks |
| `ingest_file` | `(file_path: Path) -> dict` | Stats dict | Full pipeline for one file: extract â†’ chunk â†’ embed with status updates |
| `ingest_files` | `(file_paths: list[Path]) -> list[dict]` | List of stats dicts | Sequential multi-file ingestion with progress tracking |

#### Key Algorithm: Content-Hash Deduplication

```python
content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:12]
chunk_id = f"{source_filename}_{content_hash}"
```

Before embedding, existing IDs are fetched from ChromaDB via `vectorstore.get(ids=ids)`. Only chunks with new IDs are embedded, preventing duplicate entries on re-upload.

---

### 3.3 `core/graph_builder.py` â€” Knowledge Graph Builder

**Purpose:** Uses Gemma 4 E4B to extract entity-relationship triples from text chunks and merges them into the persistent `graph.json` file.

#### Functions

| Function | Signature | Returns | Purpose |
|---|---|---|---|
| `load_graph` | `() -> dict` | `{nodes: [], edges: []}` | Loads graph from disk; returns empty structure if missing/corrupt |
| `save_graph` | `(graph: dict) -> None` | â€” | Writes graph to disk as indented JSON |
| `_make_node_id` | `(label: str) -> str` | 10-char hex string | Deterministic node ID via `MD5(label.strip().lower())[:10]` |
| `extract_triples_from_chunk` | `(chunk_text: str) -> list[dict]` | List of `{subject, relation, object}` | LLM call with system/user role split; JSON extracted via regex; input capped at 1500 chars |
| `merge_triples_into_graph` | `(triples, source_file, graph) -> dict` | Updated graph | Deduplicates nodes by ID, edges by `(source, relation, target)` tuple |
| `build_graph_from_chunks` | `(chunks: list[Document], source_file: str) -> dict` | Stats dict | Full pipeline: extract triples from all chunks, merge, save |

#### Key Algorithm: JSON Extraction from LLM Response

```python
json_match = re.search(r'\[.*\]', raw, re.DOTALL)
if json_match:
    triples = json.loads(json_match.group())
```

This regex scans the entire response for a JSON array, handling cases where Gemma 4's thinking mode prepends reasoning tokens before the actual JSON output.

---

### 3.4 `core/retrievers.py` â€” Dual Retrieval Logic

**Purpose:** Provides two retriever classes â€” `VectorRetriever` for ChromaDB similarity search and `GraphRetriever` for knowledge graph traversal â€” plus a custom embedding wrapper.

#### Classes

**`LMStudioEmbeddings`**

| Method | Signature | Purpose |
|---|---|---|
| `__init__` | `()` | Sets `self.model = EMBEDDING_MODEL` |
| `embed_documents` | `(texts: list[str]) -> list[list[float]]` | Batch embedding (batch_size=16) via LM Studio |
| `embed_query` | `(text: str) -> list[float]` | Single query embedding |

**`VectorRetriever`**

| Method | Signature | Purpose |
|---|---|---|
| `__init__` | `()` | Opens ChromaDB collection with `LMStudioEmbeddings` |
| `query` | `(text: str, k: Optional[int] = None) -> list[Document]` | Similarity search with relevance scores; default k=`TOP_K` (5) |
| `format_context` | `(documents: list[Document]) -> str` | Formats chunks as numbered context with source/page/score metadata |

**`GraphRetriever`**

| Method | Signature | Purpose |
|---|---|---|
| `__init__` | `()` | Loads graph, builds node index + adjacency list |
| `_load_graph` | `() -> dict` | Reads `graph.json` from disk |
| `_build_indices` | `() -> None` | Builds `_node_index` (by ID and lowercase label) and `_adjacency` (bidirectional) |
| `query` | `(text: str, max_hops: int = 2) -> list[dict]` | Keyword matching on node labels â†’ BFS edge traversal |
| `format_context` | `(triples: list[dict]) -> str` | Formats triples as `â€¢ Subject [type] â€”(relation)â†’ Target [type]` |

#### Key Algorithm: Graph Traversal (BFS with hop limit)

1. Extract keywords from query (words > 2 chars, lowercased)
2. Match keywords against all node labels â†’ seed set
3. For each hop (up to `max_hops=2`): traverse edges from frontier nodes, collect triples, expand frontier
4. Avoid re-visiting seed nodes in subsequent hops

---

### 3.5 `core/engine.py` â€” LangGraph State Machine

**Purpose:** Defines the LangGraph state machine that orchestrates the full query pipeline: classify â†’ route â†’ retrieve â†’ generate.

#### State Schema

```python
class QueryState(TypedDict):
    query: str                     # User's input question
    route: str                     # "vector" or "graph"
    context: str                   # Retrieved context
    response: str                  # Final LLM answer
    reasoning_trace: list[dict]    # Step-by-step audit trail
```

#### Node Functions

| Function | Reads | Writes | LLM Call |
|---|---|---|---|
| `classify_query(state)` | `query` | `route`, `reasoning_trace` | Gemma 4, `ROUTER_MAX_TOKENS=128` |
| `retrieve_vector(state)` | `query` | `context`, `reasoning_trace` | None (embedding only) |
| `retrieve_graph(state)` | `query` | `context`, `reasoning_trace` | None (local traversal) |
| `generate_answer(state)` | `query`, `context` | `response`, `reasoning_trace` | Gemma 4, `SYNTHESIS_MAX_TOKENS=2048` |

#### Routing Function

```python
def route_decision(state: QueryState) -> Literal["retrieve_vector", "retrieve_graph"]:
    if state.get("route") == "graph":
        return "retrieve_graph"
    return "retrieve_vector"
```

#### Graph Structure

```
START â†’ classify_query â”€â”¬â”€â†’ retrieve_vector â”€â†’ generate_answer â†’ END
                        â””â”€â†’ retrieve_graph  â”€â†’ generate_answer â†’ END
```

#### Key Functions

| Function | Signature | Returns | Purpose |
|---|---|---|---|
| `build_graph` | `() -> StateGraph` | Compiled LangGraph | Constructs and compiles the state machine |
| `run_query` | `(query: str) -> dict` | Final state dict | Entry point: initializes state, invokes graph, returns result |

#### Global Objects

- `graph` â€” Compiled LangGraph singleton (created at import time)

---

### 3.6 `api/server.py` â€” FastAPI Backend

**Purpose:** REST API layer that serves the LangGraph engine, handles file uploads with background processing, and provides health check endpoints.

#### Pydantic Schemas

| Schema | Fields | Purpose |
|---|---|---|
| `QueryRequest` | `query: str` (1â€“2000 chars), `conversation_id: str | None` | Query input and optional conversation |
| `RenameConversationRequest` | `title: str` (1â€“80 chars, trimmed) | Conversation rename |
| `QueryResponse` | `response: str`, `route: str`, `reasoning_trace: list[dict]` | Query result |
| `HealthResponse` | `status: str`, `lm_studio: dict`, `database: dict` | Health status |
| `UploadResponse` | `message: str`, `files_accepted: list[str]`, `files_rejected: list[dict]` | Upload result |
| `ProcessingStatusResponse` | `is_processing`, `stage`, `progress`, `current_file`, `total_files`, `processed_files`, `message`, `error` | Processing progress |

#### Functions

| Function | Purpose |
|---|---|
| `_process_uploaded_files(file_paths)` | Background task: extract â†’ chunk â†’ embed â†’ build graph; cleans up uploaded files on completion |
| `lifespan(app)` | Startup banner with model/DB status; shutdown message |

---

### 3.7 `ui/app.py` â€” Streamlit Frontend

**Purpose:** Streamlit interface for Query Studio and Knowledge Graph Explorer, with authentication, system health display, and query reasoning traces. Document upload and processing controls are not exposed in the UI.

#### Key Sections

1. **Custom CSS**: Shared styling for navigation, status badges, query traces, graph, chat, and sidebar
2. **Session State**: `messages`, `traces`, `active_conversation_id`, cached `conversations`, authentication state, `prompt_seed`, and `graph_cache`
3. **Sidebar**: New chat, searchable recency-grouped conversation history, rename/delete actions, signed-in user, and collapsed system status
4. **Main Area**: Global system status, Query Studio, and Knowledge Graph Explorer

#### API Helper Functions

| Function | Purpose |
|---|---|
| `check_health()` | GET `/api/health` â€” supplies global system status |
| `refresh_conversations()` | GET `/api/conversations` â€” refreshes the per-session sidebar cache |
| `load_conversation(conversation_id)` | GET `/api/conversations/{id}` â€” loads messages and assistant reasoning traces |
| `load_graph_data()` | Loads and caches the persisted knowledge graph for the explorer |


---

## 4. API Reference

### 4.1 `GET /api/health`

**Purpose:** Check LM Studio connectivity and database readiness.

**Request:** No body or parameters.

**Response (200):**
```json
{
  "status": "healthy",
  "lm_studio": {
    "running": true,
    "models_available": ["google/gemma-4-e4b", "text-embedding-nomic-embed-text-v1.5"],
    "llm_ready": true,
    "embed_ready": true,
    "router_ready": true
  },
  "database": {
    "ready": true,
    "chromadb_exists": true,
    "graph_json_exists": true,
    "graph_nodes": 61,
    "graph_edges": 37,
    "chroma_path": "D:\\6th sem\\major_project\\db\\chromadb",
    "graph_path": "D:\\6th sem\\major_project\\db\\graph.json"
  }
}
```

**Status values:** `"healthy"` (LM Studio connected) or `"degraded"` (LM Studio down).

---

### 4.2 `POST /api/upload`

**Purpose:** Upload PDF/DOCX files for background ingestion.

**Request:** `multipart/form-data` with field `files` (one or more files).

**Validation rules:**
- Allowed extensions: `.pdf`, `.docx`, `.doc`
- Max file size: 100 MB per file
- Rejects upload if a processing job is already running (409)
- Rejects upload if LM Studio is not running (503)

**Response (200):**
```json
{
  "message": "âœ… 2 file(s) accepted and queued for processing.",
  "files_accepted": ["notes.pdf", "syllabus.docx"],
  "files_rejected": [{"file": "image.png", "reason": "Unsupported type: .png"}]
}
```

**Error codes:**

| Code | Condition |
|---|---|
| 400 | No valid files to process |
| 409 | Processing job already running |
| 503 | LM Studio not running |

---

### 4.3 `GET /api/upload-status`

**Purpose:** Poll background file processing progress for API clients.

**Response (200):**
```json
{
  "is_processing": true,
  "stage": "embedding",
  "progress": 0.35,
  "current_file": "notes.pdf",
  "total_files": 2,
  "processed_files": 1,
  "message": "ðŸ§® Embedding 12 chunks from notes.pdf...",
  "error": null
}
```

**Stage values:** `"starting"`, `"extracting"`, `"chunking"`, `"embedding"`, `"graphing"`, `"done"`, `"error"`

**Progress:** Float from `0.0` to `1.0`. First 50% covers extraction + embedding; second 50% covers graph building.

---

### 4.4 `POST /api/query`

**Purpose:** Run a question through the full LangGraph pipeline.

**Request:**
```json
{
  "query": "What is local graph rag?",
  "conversation_id": null
}
```
- `query`: string, 1â€“2000 characters
- `conversation_id`: optional conversation id; omit or pass `null` to create a conversation on the first message

**Response (200):**
```json
{
  "response": "Local GraphRAG is a multi-agent system that...",
  "route": "vector",
  "reasoning_trace": [
    {
      "step": "classify_query",
      "route_chosen": "vector",
      "llm_raw_output": "vector",
      "latency_sec": 3.42,
      "description": "Query classified as 'vector' path (definitions/facts â†’ Vector DB)"
    },
    {
      "step": "retrieve_vector",
      "num_chunks": 5,
      "sources": [
        {
          "source": "notes.pdf",
          "page": 3,
          "score": 0.8234,
          "preview": "Local GraphRAG is a fully local..."
        }
      ],
      "latency_sec": 0.58,
      "description": "Retrieved 5 chunks from ChromaDB"
    },
    {
      "step": "generate_answer",
      "latency_sec": 12.31,
      "description": "Generated answer using google/gemma-4-e4b (12.31s)"
    }
  ]
}
```

**Streaming response:** Send `Accept: text/event-stream` to receive server-sent events
from the same endpoint. `token` events contain an incremental text chunk:

```text
event: token
data: {"text":"Local GraphRAG"}
```

The final `complete` event contains the selected route, full reasoning trace,
`conversation_id`, and conversation `title`. The user message is stored before
generation; the assistant response and trace are stored only after successful
completion. Conversation history is not passed back to the engine.
An `error` event reports generation failures. Requests that do not ask for
`text/event-stream` continue to receive the JSON response documented above.

**Error codes:**

| Code | Condition |
|---|---|
| 503 | LM Studio not running, or no documents ingested |
| 500 | Engine error (LLM failure, retrieval error, etc.) |

---

### 4.5 `GET /api/conversations`

**Purpose:** List the signed-in user's conversations, newest activity first.

**Authentication:** Bearer token required.

**Response (200):**
```json
[
  {"id": "uuid4hex", "title": "What is GraphRAG?", "updated_at": 1792622400}
]
```

### 4.6 `GET /api/conversations/{id}`

**Purpose:** Load an owned conversation, including messages and assistant
reasoning traces. A conversation owned by another user is indistinguishable
from a missing conversation and returns 404.

**Response (200):**
```json
{
  "id": "uuid4hex",
  "title": "What is GraphRAG?",
  "messages": [
    {"role": "user", "content": "What is GraphRAG?", "reasoning_trace": []},
    {"role": "assistant", "content": "GraphRAG combines...", "reasoning_trace": []}
  ]
}
```

### 4.7 `PATCH /api/conversations/{id}`

**Purpose:** Rename an owned conversation. The trimmed title must contain
1–80 characters.

**Request:** `{"title": "A concise title"}`

**Response (200):** `{"id": "uuid4hex", "title": "A concise title", "updated_at": 1792622400}`

**Errors:** 404 if the conversation does not exist or belongs to another user;
422 if the title is blank or exceeds 80 characters.

### 4.8 `DELETE /api/conversations/{id}`

**Purpose:** Delete an owned conversation and its messages.

**Response:** 204 No Content.

**Errors:** 404 if the conversation does not exist or belongs to another user.

### 4.9 `GET /`

**Purpose:** Root info endpoint. Returns API version and available endpoint paths.

```json
{
  "message": "Local GraphRAG API v2.0",
  "docs": "/docs",
  "endpoints": {
    "health": "GET /api/health",
    "upload": "POST /api/upload",
    "upload_status": "GET /api/upload-status",
    "query": "POST /api/query",
    "conversations": "GET /api/conversations"
  }
}
```

---

## 5. Data Schemas

### 5.1 ChromaDB Collection

| Property | Value |
|---|---|
| Collection name | `local_graphrag` |
| Persist directory | `db/chromadb/` |
| Embedding model | `nomic-embed-text-v1.5` (768 dimensions) |
| Embedding wrapper | `LMStudioEmbeddings` class |
| Document ID format | `{source_filename}_{md5_hash[:12]}` |

**Document metadata fields:**

| Field | Type | Source |
|---|---|---|
| `source` | `str` | Original filename |
| `page` | `int` | Page number (PDF only) |
| `total_pages` | `int` | Total pages in PDF |
| `section` | `int` | Section number (DOCX only) |
| `file_type` | `str` | `"pdf"` or `"docx"` |
| `chunk_id` | `str` | Content-hash based ID |
| `relevance_score` | `float` | Added at query time by retriever |

---

### 5.2 `graph.json` Structure

```json
{
  "nodes": [
    {
      "id": "a1b2c3d4e5",
      "label": "Machine Learning",
      "type": "concept",
      "properties": {
        "source": "notes.pdf"
      }
    }
  ],
  "edges": [
    {
      "source": "a1b2c3d4e5",
      "relation": "prerequisite_of",
      "target": "f6g7h8i9j0"
    }
  ]
}
```

**Node schema:**

| Field | Type | How generated |
|---|---|---|
| `id` | `str` (10 hex chars) | `MD5(label.strip().lower())[:10]` |
| `label` | `str` | Extracted entity name |
| `type` | `str` | Always `"concept"` |
| `properties.source` | `str` | Source filename |

**Edge schema:**

| Field | Type | Description |
|---|---|---|
| `source` | `str` | Source node ID |
| `relation` | `str` | Relationship type |
| `target` | `str` | Target node ID |

**Relation types used:** `is_a`, `part_of`, `prerequisite_of`, `related_to`, `defines`, `contains`, `uses`, `example_of`

---

### 5.3 ProcessingStatus Fields

| Field | Type | Values / Description |
|---|---|---|
| `is_processing` | `bool` | `True` while processing |
| `current_file` | `str` | Name of file being processed |
| `stage` | `str` | `"starting"`, `"extracting"`, `"chunking"`, `"embedding"`, `"graphing"`, `"done"`, `"error"` |
| `progress` | `float` | `0.0` to `1.0` |
| `total_files` | `int` | Number of files in batch |
| `processed_files` | `int` | Files completed so far |
| `message` | `str` | Human-readable status message |
| `files_ingested` | `list` | History of processed file stats |
| `error` | `str or None` | Error message if failed |
| `started_at` | `str or None` | ISO timestamp of start time |

---

### 5.4 Conversation History (SQLite)

Conversation data is stored in `db/auth.sqlite3` beside accounts and sessions.
Both tables use foreign keys with cascading deletion; chat API operations always
scope reads and writes to the authenticated user.

| Table | Fields | Purpose |
|---|---|---|
| `conversations` | `id` (UUID4 hex PK), `user_id` (FK to users), `title`, `created_at`, `updated_at` (Unix seconds) | Per-user conversation metadata; indexed by `(user_id, updated_at DESC)` |
| `messages` | `id` (autoincrement PK), `conversation_id` (FK), `role` (`user` or `assistant`), `content`, `trace_json`, `created_at` (Unix seconds) | Ordered turns and assistant provenance traces |

### 5.5 QueryState (LangGraph)

| Field | Type | Set By | Purpose |
|---|---|---|---|
| `query` | `str` | `run_query()` (initial) | The user's question |
| `route` | `str` | `classify_query` | `"vector"` or `"graph"` |
| `context` | `str` | `retrieve_vector` or `retrieve_graph` | Retrieved context string |
| `response` | `str` | `generate_answer` | Final LLM answer |
| `reasoning_trace` | `list[dict]` | All nodes (appended) | Audit trail for the UI |

---

## 6. LangGraph Pipeline â€” Deep Dive

### 6.1 State Machine Diagram

```
         â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
         â”‚    START      â”‚
         â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
                â”‚
                â–¼
    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
    â”‚   classify_query      â”‚  â† LLM call (Gemma 4, 128 tokens)
    â”‚   Reads: query        â”‚
    â”‚   Writes: route       â”‚
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                â”‚
        route_decision()
       â”Œâ”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”
       â”‚                â”‚
  route=="vector"  route=="graph"
       â”‚                â”‚
       â–¼                â–¼
â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â” â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
â”‚retrieve_vectorâ”‚ â”‚retrieve_graph â”‚
â”‚Reads: query   â”‚ â”‚Reads: query   â”‚
â”‚Writes: contextâ”‚ â”‚Writes: contextâ”‚
â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜ â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜
       â”‚                â”‚
       â””â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”˜
               â”‚
               â–¼
    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
    â”‚   generate_answer     â”‚  â† LLM call (Gemma 4, 2048 tokens)
    â”‚   Reads: query,contextâ”‚
    â”‚   Writes: response    â”‚
    â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
                â”‚
                â–¼
         â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
         â”‚     END       â”‚
         â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

### 6.2 Node Details

**Node: `classify_query`**

- **Input state consumed:** `query`
- **Processing:** Calls Gemma 4 with `ROUTER_SYSTEM_PROMPT` as system message and the raw query as user message
- **API call:** `client.chat.completions.create(model=ROUTER_MODEL, messages=[{system}, {user}], max_tokens=128, temperature=0.2)`
- **Response parsing:** `re.findall(r'\b(vector|graph)\b', raw_route, re.IGNORECASE)` â†’ takes `matches[-1].lower()` or defaults to `"vector"`
- **Output state written:** `route` (string), `reasoning_trace` (appended with step details including raw LLM output)
- **Thinking token handling:** The regex approach extracts the LAST occurrence, so any reasoning text containing "vector" or "graph" in the middle is overridden by the final classification word

**Node: `retrieve_vector`**

- **Input state consumed:** `query`
- **Processing:** Creates `VectorRetriever()`, calls `query()` which performs `similarity_search_with_relevance_scores` against ChromaDB with `k=TOP_K` (5), then formats results into a context string
- **No LLM call** â€” uses embedding model only (via ChromaDB internally)
- **Output state written:** `context` (formatted string with chunk sources/pages/scores), `reasoning_trace` (appended with chunk metadata and source previews)

**Node: `retrieve_graph`**

- **Input state consumed:** `query`
- **Processing:** Creates `GraphRetriever()`, calls `query()` which performs keyword matching + BFS traversal (max 2 hops), then formats triples
- **No LLM call** â€” pure in-memory graph traversal
- **Output state written:** `context` (formatted triple string), `reasoning_trace` (appended with triple data, capped at 10 for display)

**Node: `generate_answer`**

- **Input state consumed:** `query`, `context`
- **Processing:** Formats `GENERATOR_SYSTEM_PROMPT` with the retrieved context, calls Gemma 4
- **API call:** `client.chat.completions.create(model=SYNTHESISER_MODEL, messages=[{system+context}, {user=query}], max_tokens=2048, temperature=0.2)`
- **Response parsing:** Direct `res.choices[0].message.content.strip()` â€” no special parsing needed
- **Output state written:** `response` (answer string), `reasoning_trace` (appended with latency)

### 6.3 Conditional Routing

The `route_decision` function reads `state["route"]`:
- If `"graph"` â†’ returns `"retrieve_graph"` (node name)
- Otherwise â†’ returns `"retrieve_vector"` (default fallback)

This is registered as a conditional edge from `classify_query` with a mapping dict that maps return values to node names.

---

## 7. Gemma 4 Integration Details

### 7.1 Model Configuration

| Setting | Value | Rationale |
|---|---|---|
| LM Studio model string | `google/gemma-4-e4b` | As reported by LM Studio's `/models` endpoint |
| Quantization | Q4_K_M GGUF | Fits within 4GB VRAM |
| Native context window | 128K tokens | Gemma 4 E4B specification |
| Configured context length | 4096 tokens | Capped for 4GB VRAM safety |
| API protocol | OpenAI-compatible | LM Studio serves at `localhost:1234/v1` |

### 7.2 System Role Usage Per Agent

**Router Agent (`classify_query`):**
- System role instructs it as a "query classification agent"
- Explicitly describes classification criteria (vector vs graph) with examples
- Instructs reasoning before output
- Demands final answer be a single word on its own line
- `max_tokens=128` (only needs one word + optional thinking)
- `temperature=0.2`

**Triple Extractor (`extract_triples_from_chunk`):**
- System role instructs it as a "knowledge graph construction agent"
- Lists extraction rules and valid relationship types
- Instructs step-by-step reasoning before extraction
- Demands output as a JSON array only
- `max_tokens=512`
- `temperature=0.0` (deterministic for structured output)

**Answer Synthesiser (`generate_answer`):**
- System role instructs it as "Local GraphRAG, an expert academic document QA assistant"
- Provides the retrieved context directly in the system message
- Instructs reasoning through context before answering
- Rules: context-only answers, cite sources, admit insufficient info
- `max_tokens=2048`
- `temperature=0.2`

### 7.3 Thinking Mode Token Handling

Gemma 4 may emit reasoning/thinking tokens before the final answer. Each agent handles this differently:

| Agent | Parsing Strategy | Implementation |
|---|---|---|
| Router | Regex: extract last `vector`/`graph` word | `re.findall(r'\b(vector\|graph)\b', raw, re.IGNORECASE)[-1]` |
| Triple Extractor | Regex: find JSON array anywhere in response | `re.search(r'\[.*\]', raw, re.DOTALL)` |
| Answer Synthesiser | No stripping needed | Response used directly (thinking is part of the answer quality) |

### 7.4 Embedding Model

| Setting | Value |
|---|---|
| Model | `text-embedding-nomic-embed-text-v1.5` |
| Dimensions | 768 |
| Batch size | 16 (in `LMStudioEmbeddings.embed_documents`) |
| Usage | ChromaDB indexing + similarity search |

This model is **separate** from Gemma 4 and runs concurrently in LM Studio.

---

## 8. Setup & Installation Guide

### 8.1 Prerequisites

| Requirement | Minimum |
|---|---|
| Python | 3.11+ |
| GPU | Any with 4GB+ VRAM (tested: AMD Radeon RX 6500M) |
| RAM | 8GB+ system RAM |
| LM Studio | Latest version |
| OS | Windows (tested), Linux/macOS compatible |

### 8.2 LM Studio Setup

1. **Download LM Studio** from [lmstudio.ai](https://lmstudio.ai)

2. **Download models:**
   - Search for `gemma-4-e4b` â†’ download the **Q4_K_M GGUF** quantization (~3GB)
   - Search for `nomic-embed-text-v1.5` â†’ download the default quantization (~250MB)

3. **Configure in LM Studio:**
   - Context Length: **4096** (match `CONTEXT_LENGTH` in config)
   - GPU Offload: **All layers** to GPU
   - Enable Flash Attention if available

4. **Start the server:**
   - Load both models in LM Studio
   - Start the local server (should be on `localhost:1234` by default)
   - Verify: visit `http://localhost:1234/v1/models` in a browser â€” should list both models

### 8.3 Python Environment

```bash
# Clone / navigate to project
cd "D:\6th sem\major_project"

# Create virtual environment
python -m venv .venv

# Activate
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

**Dependencies (`requirements.txt`):**
```
openai, langchain-chroma, langchain-core, langchain-text-splitters,
langgraph, chromadb, fastapi, uvicorn[standard], streamlit, pypdf,
python-docx, python-multipart, requests, pydantic, numpy
```

### 8.4 Running the System

**Terminal 1 â€” API Server:**
```bash
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000
```

Expected startup banner:
```
============================================================
  Local GraphRAG â€” API Server Starting
============================================================
  LM Studio:   âœ… Connected
  Router:      âœ… google/gemma-4-e4b
  Synthesiser: âœ… google/gemma-4-e4b
  Embeddings:  âœ… text-embedding-nomic-embed-text-v1.5
  ChromaDB:    ðŸ“­ Empty (upload files to start)
  Graph:       ðŸ“­ Empty
============================================================
```

**Terminal 2 â€” Streamlit UI:**
```bash
streamlit run ui/app.py
```

Open `http://localhost:8501` in your browser.

### 8.5 Verification Checklist

- [ ] LM Studio running on `localhost:1234`
- [ ] Both models loaded (check `http://localhost:1234/v1/models`)
- [ ] API server started without errors on port 8000
- [ ] Streamlit UI shows green "â— LM Studio" badge
- [ ] Upload a test PDF â†’ progress bar completes
- [ ] Ask a test question â†’ get answer with reasoning trace


---

## 9. Configuration Reference

All configurable values are defined in `core/config.py`. This is the single source of truth.

### 9.1 Path Constants

| Variable | Default Value | Type | What It Controls |
|---|---|---|---|
| `PROJECT_ROOT` | Auto-detected via `__file__` | `Path` | Root directory for all relative paths |
| `DATA_DIR` | `PROJECT_ROOT/data` | `Path` | Raw file storage directory |
| `DB_DIR` | `PROJECT_ROOT/db` | `Path` | Parent directory for all databases |
| `CHROMA_DIR` | `DB_DIR/chromadb` | `Path` | ChromaDB persistence directory |
| `GRAPH_PATH` | `DB_DIR/graph.json` | `Path` | Knowledge graph file location |
| `UPLOADS_DIR` | `PROJECT_ROOT/uploads` | `Path` | Temporary upload staging directory |

### 9.2 LM Studio Connection

| Variable | Default Value | Type | User May Change? |
|---|---|---|---|
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1` | `str` | Yes â€” if LM Studio runs on a different port |
| `LM_STUDIO_API_KEY` | `lm-studio` | `str` | Rarely â€” LM Studio default |

### 9.3 Model Names

| Variable | Default Value | Type | User May Change? |
|---|---|---|---|
| `ROUTER_MODEL` | `google/gemma-4-e4b` | `str` | Yes â€” must match LM Studio's model ID exactly |
| `SYNTHESISER_MODEL` | `google/gemma-4-e4b` | `str` | Yes â€” can be a different model for generation |
| `EMBEDDING_MODEL` | `text-embedding-nomic-embed-text-v1.5` | `str` | Yes â€” but changing requires re-embedding all documents |

### 9.4 Inference Settings

| Variable | Default Value | Type | User May Change? | Notes |
|---|---|---|---|---|
| `ROUTER_MAX_TOKENS` | `128` | `int` | Yes | Keep low for speed; 128 handles thinking overhead |
| `SYNTHESIS_MAX_TOKENS` | `2048` | `int` | Yes | Increase for longer answers; watch VRAM |
| `EXTRACTION_MAX_TOKENS` | `512` | `int` | Yes | Small JSON arrays don't need more |
| `CONTEXT_LENGTH` | `4096` | `int` | Yes | Must match LM Studio setting; max 16384 on 4GB VRAM |
| `TEMPERATURE` | `0.2` | `float` | Yes | Lower = more deterministic; higher = more creative |

### 9.5 Retrieval Settings

| Variable | Default Value | Type | Env Override | User May Change? |
|---|---|---|---|---|
| `TOP_K` | `5` | `int` | `TOP_K` | Yes â€” more chunks = richer context but slower |
| `CHROMA_COLLECTION` | `local_graphrag` | `str` | `CHROMA_COLLECTION` | Yes â€” changing creates a new collection |

### 9.6 Chunking Settings

| Variable | Default Value | Type | Env Override | User May Change? |
|---|---|---|---|---|
| `CHUNK_SIZE` | `1000` | `int` | `CHUNK_SIZE` | Yes â€” larger = fewer chunks, less granular retrieval |
| `CHUNK_OVERLAP` | `200` | `int` | `CHUNK_OVERLAP` | Yes â€” overlap ensures context continuity between chunks |

---

## 10. Known Limitations

| # | Limitation | Why It Exists | Fix / Workaround |
|---|---|---|---|
| 1 | **Single-model inference** â€” Router, extractor, and synthesiser share one Gemma 4 instance | LM Studio on 4GB VRAM can only load one LLM at a time | Use a GPU with more VRAM to load separate specialized models for each role |
| 2 | **Slow inference on 4GB VRAM** â€” Queries can take 15â€“60+ seconds | Q4_K_M quantization + limited GPU memory = slow token generation | Reduce `SYNTHESIS_MAX_TOKENS`, reduce `CONTEXT_LENGTH`, or upgrade GPU |
| 3 | **No streaming responses** â€” UI waits for full answer before displaying | The API returns the complete response; no SSE/WebSocket streaming implemented | Implement streaming via FastAPI's `StreamingResponse` and Streamlit's `st.write_stream` |
| 4 | **Sequential file processing** â€” Multiple files are ingested one at a time | Graph building requires sequential LLM calls per chunk | Could parallelize embedding (Stage 2) while keeping graph building sequential |
| 5 | **Keyword-based graph retrieval** â€” Graph search uses simple string matching | No embedding-based similarity on graph nodes | Add embedding similarity for node matching, or use a proper graph database (Neo4j) |
| 6 | **No role-based authorization** â€” authenticated users share the local GraphRAG corpus | Single-user/local deployment model | Add roles and per-user data isolation before multi-user network deployment |
| 7 | **No document deletion** â€” Once ingested, data cannot be removed | No delete endpoint or UI; ChromaDB and graph.json are append-only | Implement a `/api/delete` endpoint that removes chunks by source filename and prunes graph nodes |
| 8 | **Fixed chunk size** â€” Same chunking parameters for all file types | PDF pages and DOCX sections have different optimal chunk sizes | Add per-file-type chunking configuration |
| 9 | **No conversation memory** â€” Each query is independent | LangGraph state is not persisted between queries | Add a conversation history buffer to the `QueryState` and system prompt |
| 10 | **Graph traversal not weighted** â€” All edges treated equally | No relevance scoring on graph triples | Add edge weights based on extraction confidence or frequency |
| 11 | **MD5 node IDs may collide** â€” 10-character hex truncation | `MD5[:10]` gives ~40 bits = collision at ~1M nodes | Use full MD5 or SHA-256 for larger knowledge bases |
| 12 | **No PDF table/image extraction** â€” Only text is extracted | `pypdf` extracts text only, not tables or images | Use `pymupdf` or `unstructured` for richer extraction |
| 13 | **Uploaded files deleted after processing** â€” No re-processing possible | Files are cleaned up in the `finally` block of `_process_uploaded_files` | Keep original files in a permanent storage directory |

---

## 11. Future Enhancements

| # | Enhancement | Module(s) Affected | Complexity | Benefit |
|---|---|---|---|---|
| 1 | **Streaming responses** â€” Show tokens as they generate | `engine.py`, `server.py`, `app.py` | Medium | Dramatically improves perceived latency; users see partial answers immediately |
| 2 | **Conversation memory** â€” Multi-turn chat with context | `engine.py`, `config.py` | Medium | Enables follow-up questions ("What about X?") without restating context |
| 3 | **Hybrid retrieval fusion** â€” Combine vector + graph results for every query | `engine.py`, `retrievers.py` | Medium | Better answer quality by always considering both sources |
| 4 | **Document management UI** â€” View, delete, re-process uploaded documents | `server.py`, `app.py`, `ingest.py` | Medium | Users can manage their knowledge base without touching files |
| 5 | **Neo4j integration** â€” Replace `graph.json` with a proper graph database | `graph_builder.py`, `retrievers.py`, `config.py` | High | Scalable graph queries, ACID transactions, visualization tools |
| 6 | **Semantic graph search** â€” Embed node labels for similarity matching | `retrievers.py` | Medium | Better graph retrieval for paraphrased queries |
| 7 | **Multi-model routing** â€” Use a small model for routing, larger for generation | `config.py`, `engine.py` | Low | Faster routing with a lightweight model while keeping quality for generation |
| 8 | **PDF table extraction** â€” Extract structured tables from PDFs | `ingest.py` | Medium | Richer knowledge base for technical documents with tables |
| 9 | **Export knowledge graph** â€” Visualize the graph in the UI | `app.py`, `server.py` | Medium | Users can see the knowledge graph structure and verify extraction quality |
| 10 | **Batch query API** â€” Process multiple questions at once | `server.py`, `engine.py` | Low | Useful for automated testing and evaluation |
| 11 | **Caching layer** â€” Cache LLM responses for repeated queries | `engine.py` | Low | Instant responses for previously-asked questions |
| 12 | **Configurable system prompts via UI** â€” Let users edit prompts in the sidebar | `app.py`, `config.py` | Low | Power users can tune agent behavior without editing code |
| 13 | **Evaluation framework** â€” Automated RAG quality metrics | New module | High | Measure retrieval precision, answer faithfulness, and routing accuracy |

---

## 12. Glossary

| Term | Definition |
|---|---|
| **RAG** | Retrieval-Augmented Generation â€” a technique where an LLM generates answers grounded in retrieved documents rather than relying solely on its training data |
| **GraphRAG** | An extension of RAG that adds a knowledge graph as a second retrieval source alongside vector search, enabling relationship-aware question answering |
| **LangGraph** | A Python library by LangChain for building stateful, multi-step LLM applications as directed graphs with nodes (processing steps) and edges (transitions) |
| **ChromaDB** | An open-source vector database that stores document embeddings and supports similarity search; used here for the vector retrieval path |
| **Knowledge Graph** | A structured representation of knowledge as a network of entities (nodes) connected by relationships (edges); stored as `graph.json` in this project |
| **Triple** | A unit of knowledge in a knowledge graph: `(subject, relation, object)` â€” e.g., `(Machine Learning, prerequisite_of, Deep Learning)` |
| **Embedding** | A dense numerical vector representation of text, where semantically similar texts have similar vectors; generated by `nomic-embed-text-v1.5` |
| **Vector Similarity** | A measure of how close two embedding vectors are in high-dimensional space; used to find relevant document chunks for a query |
| **Query Routing** | The process of classifying a user's question to determine which retrieval backend (vector DB or knowledge graph) should handle it |
| **Reasoning Trace** | A step-by-step audit trail of the LangGraph pipeline execution, showing which path was taken, what was retrieved, and how long each step took |
| **MoE (Mixture of Experts)** | A model architecture where different "expert" sub-networks specialize in different tasks; Gemma 4 E4B uses this to achieve high quality in a small model |
| **Quantization** | The process of reducing a model's numerical precision (e.g., from 16-bit to 4-bit) to reduce VRAM usage and increase inference speed, with minimal quality loss |
| **GGUF** | A file format for storing quantized LLM weights, widely supported by inference engines like LM Studio, llama.cpp, and Ollama |
| **Q4_K_M** | A specific 4-bit quantization scheme that balances model quality and memory efficiency; recommended for 4GB VRAM GPUs |
| **System Role** | A message type in the OpenAI chat API format that sets the LLM's persona, instructions, and behavioral constraints; Gemma 4 natively supports this |
| **Thinking Mode** | A capability of Gemma 4 where the model internally reasons step-by-step before producing its final answer; may emit reasoning tokens in the output |
| **LM Studio** | A desktop application for running LLMs locally with an OpenAI-compatible API; serves models on `localhost:1234` |
| **FastAPI** | A modern Python web framework for building REST APIs with automatic OpenAPI documentation and async support |
| **Streamlit** | A Python framework for building data-focused web applications with minimal frontend code; used for the chat UI |
| **Pydantic** | A Python data validation library used by FastAPI for request/response schema validation |

---

## Files Read for This Documentation

All documentation was generated by reading the actual source code of every file in the project:

| File | Lines | Purpose |
|---|---|---|
| `core/config.py` | 230 | Configuration, prompts, health checks |
| `core/ingest.py` | 256 | Text extraction, chunking, embedding |
| `core/graph_builder.py` | 217 | Triple extraction, graph merge |
| `core/retrievers.py` | 247 | Vector + Graph retrieval classes |
| `core/engine.py` | 256 | LangGraph state machine |
| `api/server.py` | 345 | FastAPI backend |
| `ui/app.py` | 624 | Streamlit frontend |
| `requirements.txt` | 16 | Python dependencies |
| `README.md` | 96 | Project README |

**Nothing was assumed or fabricated.** Every function name, parameter, return type, constant value, prompt template, and architectural detail documented above was verified against the actual codebase as of the documentation generation date.
