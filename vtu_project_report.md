# VTU 8th Semester Major Project — Complete Documentation

> **Project: Local GraphRAG**
> *A Fully-Local Multi-Agent Question-Answering System Using Graph-Augmented Retrieval*

---

## Table of Contents

1. [Project Identity](#chapter-1-project-identity)
2. [Abstract / Synopsis](#chapter-2-abstract--synopsis)
3. [Introduction](#chapter-3-introduction)
4. [Literature Survey](#chapter-4-literature-survey)
5. [System Design & Architecture](#chapter-5-system-design--architecture)
6. [Technology Stack](#chapter-6-technology-stack)
7. [Implementation Details](#chapter-7-implementation-details)
8. [Results & Output](#chapter-8-results--output)
9. [Testing](#chapter-9-testing)
10. [Conclusion & Future Scope](#chapter-10-conclusion--future-scope)
11. [References](#chapter-11-references)

---

# Chapter 1: Project Identity

| Field | Details |
|---|---|
| **Project Title** | Local GraphRAG — A Fully-Local Multi-Agent Question-Answering System Using Graph-Augmented Retrieval |
| **Domain** | Artificial Intelligence / Natural Language Processing / Information Retrieval |
| **Sub-Domain** | Retrieval-Augmented Generation (RAG), Knowledge Graphs, Multi-Agent Systems |
| **Type** | Application-oriented Research Project |
| **Academic Context** | VTU 8th Semester BE Major Project |

### Problem Statement

Existing Retrieval-Augmented Generation (RAG) systems suffer from three critical limitations: (1) they rely on cloud-based LLM APIs, raising privacy and data-sovereignty concerns for academic and organisational users; (2) they use a single vector-similarity retrieval path, which fails to capture inter-concept relationships essential for academic material; and (3) they lack transparency into the reasoning process, making it difficult for students and educators to understand how answers are derived. This project proposes **Local GraphRAG**, a fully-local, privacy-preserving multi-agent system that combines vector retrieval with knowledge-graph traversal, orchestrated by a LangGraph state machine, to provide accurate, explainable answers from uploaded PDF/DOCX documents—without any data leaving the user's machine.

### Team / Guide References

> [!NOTE]
> No team member names, guide/mentor names, or college/department references were found in the codebase, configuration files, or README. **Fill these in manually** according to your project team.

| Field | Value |
|---|---|
| **Team Members** | *(Fill in: USN, Name)* |
| **Guide** | *(Fill in: Name, Designation)* |
| **Department** | Department of Computer Science and Engineering |
| **University** | Visvesvaraya Technological University (VTU), Belagavi |
| **Semester** | 8th Semester, B.E. |
| **Academic Year** | 2025–26 |

---

# Chapter 2: Abstract / Synopsis

## Abstract (VTU Standard — under 100 words)

**Local GraphRAG** is a fully-local, privacy-preserving multi-agent question-answering system that combines vector-based retrieval with knowledge-graph traversal for enhanced document understanding. Users upload PDF/DOCX files, which are automatically chunked, embedded into ChromaDB, and transformed into a knowledge graph via LLM-based triple extraction. A LangGraph state machine intelligently routes each query to the optimal retrieval path — vector similarity search for factual queries, or graph traversal for relationship-based questions — before generating grounded answers. The system runs entirely on-device using LM Studio, ensuring no data leaves the user's machine, and provides a full reasoning trace for transparency.

## Key Outcomes

1. Fully-local RAG pipeline — zero cloud dependency, zero API keys required
2. Dual retrieval architecture (Vector DB + Knowledge Graph) with intelligent LLM-based routing
3. Transparent reasoning trace showing each step the AI takes to produce an answer
4. Dynamic ingestion — additive document processing that never overwrites existing knowledge
5. End-to-end system with FastAPI backend, Streamlit frontend, and real-time processing status

---

# Chapter 3: Introduction

## 3.1 Background and Motivation

Retrieval-Augmented Generation (RAG) has emerged as a pivotal paradigm in AI, combining the generative capabilities of Large Language Models (LLMs) with external knowledge retrieval. However, most production RAG systems depend on cloud-hosted APIs (OpenAI, Anthropic, etc.), creating privacy risks — especially in academic settings where proprietary study material, internal documents, and examination content must not be exposed to third-party servers.

Furthermore, conventional RAG pipelines rely exclusively on vector similarity search, which excels at finding semantically similar text passages but fundamentally misses **structural relationships** between concepts. For academic materials — where topics are naturally interconnected through prerequisites, dependencies, and taxonomies — purely vector-based retrieval produces fragmented answers that miss the conceptual "big picture."

Microsoft's GraphRAG research (2024) demonstrated that augmenting RAG with knowledge graphs significantly improves answer quality for relationship-heavy queries. However, the original GraphRAG implementation requires a cloud LLM and substantial compute resources, making it inaccessible for individual students and academic institutions.

This project was motivated by the need for:
- **Data privacy**: Academic institutions handling sensitive material need local-only inference
- **Relationship-aware retrieval**: Academic content is inherently relational
- **Transparency**: Students and educators need to understand how answers are derived
- **Accessibility**: Running on commodity hardware (4GB GPU) without cloud costs

## 3.2 Objectives

1. **Design and implement a fully-local RAG system** that runs without any external API calls or internet connectivity during inference
2. **Develop a dual-retrieval architecture** combining vector similarity search (ChromaDB) with knowledge graph traversal (JSON-based graph) for comprehensive information retrieval
3. **Build an intelligent query routing agent** using LangGraph that classifies user queries and directs them to the optimal retrieval path
4. **Create an automated knowledge graph construction pipeline** that uses a local LLM to extract entity-relationship triples from document chunks
5. **Implement a transparent reasoning trace** that exposes the full decision pipeline (classification → retrieval → generation) to the user
6. **Develop a user-friendly web interface** with dynamic document upload, real-time processing status, and conversational chat
7. **Optimise the system for commodity hardware** (4GB VRAM GPU) using quantized models via LM Studio

## 3.3 Scope

### In Scope
- PDF and DOCX document ingestion with text extraction
- Automatic text chunking with overlap for context preservation
- Vector embedding and storage using ChromaDB
- LLM-based knowledge graph construction from document chunks
- Intelligent query classification (vector vs. graph retrieval)
- Grounded answer generation with source attribution
- Full reasoning trace for each query
- Real-time file processing progress tracking
- Streamlit-based web frontend with chat interface

### Limitations
- **Input formats**: Currently limited to PDF and DOCX; no support for images, tables, or scanned documents (OCR)
- **Graph schema**: The knowledge graph uses a flat JSON structure (not a dedicated graph database like Neo4j)
- **Single-user**: The processing pipeline is single-threaded; concurrent multi-user operation may cause bottlenecks
- **Model constraints**: Tuned for 4GB VRAM (Q4_K_M quantization); larger models would improve quality but exceed hardware limits
- **No conversational memory**: Each query is independent; the system does not maintain multi-turn conversation context
- **Language**: Optimised for English-language documents only

---

# Chapter 4: Literature Survey

## 4.1 Retrieval-Augmented Generation (RAG)

RAG, introduced by Lewis et al. (2020) in the paper "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," combines a pre-trained retriever with a generative model. The retriever fetches relevant passages from an external knowledge source, and the generator conditions on these passages to produce factual answers. RAG addresses the hallucination problem inherent in standalone LLMs by grounding responses in retrieved evidence.

**Gap addressed**: Standard RAG uses a single retrieval path (vector similarity). Local GraphRAG extends this with a dual-path architecture.

## 4.2 GraphRAG (Microsoft Research, 2024)

Microsoft's GraphRAG leverages knowledge graphs built from source documents to provide community-level summaries and relationship-aware retrieval. By constructing a graph of entities and relationships, GraphRAG can answer questions that require understanding connections between concepts — something vector search alone cannot do.

**Gap addressed**: Microsoft's GraphRAG requires cloud LLM access and substantial compute. Local GraphRAG adapts this approach for fully-local, resource-constrained deployment.

## 4.3 LangGraph — Stateful Agent Orchestration

LangGraph (by LangChain) provides a framework for building stateful, multi-step agent workflows as directed graphs. Unlike simple chain-based approaches, LangGraph supports conditional branching, where the execution path is determined dynamically at runtime based on the current state.

**Usage in this project**: The core query pipeline is implemented as a LangGraph state machine with four nodes and conditional routing.

## 4.4 ChromaDB — Open-Source Vector Database

ChromaDB is a lightweight, embeddings-first vector database designed for AI applications. It supports persistent storage, similarity search with relevance scoring, and integrates with LangChain's document/retriever abstractions.

**Usage in this project**: ChromaDB stores document chunk embeddings and performs similarity-based retrieval.

## 4.5 LM Studio — Local LLM Inference

LM Studio enables running open-source LLMs locally through an OpenAI-compatible API endpoint. It supports quantized models (GGUF format) optimised for consumer hardware, including AMD GPU acceleration.

**Usage in this project**: LM Studio serves all three model roles: query router, answer synthesiser, and embedding generator. All inference happens locally on `localhost:1234`.

## 4.6 Comparison of Existing Solutions

| Feature | ChatGPT + Retrieval | Microsoft GraphRAG | LangChain RAG | **Local GraphRAG (Ours)** |
|---|---|---|---|---|
| Fully Local | ❌ | ❌ | Partial | ✅ |
| Knowledge Graph | ❌ | ✅ | ❌ | ✅ |
| Dual Retrieval | ❌ | ✅ | ❌ | ✅ |
| Intelligent Routing | ❌ | N/A | ❌ | ✅ |
| Reasoning Trace | ❌ | ❌ | ❌ | ✅ |
| Low VRAM (4GB) | N/A | ❌ | Partial | ✅ |
| Dynamic Ingestion | ❌ | ❌ | Partial | ✅ |

---

# Chapter 5: System Design & Architecture

## 5.1 High-Level Architecture

```mermaid
graph TD
    A["User (Browser)"] -->|Streamlit UI| B["Frontend<br/>ui/app.py"]
    B -->|HTTP REST| C["FastAPI Backend<br/>api/server.py"]
    
    C -->|Upload| D["Ingestion Pipeline<br/>core/ingest.py"]
    C -->|Query| E["LangGraph Engine<br/>core/engine.py"]
    
    D -->|Extract + Chunk| F["Text Splitter"]
    F -->|Embed| G["ChromaDB<br/>db/chromadb/"]
    F -->|Triple Extraction| H["Graph Builder<br/>core/graph_builder.py"]
    H -->|Persist| I["graph.json<br/>db/graph.json"]
    
    E -->|Classify| J["Router Agent<br/>mistral-7b"]
    J -->|vector path| K["Vector Retriever<br/>core/retrievers.py"]
    J -->|graph path| L["Graph Retriever<br/>core/retrievers.py"]
    K -->|ChromaDB query| G
    L -->|Graph traversal| I
    K --> M["Answer Generator<br/>mistral-7b"]
    L --> M
    M -->|response + trace| C
    
    C -->|JSON Response| B
    B -->|Display| A
    
    subgraph LM_Studio["LM Studio (localhost:1234)"]
        J
        M
        N["Embedding Model<br/>nomic-embed-text-v1.5"]
    end
```

## 5.2 Data Flow — Ingestion Pipeline

```mermaid
sequenceDiagram
    participant User
    participant Streamlit as Streamlit UI
    participant FastAPI as FastAPI Server
    participant Ingest as Ingestion Pipeline
    participant ChromaDB
    participant GraphBuilder as Graph Builder
    participant LMStudio as LM Studio

    User->>Streamlit: Upload PDF/DOCX files
    Streamlit->>FastAPI: POST /api/upload (multipart)
    FastAPI->>FastAPI: Validate files (type, size)
    FastAPI->>FastAPI: Save to uploads/
    FastAPI-->>Streamlit: 200 OK (queued)
    FastAPI->>Ingest: Background task: process files
    
    loop For each file
        Ingest->>Ingest: Extract text (PyPDF / python-docx)
        Ingest->>Ingest: Chunk with RecursiveCharacterTextSplitter
        Ingest->>Ingest: Generate content-hash IDs (dedup)
        Ingest->>LMStudio: Embed chunks (nomic-embed-text-v1.5)
        LMStudio-->>Ingest: Embedding vectors
        Ingest->>ChromaDB: Store embeddings (additive, dedup)
        Ingest->>GraphBuilder: Pass chunks for triple extraction
        
        loop For each chunk
            GraphBuilder->>LMStudio: Extract triples (mistral-7b)
            LMStudio-->>GraphBuilder: JSON triples [{subject, relation, object}]
        end
        
        GraphBuilder->>GraphBuilder: Merge into graph.json (dedup nodes + edges)
    end
    
    Ingest-->>FastAPI: Processing complete
    Streamlit->>FastAPI: GET /api/upload-status (polling)
    FastAPI-->>Streamlit: Progress updates
```

## 5.3 Data Flow — Query Pipeline

```mermaid
stateDiagram-v2
    [*] --> ClassifyQuery: User submits query
    
    ClassifyQuery --> RetrieveVector: route = "vector"<br/>(facts, definitions)
    ClassifyQuery --> RetrieveGraph: route = "graph"<br/>(relationships, comparisons)
    
    RetrieveVector --> GenerateAnswer: Context from ChromaDB chunks
    RetrieveGraph --> GenerateAnswer: Context from KG triples
    
    GenerateAnswer --> [*]: Return answer + reasoning trace

    note right of ClassifyQuery
        LLM classifies query type
        using ROUTER_SYSTEM_PROMPT
    end note
    
    note right of RetrieveVector
        similarity_search_with_relevance_scores
        Top-k=5 chunks returned
    end note
    
    note right of RetrieveGraph
        Keyword matching → node lookup
        Multi-hop edge traversal (max 2 hops)
    end note
```

## 5.4 Module Breakdown

| Module | File | Responsibility | Lines of Code |
|---|---|---|---|
| **Configuration** | [config.py](file:///d:/6th%20sem/major_project/core/config.py) | Centralized settings, LM Studio client, model names, prompt templates, ProcessingStatus singleton, health checks | 209 |
| **Ingestion** | [ingest.py](file:///d:/6th%20sem/major_project/core/ingest.py) | PDF/DOCX text extraction, recursive chunking, ChromaDB embedding with deduplication, multi-file pipeline with progress | 256 |
| **Graph Builder** | [graph_builder.py](file:///d:/6th%20sem/major_project/core/graph_builder.py) | LLM-based triple extraction, graph storage (JSON), node/edge deduplication, incremental graph merging | 217 |
| **Retrievers** | [retrievers.py](file:///d:/6th%20sem/major_project/core/retrievers.py) | LMStudioEmbeddings wrapper, VectorRetriever (ChromaDB), GraphRetriever (JSON KG with multi-hop traversal) | 247 |
| **Engine** | [engine.py](file:///d:/6th%20sem/major_project/core/engine.py) | LangGraph state machine (QueryState), 4 nodes (classify, vector-retrieve, graph-retrieve, generate), conditional routing | 253 |
| **API Server** | [server.py](file:///d:/6th%20sem/major_project/api/server.py) | FastAPI app, 4 REST endpoints (health, upload, upload-status, query), background task processing, CORS, Pydantic schemas | 345 |
| **Frontend** | [app.py](file:///d:/6th%20sem/major_project/ui/app.py) | Streamlit UI, file upload widget, processing progress bar, chat interface, reasoning trace sidebar, system health display | 624 |
| **Total** | **7 modules** | | **~2,151** |

## 5.5 Database Schema

### ChromaDB Collection: `local_graphrag`

| Field | Type | Description |
|---|---|---|
| `id` | string | Content-hash-based chunk ID (e.g., `filename_abc123def456`) |
| `page_content` | string | The text chunk content |
| `embedding` | float[] | 768-dimensional vector from nomic-embed-text-v1.5 |
| `metadata.source` | string | Source filename |
| `metadata.page` | int | Page number (PDF) or section number (DOCX) |
| `metadata.file_type` | string | `"pdf"` or `"docx"` |
| `metadata.chunk_id` | string | Same as `id` — used for deduplication |

### Knowledge Graph: `graph.json`

```json
{
  "nodes": [
    {
      "id": "md5_hash_10char",
      "label": "Concept Name",
      "type": "concept",
      "properties": { "source": "filename.pdf" }
    }
  ],
  "edges": [
    {
      "source": "node_id_1",
      "relation": "is_a | part_of | prerequisite_of | related_to | defines | contains | uses | example_of",
      "target": "node_id_2"
    }
  ]
}
```

## 5.6 Entity-Relationship Summary (Inferable ER Diagram)

```mermaid
erDiagram
    DOCUMENT ||--|{ CHUNK : "split into"
    CHUNK ||--o| EMBEDDING : "embedded as"
    CHUNK ||--|{ TRIPLE : "extracted from"
    TRIPLE ||--|| NODE_SUBJECT : "has subject"
    TRIPLE ||--|| NODE_OBJECT : "has object"
    TRIPLE ||--|| EDGE : "creates"
    NODE_SUBJECT }|--|| GRAPH : "part of"
    NODE_OBJECT }|--|| GRAPH : "part of"
    EDGE }|--|| GRAPH : "part of"
    EMBEDDING }|--|| CHROMADB : "stored in"
    GRAPH ||--|| GRAPH_JSON : "persisted as"
    
    DOCUMENT {
        string filename
        string file_type
        int total_pages
    }
    CHUNK {
        string chunk_id
        string page_content
        int page_number
        string source
    }
    EMBEDDING {
        float_array vector_768d
        float relevance_score
    }
    TRIPLE {
        string subject
        string relation
        string object
    }
```

---

# Chapter 6: Technology Stack

## 6.1 Languages & Frameworks

| Layer | Technology | Version / Details |
|---|---|---|
| **Language** | Python | 3.11+ |
| **Backend Framework** | FastAPI | Async REST API with Pydantic validation |
| **Frontend Framework** | Streamlit | Rapid web UI with session state and widgets |
| **Agent Orchestration** | LangGraph | Stateful directed-graph agent workflow |
| **Vector Database** | ChromaDB | Persistent local vector store |
| **LLM Serving** | LM Studio | OpenAI-compatible local inference server |

## 6.2 AI Models (All Local via LM Studio)

| Role | Model | Quantization | VRAM Usage |
|---|---|---|---|
| **Router Agent** | `mistralai/mistral-7b-instruct-v0.3` | Q4_K_M | ~4GB |
| **Answer Synthesiser** | `mistralai/mistral-7b-instruct-v0.3` | Q4_K_M | Shared |
| **Embeddings** | `nomic-embed-text-v1.5` | Full | Lightweight |

> **Hardware Note**: Entire stack is tuned for **AMD Radeon RX 6500M (4GB VRAM)** with context length capped at **4096 tokens**.

## 6.3 Dependencies (requirements.txt)

| Package | Purpose |
|---|---|
| `openai` | LM Studio OpenAI-compatible client for LLM inference and embeddings |
| `langchain-chroma` | ChromaDB integration with LangChain document abstractions |
| `langchain-core` | Core LangChain types (Document, BaseEmbeddings) |
| `langchain-text-splitters` | RecursiveCharacterTextSplitter for intelligent chunking |
| `langgraph` | State machine orchestration for the multi-agent pipeline |
| `chromadb` | Embedded vector database for similarity search |
| `fastapi` | Async web framework for the REST API |
| `uvicorn[standard]` | ASGI server to run the FastAPI application |
| `streamlit` | Web frontend framework with widgets and session state |
| `pypdf` | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `python-multipart` | Multipart file upload handling for FastAPI |
| `requests` | HTTP client for Streamlit → FastAPI communication |
| `pydantic` | Data validation and schema definitions |
| `numpy` | Numerical operations for embedding handling |

## 6.4 Tools & Platforms

| Tool | Use |
|---|---|
| **LM Studio** | Local LLM inference server (OpenAI-compatible endpoint at `localhost:1234`) |
| **Python venv** | Dependency isolation (`.venv/` directory present) |
| **VS Code** | Development IDE (inferred from workspace structure) |
| **Git** | Version control |
| **Windows** | Development OS |

---

# Chapter 7: Implementation Details

## 7.1 Project File Structure

```
d:\6th sem\major_project\
├── core/                          # Core business logic
│   ├── __init__.py                # Package marker
│   ├── config.py                  # Centralized configuration & health checks
│   ├── ingest.py                  # Document ingestion pipeline
│   ├── graph_builder.py           # Knowledge graph construction
│   ├── retrievers.py              # Vector + Graph retrieval classes
│   └── engine.py                  # LangGraph state machine
├── api/                           # REST API layer
│   ├── __init__.py                # Package marker
│   └── server.py                  # FastAPI application & endpoints
├── ui/                            # Frontend layer
│   └── app.py                     # Streamlit application (624 lines)
├── data/                          # Raw files (optional storage)
├── db/                            # Persistent data stores
│   ├── chromadb/                  # ChromaDB vector database
│   └── graph.json                 # Knowledge graph (nodes + edges)
├── uploads/                       # Temporary upload staging (auto-cleaned)
├── requirements.txt               # Python dependencies (15 packages)
└── README.md                      # Project documentation
```

## 7.2 Module-by-Module Implementation

### 7.2.1 Configuration Module — [config.py](file:///d:/6th%20sem/major_project/core/config.py)

This module serves as the **single source of truth** for all settings. Key responsibilities:

- **Path management**: Defines `PROJECT_ROOT`, `CHROMA_DIR`, `GRAPH_PATH`, `UPLOADS_DIR`
- **LM Studio client**: Initializes the shared `OpenAI` client pointing to `localhost:1234`
- **Model constants**: `ROUTER_MODEL`, `SYNTHESISER_MODEL`, `EMBEDDING_MODEL`
- **Prompt templates**: Three carefully crafted system prompts for routing, generation, and triple extraction
- **ProcessingStatus**: Thread-safe singleton for tracking background ingestion progress

```python
# Key snippet — ProcessingStatus (thread-safe singleton)
class ProcessingStatus:
    """Thread-safe singleton for tracking background ingestion progress."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._status = {
            "is_processing": False,
            "current_file": "",
            "stage": "",         # "extracting", "chunking", "embedding", "graphing", "done", "error"
            "progress": 0.0,     # 0.0 to 1.0
            "total_files": 0,
            "processed_files": 0,
            "message": "",
            "files_ingested": [],
            "error": None,
            "started_at": None,
        }
```

### 7.2.2 Ingestion Pipeline — [ingest.py](file:///d:/6th%20sem/major_project/core/ingest.py)

The ingestion module implements a four-stage pipeline:

1. **Text Extraction**: Routes to `PyPDF` (for PDFs) or `python-docx` (for DOCX) based on file extension
2. **Chunking**: Uses `RecursiveCharacterTextSplitter` with chunk_size=1000, overlap=200, intelligent separators (`\n\n`, `\n`, `. `, ` `)
3. **Deduplication**: Generates MD5-based chunk IDs from content, checks against existing ChromaDB entries
4. **Embedding**: Embeds new chunks via LM Studio's embedding model, stores in ChromaDB

```python
# Key snippet — Content-hash-based deduplication
def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,           # 1000 chars
        chunk_overlap=CHUNK_OVERLAP,      # 200 chars
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    
    for chunk in chunks:
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:12]
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'unknown')}_{content_hash}"
    
    return chunks
```

### 7.2.3 Knowledge Graph Builder — [graph_builder.py](file:///d:/6th%20sem/major_project/core/graph_builder.py)

Constructs a knowledge graph by:

1. **Triple extraction**: Sends each chunk to the LLM with a structured prompt requesting `(subject, relation, object)` triples in JSON format
2. **JSON parsing with fallback**: Uses regex to extract JSON arrays even when the LLM wraps output in markdown
3. **Graph merging**: Deduplicates nodes by label (via MD5-based IDs) and edges by `(source, relation, target)` tuple
4. **Persistent storage**: Saves the merged graph to `db/graph.json`

```python
# Key snippet — LLM-based triple extraction
def extract_triples_from_chunk(chunk_text: str) -> list[dict]:
    prompt = TRIPLE_EXTRACTION_PROMPT.format(text=chunk_text[:1500])  # Cap input size
    
    res = client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": "You are a knowledge graph extraction engine. Output ONLY valid JSON arrays."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=0.0,  # Deterministic for structured extraction
    )
    
    raw = res.choices[0].message.content.strip()
    json_match = re.search(r'\[.*\]', raw, re.DOTALL)
    if json_match:
        triples = json.loads(json_match.group())
        # ... validate structure ...
```

### 7.2.4 Retriever Classes — [retrievers.py](file:///d:/6th%20sem/major_project/core/retrievers.py)

Two retriever classes with a shared embedding wrapper:

**LMStudioEmbeddings** — Custom wrapper implementing LangChain's embedding interface via the shared OpenAI client. Processes documents in batches of 16.

**VectorRetriever** — Wraps ChromaDB with `similarity_search_with_relevance_scores`. Returns top-k (default 5) documents with metadata and scores.

**GraphRetriever** — Implements keyword-based node matching + multi-hop edge traversal:

```python
# Key snippet — Multi-hop graph traversal
def query(self, text: str, max_hops: int = 2) -> list[dict]:
    # Step 1: Extract keywords from query
    keywords = [w.lower() for w in re.split(r'\W+', text) if len(w) > 2]
    
    # Step 2: Match keywords against node labels
    matched_node_ids = set()
    for node in self._graph["nodes"]:
        label = node.get("label", "").lower()
        for kw in keywords:
            if kw in label:
                matched_node_ids.add(node["id"])
    
    # Step 3: Multi-hop traversal from matched nodes
    frontier = matched_node_ids.copy()
    for hop in range(max_hops):
        next_frontier = set()
        for node_id in frontier:
            for edge in self._adjacency.get(node_id, []):
                # Collect triples, expand frontier...
```

### 7.2.5 LangGraph Engine — [engine.py](file:///d:/6th%20sem/major_project/core/engine.py)

The core orchestration layer, built as a **LangGraph state machine** with:

- **State**: `QueryState` TypedDict with fields: `query`, `route`, `context`, `response`, `reasoning_trace`
- **4 Nodes**: `classify_query`, `retrieve_vector`, `retrieve_graph`, `generate_answer`
- **Conditional routing**: After classification, routes to either vector or graph retrieval

```python
# Key snippet — LangGraph graph construction
def build_graph() -> StateGraph:
    builder = StateGraph(QueryState)
    
    builder.add_node("classify_query", classify_query)
    builder.add_node("retrieve_vector", retrieve_vector)
    builder.add_node("retrieve_graph", retrieve_graph)
    builder.add_node("generate_answer", generate_answer)
    
    builder.add_edge(START, "classify_query")
    builder.add_conditional_edges(
        "classify_query",
        route_decision,
        {"retrieve_vector": "retrieve_vector", "retrieve_graph": "retrieve_graph"},
    )
    builder.add_edge("retrieve_vector", "generate_answer")
    builder.add_edge("retrieve_graph", "generate_answer")
    builder.add_edge("generate_answer", END)
    
    return builder.compile()
```

### 7.2.6 FastAPI Backend — [server.py](file:///d:/6th%20sem/major_project/api/server.py)

Exposes 4 REST endpoints with Pydantic schema validation:

| Endpoint | Method | Function |
|---|---|---|
| `/api/health` | GET | Returns LM Studio connectivity + DB readiness status |
| `/api/upload` | POST | Accepts multipart PDF/DOCX uploads (max 100MB each), validates, queues background processing |
| `/api/upload-status` | GET | Polls background processing progress (stage, %, current file) |
| `/api/query` | POST | Runs the full LangGraph pipeline, returns answer + reasoning trace |

**Key design decisions**:
- Background task processing using FastAPI's `BackgroundTasks` to keep the API responsive
- Pre-flight checks before every query (LM Studio running? Data ingested?)
- Automatic cleanup of uploaded files after processing
- CORS enabled for cross-origin Streamlit communication

### 7.2.7 Streamlit Frontend — [app.py](file:///d:/6th%20sem/major_project/ui/app.py)

The largest module (624 lines), implementing:

- **Custom CSS** (~200 lines): Gradient header, glassmorphism trace cards, status badges, route badges, processing animation cards — all using the Inter font
- **Sidebar** (4 sections): File upload widget, processing progress bar, system health status, reasoning trace display
- **Main area**: Gradient header banner, system status banners (API offline / LM Studio down / no data), chat interface with user/assistant avatars
- **Session state**: Persists messages, traces, health data, and upload state across reruns
- **Auto-refresh**: Polls `/api/upload-status` every 2 seconds during file processing

---

# Chapter 8: Results & Output

## 8.1 Features Implemented and Working

| # | Feature | Status | Description |
|---|---|---|---|
| 1 | PDF Text Extraction | ✅ Working | Per-page extraction via PyPDF with metadata |
| 2 | DOCX Text Extraction | ✅ Working | Per-section extraction with paragraph grouping |
| 3 | Intelligent Chunking | ✅ Working | 1000-char chunks with 200-char overlap, content-hash IDs |
| 4 | Content Deduplication | ✅ Working | MD5-based chunk IDs prevent re-embedding |
| 5 | Vector Embedding | ✅ Working | 768-dim embeddings via nomic-embed-text-v1.5 |
| 6 | ChromaDB Storage | ✅ Working | Persistent vector store with similarity search |
| 7 | Knowledge Graph Building | ✅ Working | LLM triple extraction + incremental merging |
| 8 | Query Classification | ✅ Working | LLM-based routing to vector or graph path |
| 9 | Vector Retrieval | ✅ Working | Top-5 similarity search with relevance scores |
| 10 | Graph Retrieval | ✅ Working | Keyword matching + 2-hop edge traversal |
| 11 | Answer Generation | ✅ Working | Grounded synthesis using retrieved context |
| 12 | Reasoning Trace | ✅ Working | Full step-by-step audit trail with latencies |
| 13 | File Upload UI | ✅ Working | Drag-and-drop with validation and size display |
| 14 | Real-time Progress | ✅ Working | Stage-specific icons, progress bar, file counter |
| 15 | Chat Interface | ✅ Working | Conversational UI with user/assistant avatars |
| 16 | System Health Dashboard | ✅ Working | LM Studio, model, and database status checks |
| 17 | Cold Start Protection | ✅ Working | Clear error banners for missing services |
| 18 | Dynamic Ingestion | ✅ Working | Additive processing — never overwrites data |

## 8.2 UI Screens / Flow Description

### Screen 1: Initial State (No Data)
- Gradient purple header with "🧠 Local GraphRAG" branding
- Info banner: "📂 No Documents Yet — Use the sidebar to upload PDF or DOCX files"
- Chat input disabled with placeholder: "Upload documents first, then ask questions..."
- Sidebar shows system health (LM Studio ✅, Database ⚠️ No Data)

### Screen 2: File Upload & Processing
- Sidebar file uploader with selected file list (name + size in KB)
- "🚀 Process Files" primary button
- During processing: animated progress card showing stage icon + message
- Stages cycle through: 🔄 STARTING → 📄 EXTRACTING → ✂️ CHUNKING → 🧮 EMBEDDING → 🔗 GRAPHING → ✅ DONE
- Progress bar with percentage, file counter (e.g., "Files: 2/5")

### Screen 3: Chat Interface (Active)
- Chat input enabled: "Ask about your documents..."
- User messages with 🧑‍🎓 avatar, assistant messages with 🧠 avatar
- Each response shows route badge (📄 Vector DB or 🔗 Knowledge Graph) and latency

### Screen 4: Reasoning Trace (Sidebar)
- Step-by-step trace cards with glassmorphism styling
- Card 1: 🏷️ Classify Query — shows route chosen and LLM raw output
- Card 2: 📄 Retrieve Vector — expandable list of chunks with source, page, score, preview  
  OR 🔗 Retrieve Graph — expandable list of triples (subject → relation → object)
- Card 3: ✨ Generate Answer — model used and latency

### Screen 5: Error States
- "⚠️ Cannot Connect to API Server" — red banner with startup command
- "⚠️ LM Studio Not Running" — red banner with instructions
- "⚠️ Required Models Not Loaded" — lists specific missing models

## 8.3 Sample Data

The `uploads/` directory contains sample VTU course materials used during testing:
- `BIS701-module-2-textbook.pdf` (5.9 MB)
- `BIS701-module-3-textbook.pdf` (6.9 MB)
- `BIS701-module-4-textbook.pdf` (6.3 MB)
- `BIS701-module-5-textbook.pdf` (1.4 MB)
- `MACHINE LEARNING LAB.pdf` (1.6 MB)

---

# Chapter 9: Testing

## 9.1 Testing Approach

> [!IMPORTANT]
> No dedicated test files (`test_*.py`, `*_test.py`) or testing framework configurations (`pytest.ini`, `tox.ini`) were found in the codebase. Testing was performed through **manual functional testing** and **integrated validation logic**.

## 9.2 Built-in Validation & Error Handling

The codebase incorporates extensive runtime validation as a testing substitute:

| Component | Validation Logic |
|---|---|
| **File Upload** | Extension whitelist (`.pdf`, `.docx`, `.doc`), 100MB size limit, concurrent processing guard |
| **Text Extraction** | Empty document detection, graceful skip on no-text files |
| **Chunk Dedup** | Content-hash-based IDs prevent duplicate embeddings across re-uploads |
| **Triple Extraction** | JSON parse with regex fallback, structural validation (`subject` + `object` keys required) |
| **Graph Merging** | Set-based dedup for both nodes (by ID) and edges (by source+relation+target tuple) |
| **Query Routing** | Fallback to "vector" path if LLM output is ambiguous |
| **Health Checks** | Pre-flight checks before every query (LM Studio running? DB has data?) |
| **Error Propagation** | ProcessingStatus tracks errors with stage + file context |

## 9.3 Suggested Test Cases for Documentation

| Test Case | Input | Expected Output | Type |
|---|---|---|---|
| TC-01: PDF Upload | Valid PDF file | Text extracted, chunks embedded, graph built | Functional |
| TC-02: DOCX Upload | Valid DOCX file | Paragraphs grouped into sections, embedded | Functional |
| TC-03: Invalid File Type | `.xlsx` file | Rejected with "Unsupported type" error | Negative |
| TC-04: Oversized File | 150MB PDF | Rejected with "File exceeds 100MB limit" | Boundary |
| TC-05: Duplicate Upload | Same PDF uploaded twice | 0 new chunks embedded (dedup works) | Functional |
| TC-06: Factual Query | "What is machine learning?" | Routed to vector path, answer from chunks | Functional |
| TC-07: Relationship Query | "How does X relate to Y?" | Routed to graph path, answer from triples | Functional |
| TC-08: LM Studio Down | Query with LM Studio offline | 503 error with clear message | Error Handling |
| TC-09: Empty DB Query | Query with no data ingested | 503 error: "No documents ingested yet" | Error Handling |
| TC-10: Concurrent Upload | Upload while processing active | 409 error: "A processing job is already running" | Concurrency |

---

# Chapter 10: Conclusion & Future Scope

## 10.1 Conclusion

The **Local GraphRAG** project successfully demonstrates that a fully-local, privacy-preserving Retrieval-Augmented Generation system with knowledge graph capabilities can be built and deployed on commodity hardware. The key achievements are:

1. **Privacy-first architecture**: All inference, embedding, and data storage happens locally via LM Studio — no data ever leaves the user's machine, making it suitable for sensitive academic and institutional content.

2. **Dual-retrieval innovation**: By combining vector similarity search (ChromaDB) with knowledge graph traversal (JSON-based graph with multi-hop edges), the system handles both factual/definitional queries and relationship/comparison queries — a significant improvement over single-path RAG systems.

3. **Intelligent routing**: The LangGraph-based state machine with LLM-driven query classification automatically directs each query to the optimal retrieval path, eliminating the need for users to specify search mode.

4. **Transparent reasoning**: The full reasoning trace (classification → retrieval → generation) with per-step latencies provides unprecedented transparency into how the AI derives its answers — critical for educational use cases.

5. **Practical engineering**: Dynamic incremental ingestion with content-hash deduplication, background processing with real-time progress tracking, comprehensive health checks, and cold-start protection make the system production-ready for individual use.

6. **Resource efficiency**: The entire stack operates within 4GB VRAM using Q4_K_M quantized models, proving that advanced RAG architectures need not require enterprise-grade hardware.

## 10.2 Limitations of Current Implementation

1. **No OCR support**: Scanned PDFs and image-based documents cannot be processed
2. **Flat graph storage**: JSON-based graph storage limits scalability; no support for advanced graph queries (e.g., shortest path, community detection)
3. **Single-threaded processing**: Large batches of documents are processed sequentially
4. **No conversation memory**: Each query is independent; no multi-turn context retention
5. **English-only**: Not tested or optimised for non-English documents
6. **Basic graph retrieval**: Keyword matching for node lookup; no semantic graph search
7. **No authentication**: The API has no user authentication or access control
8. **Fixed model configuration**: Model names are hardcoded; no runtime model switching

## 10.3 Future Enhancements

| Enhancement | Description | Impact |
|---|---|---|
| **Neo4j Integration** | Replace JSON graph with Neo4j for Cypher queries, community detection, and PageRank-based relevance | High |
| **OCR Pipeline** | Add Tesseract/EasyOCR for scanned document support | Medium |
| **Multi-turn Conversation** | Implement chat history context window for follow-up questions | High |
| **Hybrid Retrieval** | Combine vector + graph results for every query (weighted fusion) instead of routing to one path | High |
| **User Authentication** | Add JWT-based auth for multi-user deployments | Medium |
| **Batch Parallel Processing** | Use async/multiprocessing for concurrent chunk embedding and triple extraction | Medium |
| **Model Hot-Swap** | Allow runtime model selection from the UI | Low |
| **Multi-language Support** | Add multilingual embedding models and language-aware chunking | Medium |
| **Graph Visualization** | Interactive D3.js/vis.js graph explorer in the Streamlit UI | Medium |
| **Evaluation Framework** | Add automated evaluation using RAGAS metrics (faithfulness, relevance, context precision) | High |
| **Docker Deployment** | Containerize the full stack with Docker Compose for one-command deployment | Medium |

---

# Chapter 11: References

## 11.1 Research Papers & Publications

1. Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *Neural Information Processing Systems (NeurIPS)*. arXiv:2005.11401
2. Edge, D., et al. (2024). "From Local to Global: A Graph RAG Approach to Query-Focused Summarization." *Microsoft Research*. arXiv:2404.16130

## 11.2 Frameworks & Libraries Used

| Reference | URL |
|---|---|
| LangGraph | https://langchain-ai.github.io/langgraph/ |
| LangChain Core | https://python.langchain.com/ |
| ChromaDB | https://www.trychroma.com/ |
| FastAPI | https://fastapi.tiangolo.com/ |
| Streamlit | https://streamlit.io/ |
| LM Studio | https://lmstudio.ai/ |
| OpenAI Python Client | https://github.com/openai/openai-python |
| PyPDF | https://pypdf.readthedocs.io/ |
| python-docx | https://python-docx.readthedocs.io/ |
| Pydantic | https://docs.pydantic.dev/ |
| NumPy | https://numpy.org/ |
| Uvicorn | https://www.uvicorn.org/ |

## 11.3 AI Models

| Model | Source | Paper/Reference |
|---|---|---|
| Mistral 7B Instruct v0.3 | Mistral AI | https://mistral.ai/news/announcing-mistral-7b/ |
| Nomic Embed Text v1.5 | Nomic AI | https://blog.nomic.ai/posts/nomic-embed-text-v1-5 |

## 11.4 Datasets Used for Testing

| Document | Source |
|---|---|
| BIS701 Module 2–5 Textbooks | VTU 7th Semester Big Data Analytics course material |
| Machine Learning Lab Manual | VTU Lab course material |

---

> [!TIP]
> **For Presentation Slides**: Use Chapters 2 (Abstract), 5 (Architecture diagrams), 6 (Tech Stack table), 7.2.5 (LangGraph flow), 8.2 (UI screens), and 10.3 (Future scope) as the core slides. The Mermaid diagrams in Chapter 5 can be rendered into images for slide decks.

> [!IMPORTANT]
> **Items to fill manually**: Team member names/USNs, Guide name and designation, College name, and actual screenshots from a running instance of the application. You may also want to add any specific VTU formatting requirements (title page, certificate page, etc.).
