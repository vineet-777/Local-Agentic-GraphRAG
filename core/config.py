"""
Local GraphRAG — Centralized Configuration
All settings for LM Studio, ChromaDB, Knowledge Graph, and dynamic ingestion.
Single source of truth for model names and inference settings.
"""

import os
import json
import threading
import requests
from pathlib import Path
from datetime import datetime
from openai import OpenAI

# ─── Base Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = PROJECT_ROOT / "db"
CHROMA_DIR = DB_DIR / "chromadb"
GRAPH_PATH = DB_DIR / "graph.json"
UPLOADS_DIR = PROJECT_ROOT / "uploads"

# Ensure critical directories exist
DB_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ─── LM Studio Configuration ─────────────────────────────────────────────────

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"

client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)

# Models tuned for AMD Radeon RX 6500M — 4GB VRAM, Q4_K_M quants
ROUTER_MODEL = os.getenv("ROUTER_MODEL", "google/gemma-4-e4b")
SYNTHESISER_MODEL = os.getenv("SYNTHESISER_MODEL", "google/gemma-4-e4b")

# Embedding model — Native support for Google's EmbeddingGemma-2 with automatic fallback
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "google/embeddinggemma-2")
FALLBACK_EMBEDDING_MODEL = "text-embedding-nomic-embed-text-v1.5"

# Matryoshka Representation Learning (MRL) dimension (e.g. 768, 512, 256, 128)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# LM Studio inference settings — Gemma 4 E4B supports 128K context,
# but we keep limits tight for 4GB VRAM (AMD Radeon RX 6500M)
ROUTER_MAX_TOKENS = 128       # Router only outputs one word; 128 covers thinking overhead
SYNTHESIS_MAX_TOKENS = 2048   # Plenty for academic answers without VRAM pressure
EXTRACTION_MAX_TOKENS = 512   # Triple extraction — small JSON arrays
CONTEXT_LENGTH = 4096         # Safe for 4GB VRAM — higher values cause extreme slowdown
TEMPERATURE = 0.2

# ─── Retrieval Settings ──────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "5"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "local_graphrag")

# ─── Chunking Settings ───────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ─── LLM Prompt Templates (Gemma 4 E4B — native system role + thinking) ──────

ROUTER_SYSTEM_PROMPT = """You are a query classification agent for a general-purpose aid system.

Your sole purpose is to decide which retrieval backend should handle the user's query.
Use your internal reasoning to analyse the query carefully, then output your final classification.

Classification criteria:
• "vector" — factual, definitional, descriptive, or explanatory questions.
  Examples: "What is X?", "Explain Y", "Summarise Z", "List the features of…"
• "graph" — relational, comparative, dependency, or connection-based questions.
  Examples: "How does X relate to Y?", "What are the prerequisites of X?", "Compare A and B"

If you are uncertain, default to "vector".

IMPORTANT: After your reasoning, your final answer MUST be a single line containing ONLY the word:
vector
OR
graph

Nothing else on that final line."""

GENERATOR_SYSTEM_PROMPT = """You are Local GraphRAG, a general-purpose document QA assistant.

Use your reasoning capabilities to think through the retrieved context carefully before
formulating your answer. Follow these rules strictly:

1. Answer the user's question using ONLY the provided context below.
2. Cite specific sources, topics, or modules from the context when relevant.
3. Use bullet points or numbered lists for multi-part answers.
4. If the context does not contain enough information to answer fully, say:
   "I don't have enough information in the provided documents to answer this."
5. NEVER hallucinate or fabricate information that isn't in the context.
6. Be clear, concise, and academically accurate.

Context:
{context}"""

TRIPLE_EXTRACTION_SYSTEM_PROMPT = """You are a knowledge graph construction agent.

Your task is to extract meaningful entity-relationship triples from text.
Use step-by-step reasoning to think through the chunk carefully before extracting.

Extraction rules:
• Focus on academic concepts, topics, definitions, and their relationships.
• Use clear relationship types: "is_a", "part_of", "prerequisite_of", "related_to",
  "defines", "contains", "uses", "example_of".
• Keep entity names concise but descriptive.
• Extract 3–8 of the most meaningful triples per chunk — avoid redundant or trivial ones.
• Identify the most important relationships first.

Output format — respond with ONLY a valid JSON array, nothing else:
[
  {"subject": "...", "relation": "...", "object": "..."}
]"""

TRIPLE_EXTRACTION_USER_PROMPT = """Extract knowledge graph triples from the following text:

{text}"""


# ─── Processing Status (thread-safe) ─────────────────────────────────────────

class ProcessingStatus:
    """Thread-safe singleton for tracking background ingestion progress."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._status = {
            "is_processing": False,
            "current_file": "",
            "stage": "",           # "extracting", "chunking", "embedding", "graphing", "done", "error"
            "progress": 0.0,       # 0.0 to 1.0
            "total_files": 0,
            "processed_files": 0,
            "message": "",
            "files_ingested": [],   # History of processed files
            "error": None,
            "started_at": None,
        }
    
    def get(self) -> dict:
        with self._lock:
            return dict(self._status)
    
    def update(self, **kwargs):
        with self._lock:
            self._status.update(kwargs)
    
    def start(self, total_files: int):
        with self._lock:
            self._status.update({
                "is_processing": True,
                "total_files": total_files,
                "processed_files": 0,
                "progress": 0.0,
                "error": None,
                "stage": "starting",
                "message": f"Preparing to process {total_files} file(s)...",
                "started_at": datetime.now().isoformat(),
            })
    
    def finish(self):
        with self._lock:
            self._status.update({
                "is_processing": False,
                "stage": "done",
                "progress": 1.0,
                "message": f"✅ All files processed successfully!",
            })
    
    def fail(self, error: str):
        with self._lock:
            self._status.update({
                "is_processing": False,
                "stage": "error",
                "error": error,
                "message": f"❌ Error: {error}",
            })


# Global singleton
processing_status = ProcessingStatus()


# ─── Health Check Utilities ───────────────────────────────────────────────────

def is_lm_studio_running() -> dict:
    """Check if LM Studio server is reachable and return model availability."""
    try:
        resp = requests.get(f"{LM_STUDIO_BASE_URL}/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            # Flexible matching for LLM and Embedding models
            llm_active = SYNTHESISER_MODEL in models or any("gemma" in m.lower() for m in models)
            embed_active = (
                EMBEDDING_MODEL in models
                or FALLBACK_EMBEDDING_MODEL in models
                or any("embeddinggemma" in m.lower() for m in models)
                or any("embed" in m.lower() for m in models)
            )
            router_active = ROUTER_MODEL in models or any("gemma" in m.lower() for m in models)

            return {
                "running": True,
                "models_available": models,
                "llm_ready": llm_active,
                "embed_ready": embed_active,
                "router_ready": router_active,
                "configured_embed_model": EMBEDDING_MODEL,
                "configured_llm_model": SYNTHESISER_MODEL,
                "embedding_dim": EMBEDDING_DIM,
            }
        return {"running": False, "error": f"Status {resp.status_code}"}
    except requests.ConnectionError:
        return {"running": False, "error": "Cannot connect to LM Studio"}
    except Exception as e:
        return {"running": False, "error": str(e)}


def is_db_ready() -> dict:
    """Check if the database assets exist on disk."""
    chroma_exists = CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()) if CHROMA_DIR.exists() else False
    graph_exists = GRAPH_PATH.exists() and GRAPH_PATH.stat().st_size > 0

    # Count documents in graph
    graph_nodes = 0
    graph_edges = 0
    if graph_exists:
        try:
            with open(GRAPH_PATH, "r") as f:
                g = json.load(f)
                graph_nodes = len(g.get("nodes", []))
                graph_edges = len(g.get("edges", []))
        except Exception:
            pass

    return {
        "ready": chroma_exists or graph_exists,  # Either source is enough to start chatting
        "chromadb_exists": chroma_exists,
        "graph_json_exists": graph_exists,
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "chroma_path": str(CHROMA_DIR),
        "graph_path": str(GRAPH_PATH),
    }
