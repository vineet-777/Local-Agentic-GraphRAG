"""
Local GraphRAG — Retriever Logic
Dual retrieval: ChromaDB vector search + JSON Knowledge Graph traversal.
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from core.config import (
    CHROMA_DIR,
    GRAPH_PATH,
    EMBEDDING_MODEL,
    FALLBACK_EMBEDDING_MODEL,
    EMBEDDING_DIM,
    LM_STUDIO_BASE_URL,
    LM_STUDIO_API_KEY,
    TOP_K,
    CHROMA_COLLECTION,
    client,
)
from core.neo4j_engine import neo4j_engine


# ═══════════════════════════════════════════════════════════════════════════════
#  LM STUDIO EMBEDDING WRAPPER (for LangChain/Chroma compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

class LMStudioEmbeddings:
    """
    Custom embedding class supporting Google's EmbeddingGemma-2 and Nomic Embed.
    Includes Matryoshka Representation Learning (MRL) dimensionality truncation
    and dynamic model detection from LM Studio.
    """

    def __init__(self, target_dim: Optional[int] = None):
        self.model = EMBEDDING_MODEL
        self.fallback_model = FALLBACK_EMBEDDING_MODEL
        self.target_dim = target_dim or EMBEDDING_DIM
        self._active_model: Optional[str] = None

    def _resolve_model(self) -> str:
        """Dynamically detect which embedding model is currently loaded in LM Studio."""
        if self._active_model:
            return self._active_model

        try:
            available = [m.id for m in client.models.list().data]
            if not available:
                self._active_model = self.model
                return self._active_model

            # Priority 1: Exact configured EMBEDDING_MODEL
            if self.model in available:
                self._active_model = self.model
            # Priority 2: Any model matching EmbeddingGemma
            elif any("embeddinggemma" in m.lower() for m in available):
                self._active_model = next(m for m in available if "embeddinggemma" in m.lower())
            # Priority 3: Fallback Nomic Embed model
            elif self.fallback_model in available:
                self._active_model = self.fallback_model
            # Priority 4: Any loaded model with 'embed' in name
            elif any("embed" in m.lower() for m in available):
                self._active_model = next(m for m in available if "embed" in m.lower())
            else:
                self._active_model = self.model
        except Exception:
            self._active_model = self.model

        return self._active_model

    def _apply_matryoshka(self, vec: list[float]) -> list[float]:
        """Apply Matryoshka dimension truncation and L2-normalization for EmbeddingGemma."""
        if not self.target_dim or self.target_dim >= len(vec):
            return vec
        arr = np.array(vec[:self.target_dim], dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        model = self._resolve_model()
        embeddings = []
        # Process in batches to avoid overwhelming the local server
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            res = client.embeddings.create(
                model=model,
                input=batch,
            )
            for item in res.data:
                vec = self._apply_matryoshka(item.embedding)
                embeddings.append(vec)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        model = self._resolve_model()
        res = client.embeddings.create(
            model=model,
            input=[text],
        )
        return self._apply_matryoshka(res.data[0].embedding)


# ═══════════════════════════════════════════════════════════════════════════════
#  VECTOR RETRIEVER — ChromaDB
# ═══════════════════════════════════════════════════════════════════════════════

class VectorRetriever:
    """Retrieves relevant document chunks from a persisted ChromaDB collection."""

    def __init__(self):
        self._embeddings = LMStudioEmbeddings()
        self._vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION,
            persist_directory=str(CHROMA_DIR),
            embedding_function=self._embeddings,
        )

    def query(self, text: str, k: Optional[int] = None) -> list[Document]:
        """
        Similarity-search against the vector store.
        
        Returns:
            List of LangChain Document objects with page_content and metadata.
        """
        k = k or TOP_K
        results = self._vectorstore.similarity_search_with_relevance_scores(text, k=k)

        documents = []
        for doc, score in results:
            doc.metadata["relevance_score"] = round(score, 4)
            documents.append(doc)

        return documents

    def format_context(self, documents: list[Document]) -> str:
        """Format retrieved documents into a context string for the LLM."""
        if not documents:
            return "No relevant documents found."

        chunks = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            score = doc.metadata.get("relevance_score", "?")
            chunks.append(
                f"[Chunk {i}] (Source: {source}, Page: {page}, Score: {score})\n"
                f"{doc.page_content}"
            )
        return "\n\n---\n\n".join(chunks)


# ═══════════════════════════════════════════════════════════════════════════════
#  GRAPH RETRIEVER — JSON Knowledge Graph
# ═══════════════════════════════════════════════════════════════════════════════

class GraphRetriever:
    """
    Retrieves relationship triples from a JSON-based knowledge graph.
    
    Expected graph.json format:
    {
        "nodes": [
            {"id": "...", "label": "...", "type": "...", "properties": {...}}
        ],
        "edges": [
            {"source": "...", "target": "...", "relation": "..."}
        ]
    }
    """

    def __init__(self):
        self._graph = self._load_graph()
        # Build lookup indices for fast access
        self._node_index: dict[str, dict] = {}
        self._adjacency: dict[str, list[dict]] = {}
        self._build_indices()

    def _load_graph(self) -> dict:
        """Load the knowledge graph JSON from disk."""
        if not GRAPH_PATH.exists():
            return {"nodes": [], "edges": []}

        with open(GRAPH_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate structure
        if "nodes" not in data or "edges" not in data:
            raise ValueError(
                f"graph.json must have 'nodes' and 'edges' keys. "
                f"Found keys: {list(data.keys())}"
            )
        return data

    def _build_indices(self):
        """Build in-memory indices for O(1) node lookup and adjacency traversal."""
        for node in self._graph.get("nodes", []):
            node_id = node.get("id", "")
            self._node_index[node_id] = node
            # Also index by lowercase label for fuzzy matching
            label = node.get("label", "").lower()
            if label:
                self._node_index[label] = node

        for edge in self._graph.get("edges", []):
            source = edge.get("source", "")
            target = edge.get("target", "")
            # Bidirectional adjacency
            self._adjacency.setdefault(source, []).append(edge)
            self._adjacency.setdefault(target, []).append(edge)

    def query(self, text: str, max_hops: int = 2) -> list[dict]:
        """
        Find relevant triples by matching query keywords against node labels.
        Then traverse edges up to max_hops to gather relationship context.

        Returns:
            List of triple dicts: {"source": ..., "relation": ..., "target": ..., "source_type": ..., "target_type": ...}
        """
        if not self._graph.get("nodes"):
            return []

        # Extract keywords from query (lowercase, >2 chars)
        keywords = [
            w.lower() for w in re.split(r'\W+', text)
            if len(w) > 2
        ]
        if not keywords:
            return []

        # ── 1. Enterprise Neo4j Cypher Traversal (if connected) ─────────────
        if neo4j_engine.is_connected():
            neo_triples = neo4j_engine.multi_hop_query(keywords, max_hops=max_hops)
            if neo_triples:
                return neo_triples

        # ── 2. Local In-Memory Graph Fallback ────────────────────────────────
        if not self._graph.get("nodes"):
            return []

        # Find matching nodes
        matched_node_ids = set()
        for node in self._graph["nodes"]:
            label = node.get("label", "").lower()
            node_id = node.get("id", "")
            for kw in keywords:
                if kw in label or kw in node_id.lower():
                    matched_node_ids.add(node_id)
                    break

        # Traverse edges from matched nodes
        triples = []
        visited_edges = set()
        frontier = matched_node_ids.copy()

        for hop in range(max_hops):
            next_frontier = set()
            for node_id in frontier:
                for edge in self._adjacency.get(node_id, []):
                    edge_key = (edge["source"], edge["relation"], edge["target"])
                    if edge_key not in visited_edges:
                        visited_edges.add(edge_key)

                        source_node = self._node_index.get(edge["source"], {})
                        target_node = self._node_index.get(edge["target"], {})

                        triples.append({
                            "source": source_node.get("label", edge["source"]),
                            "relation": edge.get("relation", "related_to"),
                            "target": target_node.get("label", edge["target"]),
                            "source_type": source_node.get("type", ""),
                            "target_type": target_node.get("type", ""),
                            "hops": hop + 1,
                        })

                        # Expand frontier for next hop
                        other = edge["target"] if edge["source"] == node_id else edge["source"]
                        next_frontier.add(other)

            frontier = next_frontier - matched_node_ids  # Don't re-visit seed nodes

        return triples

    def format_context(self, triples: list[dict]) -> str:
        """Format triples into a readable context string for the LLM."""
        if not triples:
            return "No relevant relationships found in the knowledge graph."

        lines = []
        for t in triples:
            src_type = f" [{t.get('source_type', '')}]" if t.get('source_type') else ""
            tgt_type = f" [{t.get('target_type', '')}]" if t.get('target_type') else ""
            hops_info = f" (Hops: {t.get('hops', 1)})" if "hops" in t else ""
            lines.append(
                f"• {t['source']}{src_type}  —({t['relation']})→  {t['target']}{tgt_type}{hops_info}"
            )

        return "Knowledge Graph Relationships:\n" + "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  HYBRID RAG FUSION — Reciprocal Rank Fusion (RRF)
# ═══════════════════════════════════════════════════════════════════════════════

def reciprocal_rank_fusion(
    vector_docs: list[Document],
    graph_triples: list[dict],
    k: int = 60,
    max_tokens: int = 2500,
) -> str:
    """
    Combines Vector Search (ChromaDB) and Knowledge Graph traversal (Neo4j / In-Memory)
    using Reciprocal Rank Fusion (RRF). Produces a unified, ranked evidence block.
    
    RRF Score formula:
        RRF_Score(d) = sum( 1 / (k + rank) )
    """
    scored_items = []

    # 1. Rank vector documents
    for rank, doc in enumerate(vector_docs, 1):
        score = 1.0 / (k + rank)
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        content = (
            f"[Passage Evidence - Rank {rank} | Source: {source}, Page: {page}]\n"
            f"{doc.page_content.strip()}"
        )
        scored_items.append({"score": score, "text": content, "type": "vector"})

    # 2. Rank graph relationships (higher priority for 1-hop direct relationships)
    for rank, t in enumerate(graph_triples[:15], 1):
        hop_penalty = t.get("hops", 1)
        score = (1.0 / (k + rank)) * (1.0 / hop_penalty)
        content = (
            f"[Graph Relationship Evidence - Rank {rank}]\n"
            f"• {t['source']}  —({t['relation']})→  {t['target']}"
        )
        scored_items.append({"score": score, "text": content, "type": "graph"})

    # 3. Sort all evidence descending by RRF score
    scored_items.sort(key=lambda x: x["score"], reverse=True)

    # 4. Assemble fused context block
    sections = []
    current_len = 0
    for item in scored_items:
        text = item["text"]
        if current_len + len(text) > max_tokens * 4:  # Rough character budget
            break
        sections.append(text)
        current_len += len(text)

    if not sections:
        return "No sufficient context retrieved from vector or graph stores."

    return "=== HYBRID FUSED RETRIEVAL CONTEXT (Vector + Knowledge Graph) ===\n\n" + "\n\n---\n\n".join(sections)

