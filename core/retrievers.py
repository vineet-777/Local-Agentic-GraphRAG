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
    LM_STUDIO_BASE_URL,
    LM_STUDIO_API_KEY,
    TOP_K,
    CHROMA_COLLECTION,
    client,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  LM STUDIO EMBEDDING WRAPPER (for LangChain/Chroma compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

class LMStudioEmbeddings:
    """
    Custom embedding class that uses LM Studio's OpenAI-compatible
    embeddings endpoint via the shared OpenAI client from config.
    Implements the interface expected by langchain_chroma.
    """

    def __init__(self):
        self.model = EMBEDDING_MODEL

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents."""
        embeddings = []
        # Process in batches to avoid overwhelming the local server
        batch_size = 16
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            res = client.embeddings.create(
                model=self.model,
                input=batch,
            )
            for item in res.data:
                embeddings.append(item.embedding)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        res = client.embeddings.create(
            model=self.model,
            input=[text],
        )
        return res.data[0].embedding


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
            src_type = f" [{t['source_type']}]" if t['source_type'] else ""
            tgt_type = f" [{t['target_type']}]" if t['target_type'] else ""
            lines.append(
                f"• {t['source']}{src_type}  —({t['relation']})→  {t['target']}{tgt_type}"
            )

        return "Knowledge Graph Relationships:\n" + "\n".join(lines)
