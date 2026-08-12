"""
Local GraphRAG — Knowledge Graph Builder
Uses a fast local LLM to extract entity-relationship triples from text chunks,
then merges them into the persistent graph.json.
"""

import json
import re
import hashlib
from pathlib import Path

from langchain_core.documents import Document

from core.config import (
    GRAPH_PATH,
    ROUTER_MODEL,
    TRIPLE_EXTRACTION_SYSTEM_PROMPT,
    TRIPLE_EXTRACTION_USER_PROMPT,
    processing_status,
    client,
    EXTRACTION_MAX_TOKENS,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  GRAPH STORAGE
# ═══════════════════════════════════════════════════════════════════════════════

def load_graph() -> dict:
    """Load existing graph from disk, or return empty structure."""
    if GRAPH_PATH.exists() and GRAPH_PATH.stat().st_size > 0:
        try:
            with open(GRAPH_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Ensure correct structure
            if "nodes" in data and "edges" in data:
                return data
        except (json.JSONDecodeError, KeyError):
            pass

    return {"nodes": [], "edges": []}


def save_graph(graph: dict):
    """Persist graph to disk."""
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_PATH, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  TRIPLE EXTRACTION VIA LLM
# ═══════════════════════════════════════════════════════════════════════════════

def _make_node_id(label: str) -> str:
    """Create a deterministic node ID from a label."""
    normalized = label.strip().lower()
    return hashlib.md5(normalized.encode()).hexdigest()[:10]


def extract_triples_from_chunk(chunk_text: str) -> list[dict]:
    """
    Use Gemma 4 E4B via LM Studio to extract knowledge graph triples
    from a text chunk.
    
    Returns:
        List of dicts: [{"subject": ..., "relation": ..., "object": ...}]
    """
    user_prompt = TRIPLE_EXTRACTION_USER_PROMPT.format(text=chunk_text[:1500])  # Cap input size

    try:
        res = client.chat.completions.create(
            model=ROUTER_MODEL,  # Gemma 4 E4B — used for all LLM tasks
            messages=[
                {"role": "system", "content": TRIPLE_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=EXTRACTION_MAX_TOKENS,
            temperature=0.0,  # Deterministic for structured extraction
        )

        raw = res.choices[0].message.content.strip()

        # Try to extract JSON from the response (LLM may wrap in markdown)
        json_match = re.search(r'\[.*\]', raw, re.DOTALL)
        if json_match:
            triples = json.loads(json_match.group())
            # Validate structure
            valid = []
            for t in triples:
                if isinstance(t, dict) and "subject" in t and "object" in t:
                    valid.append({
                        "subject": str(t["subject"]).strip(),
                        "relation": str(t.get("relation", "related_to")).strip(),
                        "object": str(t["object"]).strip(),
                    })
            return valid

    except (json.JSONDecodeError, Exception):
        pass  # Skip malformed responses silently

    return []


# ═══════════════════════════════════════════════════════════════════════════════
#  GRAPH MERGING
# ═══════════════════════════════════════════════════════════════════════════════

def merge_triples_into_graph(triples: list[dict], source_file: str, graph: dict) -> dict:
    """
    Merge extracted triples into the existing graph.
    Deduplicates nodes by label and edges by (source, relation, target).
    
    Args:
        triples: List of {"subject": ..., "relation": ..., "object": ...}
        source_file: Name of the source document
        graph: Current graph dict
        
    Returns:
        Updated graph dict
    """
    # Build lookup sets for existing data
    existing_node_ids = {n["id"] for n in graph["nodes"]}
    existing_edges = {
        (e["source"], e["relation"], e["target"])
        for e in graph["edges"]
    }

    for triple in triples:
        subj_label = triple["subject"]
        obj_label = triple["object"]
        relation = triple["relation"]

        subj_id = _make_node_id(subj_label)
        obj_id = _make_node_id(obj_label)

        # Add subject node if new
        if subj_id not in existing_node_ids:
            graph["nodes"].append({
                "id": subj_id,
                "label": subj_label,
                "type": "concept",
                "properties": {"source": source_file},
            })
            existing_node_ids.add(subj_id)

        # Add object node if new
        if obj_id not in existing_node_ids:
            graph["nodes"].append({
                "id": obj_id,
                "label": obj_label,
                "type": "concept",
                "properties": {"source": source_file},
            })
            existing_node_ids.add(obj_id)

        # Add edge if new
        edge_key = (subj_id, relation, obj_id)
        if edge_key not in existing_edges:
            graph["edges"].append({
                "source": subj_id,
                "relation": relation,
                "target": obj_id,
            })
            existing_edges.add(edge_key)

    return graph


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL GRAPH BUILDING PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph_from_chunks(chunks: list[Document], source_file: str) -> dict:
    """
    Extract triples from all chunks and merge into the persistent graph.
    Uses Gemma 4 E4B for triple extraction via LM Studio.
    
    Returns:
        Stats dict with extraction results.
    """
    stats = {
        "file": source_file,
        "chunks_processed": 0,
        "triples_extracted": 0,
        "nodes_added": 0,
        "edges_added": 0,
    }

    # Load existing graph
    graph = load_graph()
    initial_nodes = len(graph["nodes"])
    initial_edges = len(graph["edges"])

    all_triples = []

    for i, chunk in enumerate(chunks):
        processing_status.update(
            message=f"🔗 Extracting triples from chunk {i+1}/{len(chunks)} of {source_file}...",
        )

        triples = extract_triples_from_chunk(chunk.page_content)
        all_triples.extend(triples)
        stats["chunks_processed"] += 1

    stats["triples_extracted"] = len(all_triples)

    # Merge into graph
    if all_triples:
        graph = merge_triples_into_graph(all_triples, source_file, graph)
        save_graph(graph)

    stats["nodes_added"] = len(graph["nodes"]) - initial_nodes
    stats["edges_added"] = len(graph["edges"]) - initial_edges

    return stats
