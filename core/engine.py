"""
Local GraphRAG — LangGraph State Machine Engine
Routes queries between Vector DB and Knowledge Graph, then generates answers.
"""

import json
import re
import threading
import time
from typing import Callable, TypedDict, Literal

from langgraph.graph import StateGraph, START, END

from core.config import (
    client,
    ROUTER_MODEL,
    SYNTHESISER_MODEL,
    ROUTER_SYSTEM_PROMPT,
    GENERATOR_SYSTEM_PROMPT,
    ROUTER_MAX_TOKENS,
    SYNTHESIS_MAX_TOKENS,
    TEMPERATURE,
    GRAPH_PATH,
    UPLOADS_DIR,
)
from core.retrievers import VectorRetriever, GraphRetriever


# ═══════════════════════════════════════════════════════════════════════════════
#  STATE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════

class QueryState(TypedDict):
    """Shared state flowing through every node in the graph."""
    query: str                     # User's input question
    route: str                     # "vector" or "graph"
    context: str                   # Retrieved context (from either source)
    response: str                  # Final LLM-generated answer
    reasoning_trace: list[dict]    # Step-by-step audit trail for the UI
    on_chunk: Callable[[str], bool] | None
    cancel_event: threading.Event | None


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def classify_query(state: QueryState) -> dict:
    """
    Router Node: Uses the LLM to classify the query as 'vector' or 'graph'.
    This determines which retrieval path to take.
    """
    t0 = time.time()

    res = client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": state["query"]},
        ],
        max_tokens=ROUTER_MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    raw_route = res.choices[0].message.content.strip()

    # Gemma 4 thinking mode may emit reasoning tokens before the final answer.
    # Extract the LAST occurrence of "vector" or "graph" (case-insensitive).
    matches = re.findall(r'\b(vector|graph)\b', raw_route, re.IGNORECASE)
    route = matches[-1].lower() if matches else "vector"

    elapsed = round(time.time() - t0, 2)

    trace_entry = {
        "step": "classify_query",
        "route_chosen": route,
        "llm_raw_output": raw_route,
        "latency_sec": elapsed,
        "description": f"Query classified as '{route}' path"
        + (" (definitions/facts → Vector DB)" if route == "vector"
           else " (relationships/comparisons → Knowledge Graph)"),
    }

    return {
        "route": route,
        "reasoning_trace": state.get("reasoning_trace", []) + [trace_entry],
    }


def retrieve_vector(state: QueryState) -> dict:
    """
    Vector Retrieval Node: Searches ChromaDB for relevant document chunks.
    """
    t0 = time.time()

    retriever = VectorRetriever()
    documents = retriever.query(state["query"])
    context = retriever.format_context(documents)

    elapsed = round(time.time() - t0, 2)

    # Build source summary for the trace
    sources = []
    for doc in documents:
        sources.append({
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "?"),
            "score": doc.metadata.get("relevance_score", "?"),
            "preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
        })

    trace_entry = {
        "step": "retrieve_vector",
        "num_chunks": len(documents),
        "sources": sources,
        "latency_sec": elapsed,
        "description": f"Retrieved {len(documents)} chunks from ChromaDB",
    }

    return {
        "context": context,
        "reasoning_trace": state.get("reasoning_trace", []) + [trace_entry],
    }


def retrieve_graph(state: QueryState) -> dict:
    """
    Graph Retrieval Node: Traverses the JSON knowledge graph for relationship triples.
    """
    t0 = time.time()

    retriever = GraphRetriever()
    triples = retriever.query(state["query"])
    context = retriever.format_context(triples)

    elapsed = round(time.time() - t0, 2)

    trace_entry = {
        "step": "retrieve_graph",
        "num_triples": len(triples),
        "triples": triples[:10],  # Cap at 10 for the trace display
        "latency_sec": elapsed,
        "description": f"Found {len(triples)} relationship triples in Knowledge Graph",
    }

    return {
        "context": context,
        "reasoning_trace": state.get("reasoning_trace", []) + [trace_entry],
    }


def generate_answer(state: QueryState) -> dict:
    """
    Generator Node: Uses the LLM to synthesize a grounded answer from retrieved context.
    """
    t0 = time.time()

    system_prompt = GENERATOR_SYSTEM_PROMPT.format(context=state["context"])

    on_chunk = state.get("on_chunk")
    if on_chunk is None:
        res = client.chat.completions.create(
            model=SYNTHESISER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state["query"]},
            ],
            max_tokens=SYNTHESIS_MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        answer = res.choices[0].message.content.strip()
    else:
        answer_parts = []
        cancel_event = state.get("cancel_event")
        with client.chat.completions.create(
            model=SYNTHESISER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state["query"]},
            ],
            max_tokens=SYNTHESIS_MAX_TOKENS,
            temperature=TEMPERATURE,
            stream=True,
        ) as stream:
            finished = threading.Event()
            stream_cancelled = threading.Event()
            watcher = None
            if cancel_event is not None:
                def close_stream_on_cancel():
                    while not finished.wait(0.1):
                        if cancel_event.is_set():
                            stream_cancelled.set()
                            stream.close()
                            return

                watcher = threading.Thread(target=close_stream_on_cancel, daemon=True)
                watcher.start()
            try:
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    text = chunk.choices[0].delta.content
                    if text:
                        answer_parts.append(text)
                        if on_chunk(text):
                            break
            except Exception:
                if not stream_cancelled.is_set():
                    raise
            finally:
                finished.set()
                if watcher is not None:
                    watcher.join(timeout=0.2)
        answer = "".join(answer_parts).strip()

    elapsed = round(time.time() - t0, 2)

    trace_entry = {
        "step": "generate_answer",
        "latency_sec": elapsed,
        "description": f"Generated answer using {SYNTHESISER_MODEL} ({elapsed}s)",
    }

    return {
        "response": answer,
        "reasoning_trace": state.get("reasoning_trace", []) + [trace_entry],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def route_decision(state: QueryState) -> Literal["retrieve_vector", "retrieve_graph"]:
    """Conditional edge: routes to the appropriate retriever based on classification."""
    if state.get("route") == "graph":
        return "retrieve_graph"
    return "retrieve_vector"


# ═══════════════════════════════════════════════════════════════════════════════
#  GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph state machine.

    Flow:
        START → classify_query ─┬─→ retrieve_vector ─→ generate_answer → END
                                └─→ retrieve_graph  ─→ generate_answer → END
    """
    builder = StateGraph(QueryState)

    # Register nodes
    builder.add_node("classify_query", classify_query)
    builder.add_node("retrieve_vector", retrieve_vector)
    builder.add_node("retrieve_graph", retrieve_graph)
    builder.add_node("generate_answer", generate_answer)

    # Wire edges
    builder.add_edge(START, "classify_query")

    builder.add_conditional_edges(
        "classify_query",
        route_decision,
        {
            "retrieve_vector": "retrieve_vector",
            "retrieve_graph": "retrieve_graph",
        },
    )

    builder.add_edge("retrieve_vector", "generate_answer")
    builder.add_edge("retrieve_graph", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder.compile()


# ═══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

# Compiled graph singleton — import this in the API layer
graph = build_graph()


def run_query(
    query: str,
    on_chunk: Callable[[str], bool] | None = None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """
    Execute a full query through the LangGraph pipeline.
    
    Args:
        query: The user's question.
        
    Returns:
        Final state dict with 'response', 'route', 'reasoning_trace', etc.
    """
    initial_state: QueryState = {
        "query": query,
        "route": "",
        "context": "",
        "response": "",
        "reasoning_trace": [],
        "on_chunk": on_chunk,
        "cancel_event": cancel_event,
    }

    result = graph.invoke(initial_state)
    result.pop("on_chunk", None)
    result.pop("cancel_event", None)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  DYNAMIC TECHNICAL INQUIRY SUGGESTIONS (LLM-POWERED)
# ═══════════════════════════════════════════════════════════════════════════════

def get_knowledge_context_summary() -> dict:
    """Extract summary of knowledge base: document sources, top concepts, and key relationship triples."""
    sources = set()
    top_concepts = []
    sample_triples = []

    if GRAPH_PATH.exists() and GRAPH_PATH.stat().st_size > 0:
        try:
            with open(GRAPH_PATH, "r", encoding="utf-8") as f:
                gdata = json.load(f)
                nodes = gdata.get("nodes", [])
                edges = gdata.get("edges", [])

                for n in nodes:
                    src = n.get("properties", {}).get("source")
                    if src:
                        sources.add(src)

                degree = {}
                node_map = {}
                for n in nodes:
                    nid = n.get("id", "")
                    node_map[nid] = n.get("label", nid)
                    degree[nid] = 0
                for e in edges:
                    s = e.get("source", "")
                    t = e.get("target", "")
                    degree[s] = degree.get(s, 0) + 1
                    degree[t] = degree.get(t, 0) + 1

                sorted_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)
                top_concepts = [node_map[nid] for nid, _ in sorted_nodes[:12] if nid in node_map]

                for e in edges[:8]:
                    s_label = node_map.get(e.get("source"), e.get("source"))
                    t_label = node_map.get(e.get("target"), e.get("target"))
                    rel = e.get("relation", "relates_to")
                    if s_label and t_label:
                        sample_triples.append(f"{s_label} —({rel})→ {t_label}")
        except Exception:
            pass

    if UPLOADS_DIR.exists():
        for p in UPLOADS_DIR.iterdir():
            if p.suffix.lower() in {".pdf", ".docx", ".doc"}:
                sources.add(p.name)

    return {
        "sources": sorted(list(sources)),
        "top_concepts": top_concepts,
        "sample_triples": sample_triples,
    }


def _parse_suggested_questions(raw_text: str, num_expected: int = 3) -> list[str]:
    """Robust parser for LLM-generated questions in JSON array or numbered/bullet format."""
    cleaned = raw_text.strip()
    clean_fenced = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    clean_fenced = re.sub(r"\s*```$", "", clean_fenced, flags=re.MULTILINE).strip()
    try:
        data = json.loads(clean_fenced)
        if isinstance(data, list):
            items = [str(x).strip() for x in data if str(x).strip()]
            if len(items) >= num_expected:
                return items[:num_expected]
            elif items:
                return items
    except Exception:
        pass

    match = re.search(r"\[\s*\"(?:[^\"\\]|\\.)*\".*?\]", raw_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                items = [str(x).strip() for x in data if str(x).strip()]
                if len(items) >= num_expected:
                    return items[:num_expected]
                elif items:
                    return items
        except Exception:
            pass

    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    extracted = []
    for line in lines:
        c = re.sub(r"^[\d\.\-\*\•\)\s]+", "", line).strip().strip('"\'')
        if len(c) > 8 and (c.endswith("?") or any(c.lower().startswith(w) for w in ["what", "how", "why", "which", "compare", "describe", "explain", "can", "is", "does", "where"])):
            if not c.endswith("?"):
                c += "?"
            extracted.append(c)
            if len(extracted) == num_expected:
                return extracted

    return extracted


def _build_fallback_questions(sources: list[str], top_concepts: list[str], has_convo: bool) -> list[str]:
    """Generate dynamic, context-aware fallback questions when LLM is offline or timed out."""
    c1 = top_concepts[0] if len(top_concepts) > 0 else "Local GraphRAG"
    c2 = top_concepts[1] if len(top_concepts) > 1 else "vector similarity search"
    c3 = top_concepts[2] if len(top_concepts) > 2 else "knowledge graph traversal"
    doc = sources[0] if sources else "the ingested documents"

    if has_convo:
        return [
            f"How does {c1} directly interact with {c2} in this architecture?",
            f"What are the primary performance trade-offs when scaling {c1}?",
            f"What specific evidence in {doc} supports this conclusion?",
        ]
    else:
        return [
            f"What are the core technical concepts and methodologies covered in {doc}?",
            f"How does {c1} connect to {c2} across the knowledge base?",
            f"What role does {c3} play in system operations and data flow?",
        ]


def generate_suggested_inquiries(
    conversation_history: list[dict] | None = None,
    num_suggestions: int = 3,
) -> dict:
    """
    Dynamically generates contextual technical inquiries using the local LLM.
    Adapts based on whether there is an ongoing conversation or knowledge base context.

    Returns:
        {"suggestions": list[str], "context_type": str}
    """
    context = get_knowledge_context_summary()
    sources = context.get("sources", [])
    top_concepts = context.get("top_concepts", [])
    sample_triples = context.get("sample_triples", [])

    has_convo = bool(conversation_history and len(conversation_history) > 0)
    context_type = "conversation" if has_convo else "knowledge_base"

    docs_summary = ", ".join(sources) if sources else "System Architectural Documents"
    concepts_summary = ", ".join(top_concepts[:8]) if top_concepts else "Local GraphRAG Dual-Retrieval, ChromaDB, NetworkX Graph"
    triples_summary = "\n".join(sample_triples[:6]) if sample_triples else "None"

    if has_convo:
        recent_msgs = conversation_history[-4:]
        convo_lines = []
        for m in recent_msgs:
            role = m.get("role", "user").capitalize()
            content = m.get("content", "").strip()
            if len(content) > 200:
                content = content[:200] + "..."
            convo_lines.append(f"{role}: {content}")
        convo_str = "\n".join(convo_lines)

        user_prompt = (
            f"Knowledge Base Scope:\n"
            f"Documents: {docs_summary}\n"
            f"Core Concepts: {concepts_summary}\n\n"
            f"Recent Conversation:\n"
            f"{convo_str}\n\n"
            f"Generate {num_suggestions} insightful, technically rigorous follow-up questions that probe deeper into these concepts or clarify implications."
        )
    else:
        user_prompt = (
            f"Knowledge Base Scope:\n"
            f"Documents: {docs_summary}\n"
            f"Core Concepts: {concepts_summary}\n"
            f"Sample Relationships:\n{triples_summary}\n\n"
            f"Generate {num_suggestions} compelling, technical inquiry questions that a student or researcher would ask to investigate this knowledge base."
        )

    system_prompt = (
        f"You are an academic inquiry generator for a GraphRAG system. "
        f"Generate exactly {num_suggestions} concise, specific, and technically insightful questions based on the provided context. "
        f"Each question must be a direct single sentence under 18 words ending with '?'. "
        f"Respond ONLY with a valid JSON array of {num_suggestions} strings. Do NOT output markdown code blocks or extra text.\n"
        f"Example:\n[\"Question 1?\", \"Question 2?\", \"Question 3?\"]"
    )

    try:
        res = client.chat.completions.create(
            model=SYNTHESISER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=220,
            temperature=0.35,
        )
        raw_output = res.choices[0].message.content.strip()
        parsed = _parse_suggested_questions(raw_output, num_expected=num_suggestions)
        if len(parsed) >= num_suggestions:
            return {"suggestions": parsed[:num_suggestions], "context_type": context_type}
        elif parsed:
            fallbacks = _build_fallback_questions(sources, top_concepts, has_convo)
            for fb in fallbacks:
                if fb not in parsed:
                    parsed.append(fb)
                if len(parsed) >= num_suggestions:
                    break
            return {"suggestions": parsed[:num_suggestions], "context_type": context_type}
    except Exception as exc:
        print(f"Warning: Failed to generate suggestions via LLM: {exc}")

    fallback_qs = _build_fallback_questions(sources, top_concepts, has_convo)[:num_suggestions]
    return {"suggestions": fallback_qs, "context_type": context_type}

