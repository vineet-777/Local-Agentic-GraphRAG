"""
Local GraphRAG — LangGraph State Machine Engine
Routes queries between Vector DB and Knowledge Graph, then generates answers.
"""

import re
import time
from typing import TypedDict, Literal

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


def run_query(query: str) -> dict:
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
    }

    result = graph.invoke(initial_state)
    return result
