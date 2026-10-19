"""
Local GraphRAG — LangGraph State Machine Engine (God-Level Edition)
Features:
- Multi-Turn Conversation Memory (contextual follow-ups)
- Adaptive 3-Way Routing: Vector, Knowledge Graph (Neo4j/JSON), or Hybrid
- Reciprocal Rank Fusion (RRF) for Hybrid evidence ranking
- Live Token Streaming with Real-Time Reasoning Traces via SSE
"""

import re
import time
import json
from typing import TypedDict, Literal, Optional, List, Dict, AsyncGenerator

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
from core.retrievers import VectorRetriever, GraphRetriever, reciprocal_rank_fusion


# ═══════════════════════════════════════════════════════════════════════════════
#  STATE DEFINITION (With Multi-Turn Memory)
# ═══════════════════════════════════════════════════════════════════════════════

class QueryState(TypedDict):
    """Shared state flowing through every node in the graph."""
    query: str                         # User's current input question
    conversation_history: List[dict]   # Past turns: [{"role": "user"|"assistant", "content": "..."}]
    route: str                         # "vector", "graph", or "hybrid"
    context: str                       # Retrieved & fused context
    response: str                      # Final LLM-generated answer
    reasoning_trace: List[dict]        # Step-by-step audit trail for the UI


# ═══════════════════════════════════════════════════════════════════════════════
#  NODE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def classify_query(state: QueryState) -> dict:
    """
    Router Node: Uses Gemma 4 to classify the query as 'vector', 'graph', or 'hybrid'.
    Incorporates conversational context for follow-up questions.
    """
    t0 = time.time()

    # If conversation history exists, provide context summary to router
    history = state.get("conversation_history", [])
    history_context = ""
    if history:
        recent = history[-4:]  # Last 2 conversation turns
        formatted_history = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in recent])
        history_context = f"Recent Conversation Context:\n{formatted_history}\n\nCurrent User Query:\n"

    routing_input = f"{history_context}{state['query']}"

    res = client.chat.completions.create(
        model=ROUTER_MODEL,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": routing_input},
        ],
        max_tokens=ROUTER_MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    raw_route = res.choices[0].message.content.strip()

    # Extract the LAST occurrence of "vector", "graph", or "hybrid" (case-insensitive)
    matches = re.findall(r'\b(vector|graph|hybrid)\b', raw_route, re.IGNORECASE)
    route = matches[-1].lower() if matches else "vector"

    elapsed = round(time.time() - t0, 2)

    desc_map = {
        "vector": "Query classified as 'vector' (semantic search in ChromaDB passages)",
        "graph": "Query classified as 'graph' (entity-relational traversal in Knowledge Graph / Neo4j)",
        "hybrid": "Query classified as 'hybrid' (Reciprocal Rank Fusion across Vector + Graph)",
    }

    trace_entry = {
        "step": "classify_query",
        "route_chosen": route,
        "llm_raw_output": raw_route,
        "latency_sec": elapsed,
        "description": desc_map.get(route, f"Query classified as '{route}' path"),
    }

    return {
        "route": route,
        "reasoning_trace": state.get("reasoning_trace", []) + [trace_entry],
    }


def retrieve_vector(state: QueryState) -> dict:
    """Vector Retrieval Node: Searches ChromaDB for relevant document chunks."""
    t0 = time.time()

    retriever = VectorRetriever()
    documents = retriever.query(state["query"])
    context = retriever.format_context(documents)

    elapsed = round(time.time() - t0, 2)

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
    """Graph Retrieval Node: Traverses Neo4j / in-memory graph for relationship triples."""
    t0 = time.time()

    retriever = GraphRetriever()
    triples = retriever.query(state["query"])
    context = retriever.format_context(triples)

    elapsed = round(time.time() - t0, 2)

    trace_entry = {
        "step": "retrieve_graph",
        "num_triples": len(triples),
        "triples": triples[:10],
        "latency_sec": elapsed,
        "description": f"Found {len(triples)} relationship triples in Knowledge Graph",
    }

    return {
        "context": context,
        "reasoning_trace": state.get("reasoning_trace", []) + [trace_entry],
    }


def retrieve_hybrid(state: QueryState) -> dict:
    """
    Hybrid Retrieval Node: Queries BOTH Vector (ChromaDB) and Graph (Neo4j / JSON),
    then fuses them using Reciprocal Rank Fusion (RRF).
    """
    t0 = time.time()

    vec_retriever = VectorRetriever()
    graph_retriever = GraphRetriever()

    vector_docs = vec_retriever.query(state["query"], k=4)
    graph_triples = graph_retriever.query(state["query"], max_hops=2)

    fused_context = reciprocal_rank_fusion(vector_docs, graph_triples, k=60)

    elapsed = round(time.time() - t0, 2)

    trace_entry = {
        "step": "retrieve_hybrid",
        "vector_chunks": len(vector_docs),
        "graph_triples": len(graph_triples),
        "fusion_algorithm": "Reciprocal Rank Fusion (RRF)",
        "latency_sec": elapsed,
        "description": f"RRF fused {len(vector_docs)} vector chunks + {len(graph_triples)} graph triples",
    }

    return {
        "context": fused_context,
        "reasoning_trace": state.get("reasoning_trace", []) + [trace_entry],
    }


def generate_answer(state: QueryState) -> dict:
    """
    Generator Node: Uses Gemma 4 to synthesize grounded answer from retrieved context.
    Passes conversation history for contextual dialogue.
    """
    t0 = time.time()

    system_prompt = GENERATOR_SYSTEM_PROMPT.format(context=state["context"])

    messages = [{"role": "system", "content": system_prompt}]
    
    # Append past conversation turns if available
    for m in state.get("conversation_history", [])[-4:]:
        messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

    messages.append({"role": "user", "content": state["query"]})

    res = client.chat.completions.create(
        model=SYNTHESISER_MODEL,
        messages=messages,
        max_tokens=SYNTHESIS_MAX_TOKENS,
        temperature=TEMPERATURE,
    )
    answer = res.choices[0].message.content.strip()

    elapsed = round(time.time() - t0, 2)

    trace_entry = {
        "step": "generate_answer",
        "latency_sec": elapsed,
        "description": f"Synthesized grounded answer using {SYNTHESISER_MODEL} ({elapsed}s)",
    }

    return {
        "response": answer,
        "reasoning_trace": state.get("reasoning_trace", []) + [trace_entry],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

def route_decision(state: QueryState) -> Literal["retrieve_vector", "retrieve_graph", "retrieve_hybrid"]:
    """Conditional edge: routes to vector, graph, or hybrid retriever."""
    r = state.get("route", "vector")
    if r == "graph":
        return "retrieve_graph"
    elif r == "hybrid":
        return "retrieve_hybrid"
    return "retrieve_vector"


# ═══════════════════════════════════════════════════════════════════════════════
#  GRAPH CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph() -> StateGraph:
    """Constructs and compiles the 3-Way LangGraph state machine."""
    builder = StateGraph(QueryState)

    builder.add_node("classify_query", classify_query)
    builder.add_node("retrieve_vector", retrieve_vector)
    builder.add_node("retrieve_graph", retrieve_graph)
    builder.add_node("retrieve_hybrid", retrieve_hybrid)
    builder.add_node("generate_answer", generate_answer)

    builder.add_edge(START, "classify_query")

    builder.add_conditional_edges(
        "classify_query",
        route_decision,
        {
            "retrieve_vector": "retrieve_vector",
            "retrieve_graph": "retrieve_graph",
            "retrieve_hybrid": "retrieve_hybrid",
        },
    )

    builder.add_edge("retrieve_vector", "generate_answer")
    builder.add_edge("retrieve_graph", "generate_answer")
    builder.add_edge("retrieve_hybrid", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder.compile()


graph = build_graph()


# ═══════════════════════════════════════════════════════════════════════════════
#  EXECUTION ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_query(query: str, conversation_history: Optional[List[dict]] = None) -> dict:
    """
    Synchronous full query execution.
    """
    initial_state: QueryState = {
        "query": query,
        "conversation_history": conversation_history or [],
        "route": "",
        "context": "",
        "response": "",
        "reasoning_trace": [],
    }
    return graph.invoke(initial_state)


async def stream_query_events(
    query: str,
    conversation_history: Optional[List[dict]] = None
) -> AsyncGenerator[str, None]:
    """
    Asynchronous Server-Sent Events (SSE) generator.
    Yields step-by-step reasoning traces followed by live token-by-token text generation.
    """
    state: QueryState = {
        "query": query,
        "conversation_history": conversation_history or [],
        "route": "",
        "context": "",
        "response": "",
        "reasoning_trace": [],
    }

    # 1. Step: Router Classification
    classified = classify_query(state)
    state.update(classified)
    yield f"data: {json.dumps({'event': 'trace', 'data': state['reasoning_trace'][-1]})}\n\n"

    # 2. Step: Retrieval (Vector / Graph / Hybrid)
    route = state.get("route", "vector")
    if route == "graph":
        retrieved = retrieve_graph(state)
    elif route == "hybrid":
        retrieved = retrieve_hybrid(state)
    else:
        retrieved = retrieve_vector(state)

    state.update(retrieved)
    yield f"data: {json.dumps({'event': 'trace', 'data': state['reasoning_trace'][-1]})}\n\n"

    # 3. Step: Streaming Synthesis via Gemma 4
    system_prompt = GENERATOR_SYSTEM_PROMPT.format(context=state["context"])
    messages = [{"role": "system", "content": system_prompt}]
    for m in state.get("conversation_history", [])[-4:]:
        messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    messages.append({"role": "user", "content": state["query"]})

    yield f"data: {json.dumps({'event': 'start_generation', 'route': route})}\n\n"

    stream = client.chat.completions.create(
        model=SYNTHESISER_MODEL,
        messages=messages,
        max_tokens=SYNTHESIS_MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=True,
    )

    full_answer = []
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else ""
        if delta:
            full_answer.append(delta)
            yield f"data: {json.dumps({'event': 'token', 'token': delta})}\n\n"

    yield f"data: {json.dumps({'event': 'done', 'response': ''.join(full_answer)})}\n\n"
