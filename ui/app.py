"""
Local GraphRAG — Streamlit Frontend v3.0 (ChatGPT / OpenAI Codex Edition)
Ultra-Modern Conversational Interface with OpenAI o1-Style Thinking Blocks,
Gemma 4 + EmbeddingGemma-2 Model Telemetry, and Dynamic Graph Ingestion.
"""

from pathlib import Path
import time
import requests
import streamlit as st

# ─── Configuration ────────────────────────────────────────────────────────────

API_BASE = "http://127.0.0.1:8000"
API_QUERY = f"{API_BASE}/api/query"
API_HEALTH = f"{API_BASE}/api/health"
API_UPLOAD = f"{API_BASE}/api/upload"
API_UPLOAD_STATUS = f"{API_BASE}/api/upload-status"

THEME_CSS_PATH = Path(__file__).parent / "chatgpt_theme.css"

# ─── Page Setup ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ChatGPT — Local GraphRAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Load Custom ChatGPT Theme CSS ───────────────────────────────────────────

if THEME_CSS_PATH.exists():
    with open(THEME_CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ─── Session State Management ────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "traces" not in st.session_state:
    st.session_state.traces = {}
if "system_healthy" not in st.session_state:
    st.session_state.system_healthy = None
if "health_data" not in st.session_state:
    st.session_state.health_data = None
if "upload_triggered" not in st.session_state:
    st.session_state.upload_triggered = False
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ─── API Helper Functions ─────────────────────────────────────────────────────

def check_health():
    """Poll backend health status."""
    try:
        resp = requests.get(API_HEALTH, timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.health_data = data
            st.session_state.system_healthy = data.get("lm_studio", {}).get("running", False)
            return data
        st.session_state.system_healthy = False
        return None
    except requests.ConnectionError:
        st.session_state.system_healthy = False
        st.session_state.health_data = None
        return None


def get_processing_status():
    """Poll background document ingestion progress."""
    try:
        resp = requests.get(API_UPLOAD_STATUS, timeout=3)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def upload_files_to_api(files):
    """Dispatch uploaded files to FastAPI."""
    try:
        file_tuples = [("files", (f.name, f.getvalue(), "application/octet-stream")) for f in files]
        resp = requests.post(API_UPLOAD, files=file_tuples, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 409:
            return {"error": "A document ingestion job is already running."}
        elif resp.status_code == 503:
            return {"error": resp.json().get("detail", "LM Studio or Backend unavailable.")}
        return {"error": f"Upload failed ({resp.status_code}): {resp.text}"}
    except requests.ConnectionError:
        return {"error": "Cannot connect to FastAPI backend on port 8000."}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR (ChatGPT Conversation & Document Management Drawer)
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ─── Brand & New Chat ─────────────────────────────────────────────────
    st.markdown("""
    <div class="sidebar-brand-title">
        <span style="font-size:1.15rem;">🧠</span> Local GraphRAG
    </div>
    """, unsafe_allow_html=True)

    if st.button("➕ New Chat"):
        st.session_state.messages = []
        st.session_state.traces = {}
        st.session_state.pending_query = None
        st.rerun()

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # ─── Section 1: Document Upload ───────────────────────────────────────
    st.markdown("<div class='sidebar-label'>Knowledge Ingestion</div>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Attach documents (PDF, DOCX)",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="file_uploader",
        help="Documents are chunked, embedded into ChromaDB, and extracted into the Knowledge Graph 100% locally.",
    )

    if uploaded_files:
        st.caption(f"📁 {len(uploaded_files)} file(s) staged")
        if st.button("⚡ Index Documents", type="primary"):
            res = upload_files_to_api(uploaded_files)
            if "error" in res:
                st.error(res["error"])
            else:
                st.success(res.get("message", "Processing started in background!"))
                st.session_state.upload_triggered = True
                time.sleep(0.4)
                st.rerun()

    # ─── Section 2: Real-Time Processing Status ───────────────────────────
    proc_status = get_processing_status()

    if proc_status and proc_status.get("is_processing"):
        st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)
        progress_val = proc_status.get("progress", 0.0)
        stage = proc_status.get("stage", "indexing")
        message = proc_status.get("message", "Processing chunks...")

        stage_icons = {
            "extracting": "📄",
            "chunking": "✂️",
            "embedding": "🧮",
            "graphing": "🔗",
        }
        icon = stage_icons.get(stage, "⚙️")

        st.markdown(f"""
        <div class="codex-progress-card">
            <div class="codex-progress-stage">{icon} {stage.upper()}</div>
            <div class="codex-progress-msg">{message}</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(progress_val, text=f"{int(progress_val * 100)}%")
        time.sleep(2)
        st.rerun()

    elif proc_status and proc_status.get("stage") == "done" and st.session_state.upload_triggered:
        st.success("✅ Ingestion & Knowledge Graph update complete!")
        st.session_state.upload_triggered = False
        time.sleep(1)
        st.rerun()

    # ─── Section 3: Telemetry & Models ────────────────────────────────────
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-label'>System Telemetry</div>", unsafe_allow_html=True)

    health_info = check_health()

    if health_info:
        lm_info = health_info.get("lm_studio", {})
        db_info = health_info.get("database", {})

        lm_running = lm_info.get("running", False)
        db_ready = db_info.get("ready", False)

        embed_model = lm_info.get("configured_embed_model", "embeddinggemma-2")
        short_embed = "EmbeddingGemma-2" if "embeddinggemma" in embed_model.lower() else "Nomic Embed"
        embed_dim = lm_info.get("embedding_dim", 768)

        # Status Badges
        c1, c2 = st.columns(2)
        with c1:
            if lm_running:
                st.markdown('<span class="status-badge status-ok">● LM Studio</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge status-err">● Offline</span>', unsafe_allow_html=True)
        with c2:
            if db_ready:
                st.markdown('<span class="status-badge status-ok">● Database</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge status-warn">● Empty</span>', unsafe_allow_html=True)

        # Telemetry Stat Pills
        stats_markup = f"""
        <div style="margin-top: 10px;">
            <div class="codex-stat-pill">⚡ Gemma 4 E4B</div>
            <div class="codex-stat-pill">🧬 {short_embed} ({embed_dim}d)</div>
        """
        if db_info.get("graph_nodes", 0) > 0:
            stats_markup += f"""
            <div class="codex-stat-pill">🔗 {db_info['graph_nodes']} nodes</div>
            <div class="codex-stat-pill">↔️ {db_info['graph_edges']} edges</div>
            """
        if db_info.get("chromadb_exists"):
            stats_markup += '<div class="codex-stat-pill">📄 ChromaDB Active</div>'

        stats_markup += "</div>"
        st.markdown(stats_markup, unsafe_allow_html=True)

    else:
        st.markdown('<span class="status-badge status-err">● API Disconnected</span>', unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Clear All Messages"):
        st.session_state.messages = []
        st.session_state.traces = {}
        st.session_state.pending_query = None
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN CONVERSATIONAL CANVAS (ChatGPT Spine)
# ═══════════════════════════════════════════════════════════════════════════════

# ─── Top Minimal Model Bar ───────────────────────────────────────────────────

health_info = st.session_state.health_data
active_embed = "EmbeddingGemma-2"
if health_info and "embeddinggemma" not in health_info.get("lm_studio", {}).get("configured_embed_model", "").lower():
    active_embed = "Nomic Embed"

st.markdown(f"""
<div class="chatgpt-topbar">
    <div class="chatgpt-model-badge">
        <span>🧠</span> Gemma 4 E4B <span style="color:var(--chatgpt-text-muted);font-weight:400;">•</span> 
        <span style="color:#ab68ff;">GraphRAG</span> + <span style="color:#38bdf8;">{active_embed}</span>
    </div>
    <div class="chatgpt-status-pill">
        <span class="status-dot-active"></span> 100% Local (4GB VRAM)
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Empty State (ChatGPT Landing View) ───────────────────────────────────────

if not st.session_state.messages:
    st.markdown("""
    <div class="chatgpt-hero-container">
        <div class="chatgpt-hero-logo">
            <span style="font-size:1.6rem;">🧠</span>
        </div>
        <div class="chatgpt-hero-title">What would you like to explore?</div>
        <div class="chatgpt-hero-subtitle">
            Ask questions across your documents. The system automatically routes between 
            vector similarity search and knowledge graph traversal.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Quick Starter Cards
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("💡 What is Local GraphRAG?\nExplain the core architecture and purpose."):
            st.session_state.pending_query = "What is Local GraphRAG and how does it work?"
            st.rerun()
        if st.button("🔍 Vector vs Knowledge Graph\nHow does the routing agent decide between them?"):
            st.session_state.pending_query = "How does the routing agent decide between vector search and knowledge graph traversal?"
            st.rerun()

    with col_b:
        if st.button("⚡ Offline Edge Deployment\nHow does it run on consumer 4GB VRAM hardware?"):
            st.session_state.pending_query = "How is this system optimized to run completely offline on a 4GB VRAM GPU?"
            st.rerun()
        if st.button("🔗 Concept Relationship Analysis\nExplain multi-hop graph reasoning."):
            st.session_state.pending_query = "Explain how multi-hop graph reasoning extracts relationship triples from documents."
            st.rerun()


# ─── Chat History Display with OpenAI o1 Thinking Accordion ───────────────────

def render_o1_thinking_block(trace_data: list[dict]):
    """Render OpenAI o1-style expandable 'Thought for Xs' reasoning block."""
    if not trace_data:
        return

    # Calculate total thinking latency
    total_thought_time = sum(step.get("latency_sec", 0) for step in trace_data if step.get("step") != "generate_answer")
    classify_step = next((s for s in trace_data if s.get("step") == "classify_query"), None)
    retrieval_step = next((s for s in trace_data if "retrieve" in s.get("step", "")), None)

    route_chosen = classify_step.get("route_chosen", "vector") if classify_step else "vector"
    route_class = "chatgpt-route-vector" if route_chosen == "vector" else "chatgpt-route-graph"
    route_label = "📄 Vector DB (ChromaDB)" if route_chosen == "vector" else "🔗 Knowledge Graph (2-Hop BFS)"

    with st.expander(f"💭 Thought for {total_thought_time:.1f}s • {route_label}", expanded=False):
        for step in trace_data:
            s_name = step.get("step", "unknown").replace("_", " ").title()
            s_desc = step.get("description", "")
            s_lat = step.get("latency_sec", 0)

            st.markdown(f"""
            <div class="o1-step-row">
                <span style="color:var(--chatgpt-text-muted);">↳</span>
                <div>
                    <strong style="color:var(--chatgpt-text-primary);">{s_name}</strong> 
                    <span style="color:var(--chatgpt-text-muted);font-size:0.75rem;">({s_lat}s)</span><br>
                    <span style="color:var(--chatgpt-text-secondary);">{s_desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Details for vector sources
            if step.get("step") == "retrieve_vector" and step.get("sources"):
                for s in step["sources"][:3]:
                    st.caption(f"📌 *{s.get('source', 'Doc')}* (Page {s.get('page', '?')}) — Score: `{s.get('score', '?')}`")

            # Details for graph triples
            if step.get("step") == "retrieve_graph" and step.get("triples"):
                for t in step["triples"][:4]:
                    st.caption(f"🕸️ `{t.get('source')}` ➔ *{t.get('relation')}* ➔ `{t.get('target')}`")


for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🧠"):
        if msg["role"] == "assistant":
            # If this message has an associated reasoning trace, show o1 thinking block first
            if i in st.session_state.traces:
                render_o1_thinking_block(st.session_state.traces[i])
        st.markdown(msg["content"])


# ─── Query Execution Logic ───────────────────────────────────────────────────

user_input = st.chat_input("Message Local GraphRAG...")

# Check if triggered by starter buttons
active_query = None
if st.session_state.pending_query:
    active_query = st.session_state.pending_query
    st.session_state.pending_query = None
elif user_input:
    active_query = user_input

if active_query:
    # 1. Append & display user message
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(active_query)

    # 2. Assistant response generation
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Reasoning through knowledge graph and vector context..."):
            try:
                t_start = time.time()
                resp = requests.post(API_QUERY, json={"query": active_query}, timeout=300)
                elapsed = round(time.time() - t_start, 2)

                if resp.status_code == 200:
                    payload = resp.json()
                    answer_text = payload.get("response", "No response generated.")
                    route_decision = payload.get("route", "vector")
                    reasoning_trace = payload.get("reasoning_trace", [])

                    # Store trace keyed to message index
                    msg_idx = len(st.session_state.messages)
                    st.session_state.traces[msg_idx] = reasoning_trace

                    # Render o1 thinking block
                    render_o1_thinking_block(reasoning_trace)

                    # Render answer
                    st.markdown(answer_text)

                    # Route badge footer
                    badge_cls = "chatgpt-route-vector" if route_decision == "vector" else "chatgpt-route-graph"
                    lbl = "📄 Vector Search" if route_decision == "vector" else "🔗 Graph Traversal"
                    st.markdown(
                        f'<div style="margin-top:10px;"><span class="chatgpt-route-tag {badge_cls}">{lbl}</span> '
                        f'<span style="font-size:0.75rem;color:var(--chatgpt-text-muted);margin-left:6px;">⏱️ {elapsed}s</span></div>',
                        unsafe_allow_html=True,
                    )

                    st.session_state.messages.append({"role": "assistant", "content": answer_text})
                    st.rerun()

                elif resp.status_code == 503:
                    st.error(f"⚠️ {resp.json().get('detail', 'Backend or LLM unavailable.')}")
                else:
                    st.error(f"❌ API Error ({resp.status_code}): {resp.text}")

            except requests.ConnectionError:
                st.error("❌ Cannot reach FastAPI server on localhost:8000. Please start the backend.")
            except requests.Timeout:
                st.error("⏰ Request timed out. Gemma 4 may still be initializing.")
            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")
