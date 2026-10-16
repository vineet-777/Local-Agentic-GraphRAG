"""
Local GraphRAG — Streamlit Frontend v2.0
Modern Dual-Retrieval UI with EmbeddingGemma-2 Support,
Real-Time Progress Tracking, and Transparent Reasoning Traces.
"""

import streamlit as st
import requests
import time

# ─── Configuration ────────────────────────────────────────────────────────────

API_BASE = "http://127.0.0.1:8000"
API_QUERY = f"{API_BASE}/api/query"
API_HEALTH = f"{API_BASE}/api/health"
API_UPLOAD = f"{API_BASE}/api/upload"
API_UPLOAD_STATUS = f"{API_BASE}/api/upload-status"

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Local GraphRAG v2.0",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (Modern Glassmorphism & Cyber Theme) ─────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Main Hero Header ───────────────────────────── */
    .hero-header {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 20px;
        padding: 1.6rem 2.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px -10px rgba(124, 58, 237, 0.3);
        backdrop-filter: blur(12px);
    }
    .hero-title-row {
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }
    .hero-title {
        color: #f8fafc;
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.03em;
    }
    .hero-badge {
        background: linear-gradient(135deg, #8b5cf6, #3b82f6);
        color: white;
        font-size: 0.72rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 9999px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
    }
    .hero-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 0.5rem;
    }
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 3px 10px;
        border-radius: 8px;
        font-size: 0.74rem;
        color: #cbd5e1;
        font-weight: 500;
    }
    .pill-accent {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
    }

    /* ── Quick Starter Cards ────────────────────────── */
    .quick-start-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 12px;
        margin: 1.2rem 0;
    }
    .quick-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 14px 16px;
        transition: all 0.2s ease;
    }
    .quick-card:hover {
        border-color: rgba(139, 92, 246, 0.4);
        background: rgba(30, 41, 59, 0.7);
        transform: translateY(-2px);
    }
    .quick-card-title {
        color: #e2e8f0;
        font-size: 0.84rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .quick-card-desc {
        color: #64748b;
        font-size: 0.75rem;
        line-height: 1.4;
    }

    /* ── Status Badges ──────────────────────────────── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-ok {
        background: rgba(16, 185, 129, 0.12);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-err {
        background: rgba(239, 68, 68, 0.12);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .status-warn {
        background: rgba(245, 158, 11, 0.12);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* ── Route Badges ───────────────────────────────── */
    .route-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 4px 0;
    }
    .route-vector {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(37, 99, 235, 0.25));
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.35);
    }
    .route-graph {
        background: linear-gradient(135deg, rgba(168, 85, 247, 0.15), rgba(139, 92, 246, 0.25));
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.35);
    }

    /* ── Trace Card ─────────────────────────────────── */
    .trace-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 12px 14px;
        margin: 8px 0;
        backdrop-filter: blur(8px);
    }
    .trace-step-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #a78bfa;
        margin-bottom: 4px;
    }
    .trace-detail {
        font-size: 0.77rem;
        color: #94a3b8;
        line-height: 1.5;
    }
    .trace-latency {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 5px;
    }

    /* ── Processing Card ────────────────────────────── */
    .processing-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08), rgba(5, 150, 105, 0.12));
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .processing-card .stage {
        font-size: 0.85rem;
        font-weight: 700;
        color: #34d399;
        margin-bottom: 4px;
    }
    .processing-card .message {
        font-size: 0.8rem;
        color: #cbd5e1;
    }

    /* ── Database Stat Chips ────────────────────────── */
    .db-stat {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.72rem;
        font-weight: 600;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #cbd5e1;
        margin: 2px;
    }

    /* ── Banners ────────────────────────────────────── */
    .error-banner {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(185, 28, 28, 0.1));
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .error-banner h3 {
        color: #f87171;
        font-size: 0.98rem;
        margin: 0 0 0.4rem 0;
    }
    .error-banner p {
        color: #cbd5e1;
        font-size: 0.86rem;
        margin: 0;
        line-height: 1.5;
    }

    .info-banner {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.08), rgba(99, 102, 241, 0.08));
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .info-banner h3 {
        color: #818cf8;
        font-size: 0.98rem;
        margin: 0 0 0.4rem 0;
    }
    .info-banner p {
        color: #cbd5e1;
        font-size: 0.86rem;
        margin: 0;
        line-height: 1.5;
    }

    /* ── Sidebar Polish ─────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #0d1117 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ──────────────────────────────────────────────────────

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


# ─── API Helpers ──────────────────────────────────────────────────────────────

def check_health():
    """Ping the FastAPI health endpoint."""
    try:
        resp = requests.get(API_HEALTH, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.health_data = data
            st.session_state.system_healthy = data.get("lm_studio", {}).get("running", False)
            return data
        else:
            st.session_state.system_healthy = False
            return None
    except requests.ConnectionError:
        st.session_state.system_healthy = False
        st.session_state.health_data = None
        return None


def get_processing_status():
    """Poll background file processing status."""
    try:
        resp = requests.get(API_UPLOAD_STATUS, timeout=5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def upload_files_to_api(files):
    """Send files to the /upload endpoint."""
    try:
        file_tuples = []
        for f in files:
            file_tuples.append(("files", (f.name, f.getvalue(), "application/octet-stream")))

        resp = requests.post(API_UPLOAD, files=file_tuples, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 409:
            return {"error": "A processing job is already running. Please wait."}
        elif resp.status_code == 503:
            return {"error": resp.json().get("detail", "Service unavailable")}
        else:
            return {"error": f"Upload failed ({resp.status_code}): {resp.text}"}
    except requests.ConnectionError:
        return {"error": "Cannot connect to API server."}
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ─── Section 1: File Upload ───────────────────────────────────────────
    st.markdown("### 📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Drop PDFs or DOCX files here",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="file_uploader",
        help="Upload course materials. Processed 100% locally into ChromaDB and Knowledge Graph.",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            size_kb = len(f.getvalue()) / 1024
            st.caption(f"📄 {f.name} ({size_kb:.0f} KB)")

        # Removed deprecated use_container_width=True
        if st.button("🚀 Process Files", type="primary"):
            result = upload_files_to_api(uploaded_files)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(result.get("message", "Files queued for processing!"))
                if result.get("files_rejected"):
                    for rej in result["files_rejected"]:
                        st.warning(f"⚠️ {rej['file']}: {rej['reason']}")
                st.session_state.upload_triggered = True
                time.sleep(0.5)
                st.rerun()

    # ─── Section 2: Processing Status ─────────────────────────────────────
    proc_status = get_processing_status()

    if proc_status and proc_status.get("is_processing"):
        st.markdown("---")
        st.markdown("### ⚙️ Live Ingestion")

        progress_val = proc_status.get("progress", 0.0)
        stage = proc_status.get("stage", "")
        message = proc_status.get("message", "")

        stage_icons = {
            "starting": "🔄",
            "extracting": "📄",
            "chunking": "✂️",
            "embedding": "🧮",
            "graphing": "🔗",
        }
        stage_icon = stage_icons.get(stage, "⏳")

        st.markdown(f"""
        <div class="processing-card">
            <div class="stage">{stage_icon} {stage.upper()}</div>
            <div class="message">{message}</div>
        </div>
        """, unsafe_allow_html=True)

        st.progress(progress_val, text=f"{int(progress_val * 100)}%")

        files_done = proc_status.get("processed_files", 0)
        files_total = proc_status.get("total_files", 0)
        st.caption(f"Files: {files_done}/{files_total}")

        time.sleep(2)
        st.rerun()

    elif proc_status and proc_status.get("stage") == "done" and st.session_state.upload_triggered:
        st.markdown("---")
        st.success("✅ All files processed successfully!")
        st.session_state.upload_triggered = False
        time.sleep(1)
        st.rerun()

    elif proc_status and proc_status.get("stage") == "error":
        st.markdown("---")
        st.error(f"❌ {proc_status.get('message', 'Processing failed')}")
        st.session_state.upload_triggered = False

    # ─── Section 3: System Status ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 System Status")

    health = check_health()

    if health:
        lm_info = health.get("lm_studio", {})
        lm_ok = lm_info.get("running", False)
        db = health.get("database", {})
        db_ok = db.get("ready", False)

        col1, col2 = st.columns(2)
        with col1:
            if lm_ok:
                st.markdown('<span class="status-badge status-ok">● LM Studio</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge status-err">● LM Studio</span>', unsafe_allow_html=True)
        with col2:
            if db_ok:
                st.markdown('<span class="status-badge status-ok">● Database</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge status-warn">● No Data</span>', unsafe_allow_html=True)

        # Active Model Badges
        active_embed = lm_info.get("configured_embed_model", "embeddinggemma-2")
        short_embed = "EmbeddingGemma-2" if "embeddinggemma" in active_embed.lower() else "Nomic Embed"
        embed_dim = lm_info.get("embedding_dim", 768)

        st.markdown(f"""
        <div style="margin-top:8px;">
            <span class="db-stat">🧬 {short_embed} ({embed_dim}d)</span>
            <span class="db-stat">⚡ Gemma 4 E4B</span>
        </div>
        """, unsafe_allow_html=True)

        # DB Details
        if db.get("chromadb_exists") or db.get("graph_json_exists"):
            stats_html = "<div style='margin-top:4px;'>"
            if db.get("graph_nodes", 0) > 0:
                stats_html += f'<span class="db-stat">🔗 {db["graph_nodes"]} nodes</span> '
                stats_html += f'<span class="db-stat">↔️ {db["graph_edges"]} edges</span> '
            if db.get("chromadb_exists"):
                stats_html += f'<span class="db-stat">📄 ChromaDB Active</span>'
            stats_html += "</div>"
            st.markdown(stats_html, unsafe_allow_html=True)

    else:
        st.markdown('<span class="status-badge status-err">● API Offline</span>', unsafe_allow_html=True)

    # ─── Section 4: Reasoning Trace ───────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🧠 Reasoning Trace")

    if st.session_state.traces:
        last_idx = max(st.session_state.traces.keys())
        trace_data = st.session_state.traces[last_idx]

        for step in trace_data:
            step_name = step.get("step", "unknown")
            description = step.get("description", "")
            latency = step.get("latency_sec", 0)

            icons = {
                "classify_query": "🏷️",
                "retrieve_vector": "📄",
                "retrieve_graph": "🔗",
                "generate_answer": "✨",
            }
            icon = icons.get(step_name, "▪️")

            st.markdown(f"""
            <div class="trace-card">
                <div class="trace-step-title">{icon} {step_name.replace('_', ' ').title()}</div>
                <div class="trace-detail">{description}</div>
                <div class="trace-latency">⏱️ {latency}s</div>
            </div>
            """, unsafe_allow_html=True)

            if step_name == "classify_query":
                route = step.get("route_chosen", "")
                badge_class = "route-vector" if route == "vector" else "route-graph"
                label = "📄 Vector DB (ChromaDB)" if route == "vector" else "🔗 Knowledge Graph (2-Hop BFS)"
                st.markdown(f'<span class="route-badge {badge_class}">{label}</span>', unsafe_allow_html=True)

            elif step_name == "retrieve_vector":
                sources = step.get("sources", [])
                if sources:
                    with st.expander(f"📋 {len(sources)} Chunks Retrieved", expanded=False):
                        for s in sources:
                            st.markdown(f"**{s.get('source', 'Unknown')}** (p.{s.get('page', '?')}) — *Score: {s.get('score', '?')}*")
                            st.caption(s.get("preview", ""))

            elif step_name == "retrieve_graph":
                triples = step.get("triples", [])
                if triples:
                    with st.expander(f"🔗 {len(triples)} Triples Traversed", expanded=False):
                        for t in triples:
                            st.markdown(f"**{t.get('source','')}**  `—({t.get('relation','')})→`  **{t.get('target','')}**")

    else:
        st.caption("Ask a question to inspect the live multi-agent reasoning trace here.")

    st.markdown("---")
    # Removed deprecated use_container_width=True
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.traces = {}
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════

# Hero Header with Glassmorphism
st.markdown("""
<div class="hero-header">
    <div class="hero-title-row">
        <h1 class="hero-title">🧠 Local GraphRAG</h1>
        <span class="hero-badge">v2.0 • EmbeddingGemma</span>
    </div>
    <div class="hero-subtitle">
        Fully-Local Multi-Agent Question-Answering System Using Graph-Augmented & Semantic Retrieval
    </div>
    <div class="hero-pills">
        <span class="pill pill-accent">● 100% On-Device</span>
        <span class="pill">⚡ Gemma 4 E4B</span>
        <span class="pill">🧬 Google EmbeddingGemma-2</span>
        <span class="pill">📄 ChromaDB</span>
        <span class="pill">🕸️ 2-Hop Graph BFS</span>
        <span class="pill">🔒 Zero Cloud API</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── System Checks ───────────────────────────────────────────────────────────

def show_system_status():
    """Display actionable status banners."""
    if st.session_state.health_data is None:
        st.markdown("""
        <div class="error-banner">
            <h3>⚠️ FastAPI Backend Server Offline</h3>
            <p>
                Cannot connect to backend at <code>localhost:8000</code>.<br>
                Start it via terminal: <code>python -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload</code>
            </p>
        </div>
        """, unsafe_allow_html=True)
        return "no_api"

    health = st.session_state.health_data
    lm_studio = health.get("lm_studio", {})
    db = health.get("database", {})

    if not lm_studio.get("running"):
        st.markdown("""
        <div class="error-banner">
            <h3>⚠️ LM Studio Local Server Not Running</h3>
            <p>
                Cannot connect to LM Studio at <code>http://localhost:1234/v1</code>.<br>
                Please start LM Studio, load your models, and turn ON the Local Server.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return "no_lm_studio"

    # Missing models warning
    missing = []
    if not lm_studio.get("llm_ready"):
        missing.append("google/gemma-4-e4b (Router & Synthesizer)")
    if not lm_studio.get("embed_ready"):
        missing.append("google/embeddinggemma-2 or nomic-embed-text-v1.5")
    if missing:
        m_list = "<br>".join([f"• <code>{m}</code>" for m in missing])
        st.markdown(f"""
        <div class="error-banner">
            <h3>⚠️ Required Models Not Loaded in LM Studio</h3>
            <p>Please load the following models in LM Studio:<br>{m_list}</p>
        </div>
        """, unsafe_allow_html=True)
        return "no_models"

    if not db.get("ready"):
        st.markdown("""
        <div class="info-banner">
            <h3>📂 Knowledge Base Ready for Documents</h3>
            <p>
                Your knowledge base is currently empty. Upload your <strong>PDF or DOCX</strong> files 
                in the sidebar to automatically generate dense vector embeddings and relational graph connections.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return "no_data"

    return "ready"


system_state = show_system_status()
can_chat = system_state in ("ready", "no_data")


# ─── Chat Message Loop ───────────────────────────────────────────────────────

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🧠"):
        st.markdown(msg["content"])

# Prompt Input
if prompt := st.chat_input(
    "Ask a question about your documents..." if system_state == "ready" else "Upload documents in sidebar to begin...",
    disabled=not can_chat or system_state == "no_data",
):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    # Query the engine
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Analyzing query and orchestrating retrieval..."):
            try:
                t0 = time.time()
                resp = requests.post(
                    API_QUERY,
                    json={"query": prompt},
                    timeout=300,
                )
                total_time = round(time.time() - t0, 2)

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("response", "No response received.")
                    route = data.get("route", "unknown")
                    trace = data.get("reasoning_trace", [])

                    st.markdown(answer)

                    route_label = "📄 Vector DB (ChromaDB)" if route == "vector" else "🔗 Knowledge Graph"
                    route_class = "route-vector" if route == "vector" else "route-graph"
                    st.markdown(
                        f'<span class="route-badge {route_class}">{route_label}</span> '
                        f'<span style="font-size:0.75rem;color:#64748b;margin-left:6px;">⏱️ Total Latency: {total_time}s</span>',
                        unsafe_allow_html=True,
                    )

                    msg_idx = len(st.session_state.messages)
                    st.session_state.traces[msg_idx] = trace

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()

                elif resp.status_code == 503:
                    error_detail = resp.json().get("detail", "Service unavailable")
                    st.error(f"⚠️ {error_detail}")
                else:
                    st.error(f"❌ API error ({resp.status_code}): {resp.text}")

            except requests.ConnectionError:
                st.error("❌ Cannot connect to FastAPI server. Please ensure `api.server:app` is running on port 8000.")
            except requests.Timeout:
                st.error("⏰ Request timed out. The local LLM took too long to respond.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
