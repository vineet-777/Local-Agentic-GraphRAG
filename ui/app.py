"""
Local GraphRAG — Streamlit Frontend
Dynamic file upload + chat UI with reasoning trace sidebar.
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
API_AUTH_SIGNUP = f"{API_BASE}/api/auth/signup"
API_AUTH_SIGNIN = f"{API_BASE}/api/auth/signin"
API_AUTH_ME = f"{API_BASE}/api/auth/me"
API_AUTH_LOGOUT = f"{API_BASE}/api/auth/logout"

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Local GraphRAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Global ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ─────────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: rgba(255,255,255,0.85);
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
        font-weight: 300;
    }

    /* ── Status badges ──────────────────────────────── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .status-ok {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-err {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .status-warn {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    /* ── Route badge ────────────────────────────────── */
    .route-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        margin: 4px 0;
    }
    .route-vector {
        background: linear-gradient(135deg, rgba(59,130,246,0.15), rgba(37,99,235,0.15));
        color: #3b82f6;
        border: 1px solid rgba(59,130,246,0.3);
    }
    .route-graph {
        background: linear-gradient(135deg, rgba(168,85,247,0.15), rgba(139,92,246,0.15));
        color: #a855f7;
        border: 1px solid rgba(168,85,247,0.3);
    }

    /* ── Trace card ─────────────────────────────────── */
    .trace-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        backdrop-filter: blur(10px);
    }
    .trace-step-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #a78bfa;
        margin-bottom: 4px;
    }
    .trace-detail {
        font-size: 0.77rem;
        color: rgba(255,255,255,0.6);
        line-height: 1.5;
    }
    .trace-latency {
        font-size: 0.72rem;
        color: rgba(255,255,255,0.35);
        margin-top: 4px;
    }

    /* ── Error / info banner ────────────────────────── */
    .error-banner {
        background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(185,28,28,0.1));
        border: 1px solid rgba(239,68,68,0.3);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .error-banner h3 {
        color: #ef4444;
        font-size: 1rem;
        margin: 0 0 0.5rem 0;
    }
    .error-banner p {
        color: rgba(255,255,255,0.7);
        font-size: 0.88rem;
        margin: 0;
        line-height: 1.6;
    }

    .info-banner {
        background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(99,102,241,0.08));
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .info-banner h3 {
        color: #818cf8;
        font-size: 1rem;
        margin: 0 0 0.5rem 0;
    }
    .info-banner p {
        color: rgba(255,255,255,0.7);
        font-size: 0.88rem;
        margin: 0;
        line-height: 1.6;
    }

    /* ── Processing card ────────────────────────────── */
    .processing-card {
        background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(5,150,105,0.08));
        border: 1px solid rgba(16,185,129,0.25);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
    }
    .processing-card .stage {
        font-size: 0.85rem;
        font-weight: 600;
        color: #10b981;
        margin-bottom: 4px;
    }
    .processing-card .message {
        font-size: 0.8rem;
        color: rgba(255,255,255,0.6);
    }

    /* ── Upload area ────────────────────────────────── */
    .upload-section {
        background: rgba(255,255,255,0.02);
        border: 1px dashed rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* ── DB stats ────────────────────────────────────── */
    .db-stat {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: 8px;
        font-size: 0.72rem;
        font-weight: 500;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        color: rgba(255,255,255,0.5);
        margin: 2px;
    }

    /* ── Chat area ──────────────────────────────────── */
    .stChatMessage {
        border-radius: 12px !important;
    }

    /* ── Sidebar ────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
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
if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "auth_error" not in st.session_state:
    st.session_state.auth_error = None


# ─── API Helpers ──────────────────────────────────────────────────────────────

def auth_headers():
    """Return bearer credentials for endpoints that require a signed-in user."""
    token = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def clear_session():
    """Clear local identity and user-specific UI state."""
    st.session_state.access_token = None
    st.session_state.current_user = None
    st.session_state.messages = []
    st.session_state.traces = {}
    st.session_state.upload_triggered = False


def expire_session():
    """Return to the sign-in screen when the backend rejects an expired token."""
    clear_session()
    st.session_state.auth_error = "Your session expired. Please sign in again."
    st.rerun()


def save_auth_response(response):
    """Store a successful signup/signin response in Streamlit session state."""
    data = response.json()
    st.session_state.access_token = data["access_token"]
    st.session_state.current_user = data["user"]
    st.session_state.auth_error = None
    st.rerun()


def response_detail(response):
    """Get a readable API error message, including FastAPI validation errors."""
    try:
        detail = response.json().get("detail", response.text)
        if isinstance(detail, list):
            return "; ".join(item.get("msg", str(item)) for item in detail)
        return str(detail)
    except (ValueError, AttributeError):
        return response.text or f"Request failed ({response.status_code})."


def render_auth_page():
    """Render signup and signin forms before any GraphRAG features are shown."""
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Local GraphRAG</h1>
        <p>Sign in to your local knowledge workspace</p>
    </div>
    """, unsafe_allow_html=True)
    _, auth_column, _ = st.columns([1, 1.2, 1])
    with auth_column:
        st.markdown("### Welcome")
        st.caption("Your account and session are stored locally by this GraphRAG server.")
        if st.session_state.auth_error:
            st.warning(st.session_state.auth_error)

        signin_tab, signup_tab = st.tabs(["Sign in", "Create account"])
        with signin_tab:
            with st.form("signin_form"):
                signin_email = st.text_input("Email", key="signin_email", autocomplete="email")
                signin_password = st.text_input(
                    "Password", type="password", key="signin_password", autocomplete="current-password"
                )
                signin_submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
            if signin_submitted:
                if not signin_email.strip() or not signin_password:
                    st.error("Enter your email and password.")
                else:
                    try:
                        response = requests.post(
                            API_AUTH_SIGNIN,
                            json={"email": signin_email.strip(), "password": signin_password},
                            timeout=15,
                        )
                        if response.status_code == 200:
                            save_auth_response(response)
                        else:
                            st.error(response_detail(response))
                    except requests.ConnectionError:
                        st.error("Cannot connect to the API server. Start the backend and try again.")
                    except requests.Timeout:
                        st.error("The API did not respond in time. Please try again.")

        with signup_tab:
            with st.form("signup_form"):
                signup_name = st.text_input("Name", max_chars=80, autocomplete="name")
                signup_email = st.text_input("Email", autocomplete="email")
                signup_password = st.text_input(
                    "Password", type="password", help="Use at least 8 characters.", autocomplete="new-password"
                )
                confirm_password = st.text_input(
                    "Confirm password", type="password", autocomplete="new-password"
                )
                signup_submitted = st.form_submit_button(
                    "Create account", type="primary", use_container_width=True
                )
            if signup_submitted:
                if not signup_name.strip() or not signup_email.strip() or not signup_password:
                    st.error("Complete all fields to create an account.")
                elif len(signup_password) < 8:
                    st.error("Password must be at least 8 characters.")
                elif signup_password != confirm_password:
                    st.error("The passwords do not match.")
                else:
                    try:
                        response = requests.post(
                            API_AUTH_SIGNUP,
                            json={
                                "name": signup_name.strip(),
                                "email": signup_email.strip(),
                                "password": signup_password,
                            },
                            timeout=15,
                        )
                        if response.status_code == 201:
                            save_auth_response(response)
                        else:
                            st.error(response_detail(response))
                    except requests.ConnectionError:
                        st.error("Cannot connect to the API server. Start the backend and try again.")
                    except requests.Timeout:
                        st.error("The API did not respond in time. Please try again.")


if st.session_state.access_token:
    try:
        user_response = requests.get(API_AUTH_ME, headers=auth_headers(), timeout=10)
        if user_response.status_code == 200:
            st.session_state.current_user = user_response.json()
            st.session_state.auth_error = None
        else:
            clear_session()
            st.session_state.auth_error = "Your session expired. Please sign in again."
    except requests.RequestException:
        # Keep the token during a temporary backend outage; the API error is shown below.
        pass

if not st.session_state.access_token:
    render_auth_page()
    st.stop()

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
        resp = requests.get(API_UPLOAD_STATUS, headers=auth_headers(), timeout=5)
        if resp.status_code == 401:
            expire_session()
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        pass
    return None


def upload_files_to_api(files):
    """Send files to the /upload endpoint."""
    try:
        file_tuples = []
        for f in files:
            file_tuples.append(("files", (f.name, f.getvalue(), "application/octet-stream")))

        resp = requests.post(API_UPLOAD, files=file_tuples, headers=auth_headers(), timeout=30)
        if resp.status_code == 401:
            expire_session()
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 409:
            return {"error": "A processing job is already running. Please wait."}
        elif resp.status_code == 503:
            return {"error": resp.json().get("detail", "Service unavailable")}
        else:
            return {"error": f"Upload failed ({resp.status_code}): {response_detail(resp)}"}
    except requests.ConnectionError:
        return {"error": "Cannot connect to API server."}
    except requests.Timeout:
        return {"error": "Upload request timed out. Please try again."}
    except requests.RequestException as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    signed_in_user = st.session_state.get("current_user") or {}
    st.markdown(f"**Signed in as {signed_in_user.get('name', 'User')}**")
    st.caption(signed_in_user.get("email", ""))
    if st.button("Sign out", use_container_width=True):
        try:
            requests.post(API_AUTH_LOGOUT, headers=auth_headers(), timeout=10)
        except requests.RequestException:
            pass
        clear_session()
        st.rerun()

    # ─── Section 1: File Upload ───────────────────────────────────────────
    st.markdown("### 📁 Upload Documents")

    uploaded_files = st.file_uploader(
        "Drop your PDFs / DOCX files",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
        key="file_uploader",
        help="Upload course materials. They'll be processed locally into the knowledge base.",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            size_kb = len(f.getvalue()) / 1024
            st.caption(f"📄 {f.name} ({size_kb:.0f} KB)")

        if st.button("🚀 Process Files", use_container_width=True, type="primary"):
            result = upload_files_to_api(uploaded_files)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(result.get("message", "Files queued!"))
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
        st.markdown("### ⚙️ Processing")

        progress_val = proc_status.get("progress", 0.0)
        stage = proc_status.get("stage", "")
        message = proc_status.get("message", "")

        # Stage icons
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

        # Auto-refresh while processing
        time.sleep(2)
        st.rerun()

    elif proc_status and proc_status.get("stage") == "done" and st.session_state.upload_triggered:
        st.markdown("---")
        st.success("✅ All files processed!")
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
        lm_ok = health.get("lm_studio", {}).get("running", False)
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

        # DB details
        if db.get("chromadb_exists") or db.get("graph_json_exists"):
            stats_html = ""
            if db.get("graph_nodes", 0) > 0:
                stats_html += f'<span class="db-stat">🔗 {db["graph_nodes"]} nodes</span> '
                stats_html += f'<span class="db-stat">↔️ {db["graph_edges"]} edges</span>'
            if db.get("chromadb_exists"):
                stats_html += f' <span class="db-stat">📄 Vector DB active</span>'
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
                label = "📄 Vector DB" if route == "vector" else "🔗 Knowledge Graph"
                st.markdown(f'<span class="route-badge {badge_class}">{label}</span>', unsafe_allow_html=True)

            elif step_name == "retrieve_vector":
                sources = step.get("sources", [])
                if sources:
                    with st.expander(f"📋 {len(sources)} chunks retrieved", expanded=False):
                        for s in sources:
                            st.markdown(f"**{s.get('source', 'Unknown')}** (p.{s.get('page', '?')}) — score: {s.get('score', '?')}")
                            st.caption(s.get("preview", ""))

            elif step_name == "retrieve_graph":
                triples = step.get("triples", [])
                if triples:
                    with st.expander(f"🔗 {len(triples)} triples found", expanded=False):
                        for t in triples:
                            st.markdown(f"**{t.get('source','')}** →  *{t.get('relation','')}*  → **{t.get('target','')}**")

    else:
        st.caption("Ask a question to see the reasoning trace here.")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.traces = {}
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Local GraphRAG</h1>
    <p>Upload your documents and start asking questions — 100% local, powered by GraphRAG + LM Studio</p>
</div>
""", unsafe_allow_html=True)


# ─── System Checks ───────────────────────────────────────────────────────────

def show_system_status():
    """Display actionable banners based on system state."""
    if st.session_state.health_data is None:
        st.markdown("""
        <div class="error-banner">
            <h3>⚠️ Cannot Connect to API Server</h3>
            <p>
                The FastAPI backend is not running at <code>localhost:8000</code>.<br>
                Start it with: <code>python -m uvicorn api.server:app --host 127.0.0.1 --port 8000</code>
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
            <h3>⚠️ LM Studio Not Running</h3>
            <p>
                Cannot connect to LM Studio at <code>localhost:1234</code>.<br>
                Please start LM Studio and load the required models.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return "no_lm_studio"

    # Missing models warning
    missing_models = []
    if not lm_studio.get("llm_ready"):
        missing_models.append("gemma-4-e4b-it")
    if not lm_studio.get("embed_ready"):
        missing_models.append("nomic-embed-text-v1.5")
    if not lm_studio.get("router_ready"):
        missing_models.append("gemma-4-e4b-it (router)")
    if missing_models:
        model_list = "<br>".join([f"<code>{m}</code>" for m in missing_models])
        st.markdown(f"""
        <div class="error-banner">
            <h3>⚠️ Required Models Not Loaded</h3>
            <p>Please load the following models in LM Studio:<br>{model_list}</p>
        </div>
        """, unsafe_allow_html=True)
        return "no_models"

    # No data yet — prompt to upload
    if not db.get("ready"):
        st.markdown("""
        <div class="info-banner">
            <h3>📂 No Documents Yet</h3>
            <p>
                Your knowledge base is empty. Use the <strong>sidebar</strong> to upload
                PDF or DOCX files — they'll be processed locally into embeddings and a knowledge graph.
            </p>
        </div>
        """, unsafe_allow_html=True)
        return "no_data"

    return "ready"


system_state = show_system_status()
can_chat = system_state in ("ready", "no_data")  # Allow chat if LM Studio is up, even if no data yet


# ─── Chat Interface ──────────────────────────────────────────────────────────

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "🧠"):
        st.markdown(msg["content"])

if prompt := st.chat_input(
    "Ask about your documents..." if system_state == "ready" else "Upload documents first, then ask questions...",
    disabled=not can_chat or system_state == "no_data",
):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    # Query the engine
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Thinking..."):
            try:
                t0 = time.time()
                resp = requests.post(
                    API_QUERY,
                    json={"query": prompt},
                    headers=auth_headers(),
                    timeout=300,  # Gemma 4 on 4GB VRAM can be slow
                )
                total_time = round(time.time() - t0, 2)

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("response", "No response received.")
                    route = data.get("route", "unknown")
                    trace = data.get("reasoning_trace", [])

                    st.markdown(answer)

                    route_label = "📄 Vector DB" if route == "vector" else "🔗 Knowledge Graph"
                    route_class = "route-vector" if route == "vector" else "route-graph"
                    st.markdown(
                        f'<span class="route-badge {route_class}">{route_label}</span> '
                        f'<span style="font-size:0.75rem;color:rgba(255,255,255,0.4)">({total_time}s)</span>',
                        unsafe_allow_html=True,
                    )

                    msg_idx = len(st.session_state.messages)
                    st.session_state.traces[msg_idx] = trace

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    st.rerun()

                elif resp.status_code == 401:
                    expire_session()
                elif resp.status_code == 503:
                    error_detail = response_detail(resp)
                    st.error(f"⚠️ {error_detail}")
                else:
                    st.error(f"❌ API error ({resp.status_code}): {response_detail(resp)}")

            except requests.ConnectionError:
                st.error("❌ Cannot connect to the API server. Is it running?")
            except requests.Timeout:
                st.error("⏰ Request timed out. The LLM may still be loading.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {str(e)}")
