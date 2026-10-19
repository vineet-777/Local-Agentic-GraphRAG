"""
Local GraphRAG — Production UI/UX Overhaul
Air-Gapped Engineering Studio Architecture:
- Multi-Workspace Shell (Query Studio, Graph Explorer, Knowledge Base, Engine Diagnostics)
- Interactive Force-Directed Knowledge Graph Explorer (60fps HTML5 Canvas, zero external deps)
- Real-time Dual-Stream Reasoning Telemetry & Source Provenance Drawer
- Live Pipeline Status & ChromaDB Chunk Catalog
- Air-Gapped Authentication & Session Management
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

import streamlit as st
import streamlit.components.v1 as components
import requests

# Ensure project root is accessible
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

GRAPH_FILE_PATH = PROJECT_ROOT / "db" / "graph.json"
CHROMA_DIR_PATH = PROJECT_ROOT / "db" / "chromadb"

# ─── Page Configuration ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Local-Agentic-GraphRAG Studio",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design System: Air-Gapped Engineering Studio ─────────────────────────────

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
    /* ─── CSS Variables & Design Tokens ───────────────────────────────── */
    :root {
        --bg-base: #0b0e14;
        --bg-surface: #121721;
        --bg-elevated: #1a2232;
        --bg-input: #151c29;
        
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        
        --border-subtle: rgba(255, 255, 255, 0.07);
        --border-medium: rgba(255, 255, 255, 0.14);
        --border-focus: #38bdf8;
        
        --accent-vector: #38bdf8;
        --accent-vector-bg: rgba(56, 189, 248, 0.08);
        --accent-vector-border: rgba(56, 189, 248, 0.25);
        
        --accent-graph: #10b981;
        --accent-graph-bg: rgba(16, 185, 129, 0.08);
        --accent-graph-border: rgba(16, 185, 129, 0.25);
        
        --accent-router: #f59e0b;
        --accent-router-bg: rgba(245, 158, 11, 0.08);
        --accent-router-border: rgba(245, 158, 11, 0.25);
        
        --accent-alert: #f43f5e;
        --accent-alert-bg: rgba(244, 63, 94, 0.08);
        --accent-alert-border: rgba(244, 63, 94, 0.25);
        
        --font-heading: 'Plus Jakarta Sans', system-ui, sans-serif;
        --font-body: 'Inter', system-ui, sans-serif;
        --font-mono: 'JetBrains Mono', Consolas, monospace;
        
        --radius-sm: 4px;
        --radius-md: 6px;
        --radius-lg: 10px;
    }

    /* ─── Global Base Overrides ────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: var(--font-body);
        color: var(--text-primary);
        background: var(--bg-base);
    }
    
    .stApp {
        background-color: var(--bg-base);
    }

    /* Hide default white Streamlit header */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    [data-testid="stSidebarHeader"] {
        display: none !important;
    }

    /* Streamlit block container width and spacing */
    .block-container {
        max-width: 1400px;
        padding-top: 1rem !important;
        padding-bottom: 4rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Dark styled file uploaders */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--bg-elevated) !important;
        border: 1px dashed var(--border-medium) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-secondary) !important;
    }
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small {
        color: var(--text-muted) !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--bg-surface) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-medium) !important;
    }

    /* Headings */
    h1, h2, h3, h4, h5 {
        font-family: var(--font-heading);
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
    }
    
    code, pre, .mono-data {
        font-family: var(--font-mono) !important;
    }

    /* ─── Top Application Shell Bar ────────────────────────────────────── */
    .app-shell-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.75rem 1.25rem;
        background: var(--bg-surface);
        border: 1px solid var(--border-medium);
        border-radius: var(--radius-md);
        margin-bottom: 1.25rem;
    }

    .brand-cluster {
        display: flex;
        align-items: center;
        gap: 0.85rem;
    }

    .brand-mark {
        width: 32px;
        height: 32px;
        background: var(--accent-vector-bg);
        border: 1px solid var(--accent-vector-border);
        border-radius: var(--radius-sm);
        display: grid;
        place-items: center;
        font-size: 1.1rem;
        color: var(--accent-vector);
        font-weight: 800;
    }

    .brand-title {
        font-family: var(--font-heading);
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        line-height: 1.2;
    }

    .brand-subtitle {
        font-family: var(--font-mono);
        font-size: 0.68rem;
        color: var(--text-muted);
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .telemetry-strip {
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .telemetry-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.6rem;
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        font-family: var(--font-mono);
        font-size: 0.72rem;
        color: var(--text-secondary);
    }

    .telemetry-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: var(--text-muted);
    }
    .telemetry-dot.online { background: var(--accent-graph); box-shadow: 0 0 6px rgba(16, 185, 129, 0.4); }
    .telemetry-dot.warning { background: var(--accent-router); }
    .telemetry-dot.offline { background: var(--accent-alert); }

    /* ─── Navigation Tabs Overrides ────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
        border-bottom: 1px solid var(--border-medium);
        padding-bottom: 0.25rem;
        margin-bottom: 1.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 2.6rem;
        padding: 0 1.2rem;
        background: transparent;
        border-radius: var(--radius-sm) var(--radius-sm) 0 0;
        font-family: var(--font-heading);
        font-weight: 600;
        font-size: 0.86rem;
        color: var(--text-secondary);
        border: none;
        transition: all 0.15s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background: rgba(255, 255, 255, 0.02);
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--text-primary) !important;
        background: var(--bg-surface) !important;
        border-bottom: 2px solid var(--accent-vector) !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--accent-vector) !important;
    }

    /* ─── Sidebar Styling ──────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--bg-surface);
        border-right: 1px solid var(--border-medium);
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.25rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .sidebar-user-card {
        padding: 0.85rem;
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        margin-bottom: 1.25rem;
    }

    .user-name {
        font-family: var(--font-heading);
        font-weight: 600;
        font-size: 0.88rem;
        color: var(--text-primary);
    }

    .user-email {
        font-family: var(--font-mono);
        font-size: 0.72rem;
        color: var(--text-muted);
        overflow-wrap: anywhere;
        margin-top: 0.15rem;
    }

    .sidebar-section-header {
        font-family: var(--font-mono);
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: var(--text-muted);
        margin: 1.2rem 0 0.5rem 0;
        font-weight: 600;
    }

    /* ─── Query Studio: Provenance & Route Badges ──────────────────────── */
    .provenance-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.45rem 0.85rem;
        border-radius: var(--radius-sm);
        margin: 0.5rem 0 0.85rem 0;
        font-family: var(--font-mono);
        font-size: 0.75rem;
    }

    .provenance-bar.vector {
        background: var(--accent-vector-bg);
        border: 1px solid var(--accent-vector-border);
        color: var(--accent-vector);
    }

    .provenance-bar.graph {
        background: var(--accent-graph-bg);
        border: 1px solid var(--accent-graph-border);
        color: var(--accent-graph);
    }

    .route-capsule {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-weight: 600;
    }

    .latency-tag {
        color: var(--text-muted);
        font-size: 0.7rem;
    }

    /* Trace timeline cards */
    .trace-timeline-step {
        border-left: 2px solid var(--border-medium);
        padding-left: 0.85rem;
        margin-bottom: 0.75rem;
    }
    .trace-timeline-step.active {
        border-left-color: var(--accent-vector);
    }
    .trace-step-name {
        font-family: var(--font-heading);
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    .trace-step-desc {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: 0.15rem;
    }
    .trace-step-timing {
        font-family: var(--font-mono);
        font-size: 0.68rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
    }

    /* Passage Card */
    .passage-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 0.75rem;
        margin-bottom: 0.6rem;
    }
    .passage-meta {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: var(--font-mono);
        font-size: 0.7rem;
        color: var(--accent-vector);
        margin-bottom: 0.35rem;
    }
    .passage-text {
        font-size: 0.78rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }

    /* Triples Table */
    .triples-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.6rem;
        background: var(--bg-surface);
        border-radius: var(--radius-sm);
        margin-bottom: 0.35rem;
        font-family: var(--font-mono);
        font-size: 0.72rem;
    }
    .triple-subj { color: var(--text-primary); font-weight: 600; }
    .triple-rel { color: var(--accent-graph); }
    .triple-obj { color: var(--text-primary); font-weight: 600; }

    /* ─── Technical Cards & Metric Containers ──────────────────────────── */
    .tech-card {
        background: var(--bg-surface);
        border: 1px solid var(--border-medium);
        border-radius: var(--radius-md);
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .tech-card-header {
        font-family: var(--font-heading);
        font-size: 0.95rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    /* Quick Prompts */
    .prompt-chip {
        display: inline-block;
        background: var(--bg-elevated);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 0.4rem 0.8rem;
        font-size: 0.78rem;
        color: var(--text-secondary);
        margin: 0.2rem 0.3rem 0.2rem 0;
        cursor: pointer;
        transition: all 0.15s ease;
    }
    .prompt-chip:hover {
        border-color: var(--accent-vector);
        color: var(--text-primary);
    }

    /* Interactive Buttons */
    [data-testid="stBaseButton-primary"],
    [data-testid="stBaseButton-primaryFormSubmit"] {
        background: var(--accent-vector) !important;
        color: #0b0e14 !important;
        font-family: var(--font-heading) !important;
        font-weight: 700 !important;
        font-size: 0.84rem !important;
        border-radius: var(--radius-sm) !important;
        border: none !important;
    }
    [data-testid="stBaseButton-primary"]:hover,
    [data-testid="stBaseButton-primaryFormSubmit"]:hover {
        filter: brightness(1.1);
    }
    [data-testid="stBaseButton-secondary"] {
        background: var(--bg-elevated) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-medium) !important;
        border-radius: var(--radius-sm) !important;
        font-family: var(--font-heading) !important;
        font-size: 0.84rem !important;
    }

    /* Form Inputs */
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: var(--bg-input) !important;
        border: 1px solid var(--border-medium) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-body) !important;
    }
    [data-testid="stTextInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: var(--border-focus) !important;
    }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border-bottom: 1px solid var(--border-subtle) !important;
        padding: 1.2rem 0.5rem !important;
    }
    /* ─── Ultra-Refined Engineering Prompt Console ───────────────────────── */
    [data-testid="stBottom"] {
        background: linear-gradient(180deg, rgba(11, 14, 20, 0) 0%, rgba(11, 14, 20, 0.95) 30%, #0b0e14 100%) !important;
        border-top: 1px solid rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(12px) !important;
        padding-top: 0.5rem !important;
    }
    [data-testid="stBottom"] > div {
        background: transparent !important;
        padding-bottom: 1.5rem !important;
    }

    /* Outer Chat Input Container */
    [data-testid="stChatInput"],
    .stChatInput {
        background: #111622 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 14px !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        padding: 0.35rem 0.65rem !important;
        max-width: 860px !important;
        margin: 0 auto !important;
    }

    [data-testid="stChatInput"]:hover,
    .stChatInput:hover {
        border-color: rgba(56, 189, 248, 0.35) !important;
        box-shadow: 0 10px 34px rgba(0, 0, 0, 0.6), 0 0 12px rgba(56, 189, 248, 0.08) !important;
    }

    [data-testid="stChatInput"]:focus-within,
    .stChatInput:focus-within {
        border-color: #38bdf8 !important;
        background: #151d2d !important;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.65), 0 0 0 1px #38bdf8, 0 0 22px rgba(56, 189, 248, 0.22) !important;
    }

    /* Completely eliminate nested BaseWeb borders and backgrounds */
    [data-testid="stChatInput"] div,
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="base-input"],
    .stChatInput div,
    .stChatInput [data-baseweb="textarea"],
    .stChatInput [data-baseweb="base-input"] {
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }

    /* BaseWeb inner input focus overrides */
    [data-testid="stChatInput"] [data-baseweb="base-input"]:focus-within,
    .stChatInput [data-baseweb="base-input"]:focus-within {
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
    }

    /* Textarea */
    [data-testid="stChatInputTextArea"],
    [data-testid="stChatInput"] textarea,
    .stChatInput textarea {
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
        color: #f8fafc !important;
        font-family: var(--font-body) !important;
        font-size: 0.93rem !important;
        font-weight: 450 !important;
        line-height: 1.5 !important;
        padding: 0.55rem 0.75rem !important;
        min-height: 2.65rem !important;
        caret-color: #38bdf8 !important;
    }

    [data-testid="stChatInputTextArea"]::placeholder,
    [data-testid="stChatInput"] textarea::placeholder,
    .stChatInput textarea::placeholder {
        color: #64748b !important;
        font-size: 0.9rem !important;
        font-weight: 400 !important;
    }

    /* Submit Button (Precision Pill) */
    [data-testid="stChatInputSubmitButton"],
    [data-testid="stChatInput"] button,
    .stChatInput button {
        border-radius: 9px !important;
        width: 36px !important;
        height: 36px !important;
        min-width: 36px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: none !important;
        margin-right: 0.2rem !important;
        margin-bottom: 0.2rem !important;
        transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }

    /* When Disabled / Empty */
    [data-testid="stChatInputSubmitButton"]:disabled,
    [data-testid="stChatInput"] button:disabled {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        color: rgba(148, 163, 184, 0.35) !important;
        cursor: not-allowed !important;
        transform: none !important;
        box-shadow: none !important;
    }

    [data-testid="stChatInputSubmitButton"]:disabled svg,
    [data-testid="stChatInput"] button:disabled svg {
        fill: rgba(148, 163, 184, 0.35) !important;
    }

    /* When Enabled / Active (user entered query) */
    [data-testid="stChatInputSubmitButton"]:not(:disabled),
    [data-testid="stChatInput"] button:not(:disabled) {
        background: linear-gradient(135deg, #38bdf8 0%, #0284c7 100%) !important;
        border: 1px solid #38bdf8 !important;
        color: #0b0e14 !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.45) !important;
        cursor: pointer !important;
    }

    [data-testid="stChatInputSubmitButton"]:not(:disabled):hover,
    [data-testid="stChatInput"] button:not(:disabled):hover {
        transform: scale(1.06) translateY(-1px) !important;
        box-shadow: 0 0 22px rgba(56, 189, 248, 0.65) !important;
        filter: brightness(1.1) !important;
    }

    [data-testid="stChatInputSubmitButton"]:not(:disabled) svg,
    [data-testid="stChatInput"] button:not(:disabled) svg {
        fill: #0b0e14 !important;
        font-weight: 700 !important;
    }

    /* Telemetry Context Bar above input */
    .chat-console-footer-bar {
        max-width: 860px;
        margin: 1.2rem auto 0.45rem auto;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-family: var(--font-mono);
        font-size: 0.72rem;
        color: var(--text-muted);
        padding: 0 0.25rem;
    }
    .console-telemetry-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
    }
    .telemetry-live-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 8px #10b981;
        display: inline-block;
    }
    .telemetry-label {
        color: var(--accent-vector);
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .telemetry-sub {
        color: var(--text-muted);
    }
    .console-shortcuts-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        color: var(--text-muted);
    }
    .key-hint {
        background: var(--bg-surface);
        border: 1px solid var(--border-medium);
        border-radius: 3px;
        padding: 0.1rem 0.35rem;
        font-size: 0.68rem;
        color: var(--text-secondary);
    }
    .key-sep {
        color: var(--border-medium);
    }

    /* Progress bar */
    [data-testid="stProgress"] > div > div {
        background: var(--accent-vector) !important;
    }

    /* Hide Streamlit "Press Enter to submit form" hint */
    .stTextInput small,
    [data-testid="InputInstructions"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State Initialization ─────────────────────────────────────────────

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
if "prompt_seed" not in st.session_state:
    st.session_state.prompt_seed = ""
if "graph_cache" not in st.session_state:
    st.session_state.graph_cache = None

# ─── API & Helper Utilities ───────────────────────────────────────────────────

def auth_headers() -> dict:
    token = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {token}"} if token else {}

def clear_session():
    st.session_state.access_token = None
    st.session_state.current_user = None
    st.session_state.messages = []
    st.session_state.traces = {}
    st.session_state.upload_triggered = False
    st.session_state.graph_cache = None

def expire_session():
    clear_session()
    st.session_state.auth_error = "Your session expired. Please sign in again."
    st.rerun()

def save_auth_response(response):
    data = response.json()
    st.session_state.access_token = data["access_token"]
    st.session_state.current_user = data["user"]
    st.session_state.auth_error = None
    st.rerun()

def response_detail(response) -> str:
    try:
        detail = response.json().get("detail", response.text)
        if isinstance(detail, list):
            return "; ".join(item.get("msg", str(item)) for item in detail)
        return str(detail)
    except Exception:
        return response.text or f"Request failed with status {response.status_code}."

def check_health():
    try:
        resp = requests.get(API_HEALTH, timeout=4)
        if resp.status_code == 200:
            data = resp.json()
            st.session_state.health_data = data
            st.session_state.system_healthy = data.get("lm_studio", {}).get("running", False)
            return data
    except Exception:
        pass
    st.session_state.system_healthy = False
    return None

def get_processing_status():
    try:
        resp = requests.get(API_UPLOAD_STATUS, headers=auth_headers(), timeout=5)
        if resp.status_code == 401:
            expire_session()
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None

def upload_files_to_api(files):
    try:
        file_tuples = [("files", (f.name, f.getvalue(), "application/octet-stream")) for f in files]
        resp = requests.post(API_UPLOAD, files=file_tuples, headers=auth_headers(), timeout=30)
        if resp.status_code == 401:
            expire_session()
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 409:
            return {"error": "A background ingestion job is already running. Please wait."}
        else:
            return {"error": f"Upload failed: {response_detail(resp)}"}
    except requests.ConnectionError:
        return {"error": "Backend API server offline at localhost:8000."}
    except Exception as e:
        return {"error": str(e)}

def load_graph_data() -> dict:
    if st.session_state.graph_cache is not None:
        return st.session_state.graph_cache
    if GRAPH_FILE_PATH.exists() and GRAPH_FILE_PATH.stat().st_size > 0:
        try:
            with open(GRAPH_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                st.session_state.graph_cache = data
                return data
        except Exception:
            pass
    return {"nodes": [], "edges": []}

# ─── Auth Page: Split Engineering Gateway ─────────────────────────────────────

def render_auth_page():
    left_col, right_col = st.columns([1.1, 0.9], gap="large")
    
    with left_col:
        st.markdown("""<div style="padding: 2.5rem 1rem 1rem 0;">
<div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:1rem;">
    <div class="brand-mark" style="width:40px; height:40px; font-size:1.4rem;">◈</div>
    <div>
        <h1 style="font-size:1.6rem; margin:0;">Local-Agentic-GraphRAG</h1>
        <div style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-muted); letter-spacing:0.08em;">AIR-GAPPED ACADEMIC RETRIEVAL WORKBENCH</div>
    </div>
</div>
<p style="font-size:0.95rem; color:var(--text-secondary); line-height:1.65; margin-bottom:2rem;">
    Enterprise-grade Dual-Retrieval Augmented Generation operating entirely on consumer hardware. 
    Combines high-density ChromaDB semantic vectors with structured knowledge graph traversal.
</p>

<div class="tech-card">
    <div style="font-family:var(--font-heading); font-weight:700; font-size:0.9rem; margin-bottom:0.75rem; color:var(--accent-vector);">
        CORE ARCHITECTURAL PILLARS
    </div>
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:1rem; font-size:0.8rem;">
        <div>
            <strong style="color:var(--text-primary);">⚡ Gemma 4 Router Agent</strong>
            <div style="color:var(--text-muted); font-size:0.74rem; margin-top:0.2rem;">Dynamically routes factual vs relational queries.</div>
        </div>
        <div>
            <strong style="color:var(--text-primary);">🔒 Air-Gapped Execution</strong>
            <div style="color:var(--text-muted); font-size:0.74rem; margin-top:0.2rem;">Zero cloud transmission. All inference via LM Studio.</div>
        </div>
        <div>
            <strong style="color:var(--text-primary);">🕸️ Entity Knowledge Graph</strong>
            <div style="color:var(--text-muted); font-size:0.74rem; margin-top:0.2rem;">Automatic triple extraction for multi-hop reasoning.</div>
        </div>
        <div>
            <strong style="color:var(--text-primary);">🧮 Hardware Optimized</strong>
            <div style="color:var(--text-muted); font-size:0.74rem; margin-top:0.2rem;">Customized context buffers for 4GB VRAM GPUs.</div>
        </div>
    </div>
</div>
</div>""", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div style='padding-top: 2rem;'></div>", unsafe_allow_html=True)
        with st.container():
            if st.session_state.auth_error:
                st.error(st.session_state.auth_error)
                
            tab_signin, tab_signup = st.tabs(["Sign In to Studio", "Create Local Account"])
            
            with tab_signin:
                with st.form("signin_form"):
                    st.markdown("#### Authenticate Session")
                    st.caption("Credentials verified against local encrypted SQLite auth store.")
                    in_email = st.text_input("Operator Email", placeholder="user@local.lan", key="signin_email")
                    in_password = st.text_input("Master Password", type="password", key="signin_password")
                    submit = st.form_submit_button("Launch Workspace", type="primary", use_container_width=True)
                    
                    if submit:
                        if not in_email.strip() or not in_password:
                            st.warning("Please supply both email and password.")
                        else:
                            try:
                                resp = requests.post(
                                    API_AUTH_SIGNIN,
                                    json={"email": in_email.strip(), "password": in_password},
                                    timeout=10,
                                )
                                if resp.status_code == 200:
                                    save_auth_response(resp)
                                else:
                                    st.error(response_detail(resp))
                            except requests.ConnectionError:
                                st.error("Cannot connect to backend server at localhost:8000.")
                            except Exception as ex:
                                st.error(f"Error: {ex}")

            with tab_signup:
                with st.form("signup_form"):
                    st.markdown("#### Provision Local Operator")
                    new_name = st.text_input("Full Name", placeholder="Alex Vance")
                    new_email = st.text_input("Email", placeholder="alex@local.lan")
                    new_password = st.text_input("Password (min 8 chars)", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")
                    submit_reg = st.form_submit_button("Register & Initialize", type="primary", use_container_width=True)
                    
                    if submit_reg:
                        if not new_name.strip() or not new_email.strip() or not new_password:
                            st.warning("All fields are required.")
                        elif len(new_password) < 8:
                            st.warning("Password must be at least 8 characters long.")
                        elif new_password != confirm_password:
                            st.error("Passwords do not match.")
                        else:
                            try:
                                resp = requests.post(
                                    API_AUTH_SIGNUP,
                                    json={"name": new_name.strip(), "email": new_email.strip(), "password": new_password},
                                    timeout=10,
                                )
                                if resp.status_code == 201:
                                    save_auth_response(resp)
                                else:
                                    st.error(response_detail(resp))
                            except Exception as ex:
                                st.error(f"Registration error: {ex}")

# ─── Verification of Authenticated State ──────────────────────────────────────

if st.session_state.access_token:
    try:
        user_res = requests.get(API_AUTH_ME, headers=auth_headers(), timeout=5)
        if user_res.status_code == 200:
            st.session_state.current_user = user_res.json()
            st.session_state.auth_error = None
        else:
            clear_session()
            st.session_state.auth_error = "Session expired or invalid. Please sign in."
    except requests.RequestException:
        pass

if not st.session_state.access_token:
    render_auth_page()
    st.stop()

# ─── Live Telemetry Probe ─────────────────────────────────────────────────────

health = check_health()
lm_online = health.get("lm_studio", {}).get("running", False) if health else False
db_info = health.get("database", {}) if health else {}
db_ready = db_info.get("ready", False)
graph_node_count = db_info.get("graph_nodes", 0)
graph_edge_count = db_info.get("graph_edges", 0)

# ─── Auxiliary Sidebar ────────────────────────────────────────────────────────

with st.sidebar:
    user = st.session_state.get("current_user") or {}
    st.markdown(f"""
    <div class="sidebar-user-card">
        <div style="font-family:var(--font-mono); font-size:0.65rem; color:var(--text-muted); letter-spacing:0.07em;">AUTHENTICATED OPERATOR</div>
        <div class="user-name">{user.get('name', 'Operator')}</div>
        <div class="user-email">{user.get('email', '')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Sign Out of Session", use_container_width=True):
        try:
            requests.post(API_AUTH_LOGOUT, headers=auth_headers(), timeout=5)
        except Exception:
            pass
        clear_session()
        st.rerun()

    st.markdown('<div class="sidebar-section-header">Fast Document Ingestion</div>', unsafe_allow_html=True)
    sidebar_files = st.file_uploader(
        "Upload Course Docs",
        type=["pdf", "docx", "doc"],
        accept_multiple_files=True,
        key="sb_uploader",
        help="Files are processed locally into ChromaDB and Knowledge Graph.",
    )
    if sidebar_files:
        st.caption(f"{len(sidebar_files)} document(s) staged.")
        if st.button("Process & Build Graph", type="primary", use_container_width=True):
            up_res = upload_files_to_api(sidebar_files)
            if "error" in up_res:
                st.error(up_res["error"])
            else:
                st.success("Ingestion job started in background.")
                st.session_state.upload_triggered = True
                st.session_state.graph_cache = None
                time.sleep(0.5)
                st.rerun()

    # Ingestion Monitor
    proc_status = get_processing_status()
    if proc_status and proc_status.get("is_processing"):
        st.markdown('<div class="sidebar-section-header">Pipeline Stage</div>', unsafe_allow_html=True)
        pct = proc_status.get("progress", 0.0)
        st.progress(pct, text=f"{int(pct*100)}% — {proc_status.get('stage','').upper()}")
        st.caption(proc_status.get("message", "Processing chunks..."))
        time.sleep(2)
        st.rerun()
    elif proc_status and proc_status.get("stage") == "done" and st.session_state.upload_triggered:
        st.success("Knowledge Graph & Vectors Updated!")
        st.session_state.upload_triggered = False
        st.session_state.graph_cache = None
        time.sleep(1)
        st.rerun()

    st.markdown('<div class="sidebar-section-header">Session Telemetry</div>', unsafe_allow_html=True)
    total_queries = len(st.session_state.messages) // 2
    st.markdown(f"""
    <div style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-secondary); line-height:1.7;">
        <div>Conversations: <strong style="color:var(--text-primary);">{total_queries}</strong></div>
        <div>Indexed Entities: <strong style="color:var(--accent-graph);">{graph_node_count}</strong></div>
        <div>Relationships: <strong style="color:var(--accent-graph);">{graph_edge_count}</strong></div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Reset Conversation Context", use_container_width=True):
        st.session_state.messages = []
        st.session_state.traces = {}
        st.rerun()

# ─── Top Global Application Shell Bar ─────────────────────────────────────────

lm_dot_class = "online" if lm_online else "offline"
lm_label = "LM Studio Online" if lm_online else "LM Studio Offline"
db_label = f"Graph: {graph_node_count} nodes · {graph_edge_count} triples" if db_ready else "Index Empty"

st.markdown(f"""
<div class="app-shell-header">
    <div class="brand-cluster">
        <div class="brand-mark">◈</div>
        <div>
            <div class="brand-title">Local-Agentic-GraphRAG <span style="font-size:0.75rem; color:var(--accent-vector); font-weight:600; vertical-align:middle; margin-left:0.4rem;">v2.0</span></div>
            <div class="brand-subtitle">Dual-Retrieval Studio</div>
        </div>
    </div>
    <div class="telemetry-strip">
        <div class="telemetry-pill">
            <span class="telemetry-dot {lm_dot_class}"></span>
            <span>{lm_label}</span>
        </div>
        <div class="telemetry-pill">
            <span class="telemetry-dot {'online' if db_ready else 'warning'}"></span>
            <span>{db_label}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Master Workspace Navigation (4 Dedicated Workspaces) ─────────────────────

tab_query, tab_graph, tab_knowledge, tab_diagnostics = st.tabs([
    "💬  Query Studio",
    "🕸️  Knowledge Graph Explorer",
    "📚  Knowledge Base & Ingestion",
    "⚡  Engine Diagnostics & Routing"
])

# ═══════════════════════════════════════════════════════════════════════════════
#  WORKSPACE 1: QUERY STUDIO
# ═══════════════════════════════════════════════════════════════════════════════

with tab_query:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="font-size:1.1rem; margin-bottom:0.2rem;">Dual-Retrieval Reasoning Workspace</h3>
        <p style="color:var(--text-secondary); font-size:0.84rem; margin:0;">
            Gemma 4 dynamically routes questions between dense semantic vectors (ChromaDB) and knowledge graph triples.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if not lm_online:
        st.markdown("""
        <div class="tech-card" style="border-left: 3px solid var(--accent-alert);">
            <strong style="color:var(--accent-alert);">⚠️ LM Studio Is Offline</strong>
            <div style="color:var(--text-secondary); font-size:0.82rem; margin-top:0.3rem;">
                Unable to contact LM Studio at <code>http://localhost:1234/v1</code>.<br>
                Please start LM Studio and load <code>google/gemma-4-e4b</code> and <code>text-embedding-nomic-embed-text-v1.5</code>.
            </div>
        </div>
        """, unsafe_allow_html=True)

    if not db_ready:
        st.markdown("""
        <div class="tech-card" style="border-left: 3px solid var(--accent-router);">
            <strong style="color:var(--accent-router);">ℹ️ Knowledge Base Not Indexed</strong>
            <div style="color:var(--text-secondary); font-size:0.82rem; margin-top:0.3rem;">
                No document chunks or graph nodes detected. Upload academic PDF/Word documents using the sidebar or the <strong>Knowledge Base</strong> tab to begin.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Empty State with Recommended Queries
    if not st.session_state.messages:
        st.markdown("""
        <div class="tech-card" style="text-align:center; padding: 2.5rem 1.5rem; background: var(--bg-surface);">
            <div style="font-size: 1.8rem; color:var(--accent-vector); margin-bottom:0.5rem;">◈</div>
            <h4 style="margin-bottom:0.3rem;">Knowledge Base Ready for Investigation</h4>
            <p style="color:var(--text-muted); font-size:0.85rem; max-width:540px; margin: 0 auto 1.5rem auto;">
                Ask technical and conceptual questions. The query router analyzes whether to traverse semantic passages or relational concept triples.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='font-family:var(--font-mono); font-size:0.72rem; color:var(--text-muted); margin-bottom:0.4rem;'>SUGGESTED TECHNICAL INQUIRIES:</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("What is the primary objective of Local GraphRAG?", use_container_width=True):
                st.session_state.prompt_seed = "What is the primary objective of Local GraphRAG?"
                st.rerun()
        with c2:
            if st.button("How does the Dual-Retrieval architecture work?", use_container_width=True):
                st.session_state.prompt_seed = "How does the Dual-Retrieval architecture work?"
                st.rerun()
        with c3:
            if st.button("What are the key technical modules of this system?", use_container_width=True):
                st.session_state.prompt_seed = "What are the key technical modules of this system?"
                st.rerun()

    # Render Conversation Messages with Inline Provenance
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # If this is an assistant message and we have trace metadata
            if msg["role"] == "assistant" and i in st.session_state.traces:
                trace_list = st.session_state.traces[i]
                route_used = "vector"
                total_latency = 0.0
                
                for step in trace_list:
                    if step.get("step") == "classify_query":
                        route_used = step.get("route_chosen", "vector")
                    total_latency += step.get("latency_sec", 0.0)

                route_label = "Vector Semantic Search" if route_used == "vector" else "Knowledge Graph Traversal"
                badge_class = "vector" if route_used == "vector" else "graph"
                icon = "⊞" if route_used == "vector" else "☊"

                st.markdown(f"""
                <div class="provenance-bar {badge_class}">
                    <div class="route-capsule">
                        <span>{icon}</span>
                        <span>Provenance: {route_label}</span>
                    </div>
                    <div class="latency-tag">Total Pipeline Latency: {round(total_latency, 2)}s</div>
                </div>
                """, unsafe_allow_html=True)

                with st.expander("🔍 Deep Trace & Provenance Inspector", expanded=False):
                    tab_t1, tab_t2, tab_t3 = st.tabs(["Pipeline Steps", "Retrieved Evidence", "Routing Agent Output"])
                    
                    with tab_t1:
                        for s in trace_list:
                            step_name = s.get("step", "").replace("_", " ").title()
                            st.markdown(f"""
                            <div class="trace-timeline-step active">
                                <div class="trace-step-name">{step_name}</div>
                                <div class="trace-step-desc">{s.get('description', '')}</div>
                                <div class="trace-step-timing">Execution: {s.get('latency_sec', 0)}s</div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    with tab_t2:
                        for s in trace_list:
                            if s.get("step") == "retrieve_vector" and s.get("sources"):
                                st.markdown(f"**Retrieved {len(s['sources'])} Semantic Chunks:**")
                                for doc in s["sources"]:
                                    st.markdown(f"""
                                    <div class="passage-card">
                                        <div class="passage-meta">
                                            <span>📄 {doc.get('source', 'Unknown')} (Page {doc.get('page', '?')})</span>
                                            <span>Score: {doc.get('score', 'N/A')}</span>
                                        </div>
                                        <div class="passage-text">{doc.get('preview', '')}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            elif s.get("step") == "retrieve_graph" and s.get("triples"):
                                st.markdown(f"**Discovered {len(s['triples'])} Relationship Triples:**")
                                for t in s["triples"]:
                                    st.markdown(f"""
                                    <div class="triples-item">
                                        <span class="triple-subj">{t.get('source','')}</span>
                                        <span class="triple-rel">──({t.get('relation','')})──▶</span>
                                        <span class="triple-obj">{t.get('target','')}</span>
                                    </div>
                                    """, unsafe_allow_html=True)

                    with tab_t3:
                        for s in trace_list:
                            if s.get("step") == "classify_query":
                                st.caption("Raw classification token emitted by Gemma 4:")
                                st.code(s.get("llm_raw_output", s.get("route_chosen", "")), language="text")

    # Chat Input Box
    st.markdown("""
    <div class="chat-console-footer-bar">
        <div class="console-telemetry-pill">
            <span class="telemetry-live-dot"></span>
            <span class="telemetry-label">DUAL-RETRIEVAL CONSOLE</span>
            <span class="telemetry-sub">· Dense ChromaDB + NetworkX Graph</span>
        </div>
        <div class="console-shortcuts-pill">
            <span class="key-hint">↵ Enter</span> to query
            <span class="key-sep">·</span>
            <span class="key-hint">⇧ Shift+↵</span> multiline
        </div>
    </div>
    """, unsafe_allow_html=True)
    chat_prompt = st.chat_input("Enter your academic inquiry (e.g., 'What is GraphRAG and how do triples connect?')...")
    if st.session_state.prompt_seed:
        chat_prompt = st.session_state.prompt_seed
        st.session_state.prompt_seed = ""

    if chat_prompt:
        st.session_state.messages.append({"role": "user", "content": chat_prompt})
        with st.chat_message("user"):
            st.markdown(chat_prompt)

        with st.chat_message("assistant"):
            answer_holder = st.empty()
            answer_text = ""
            stream_complete = False
            stream_error = None
            trace_payload = []
            route_payload = "vector"

            with st.spinner("Agent routing & retrieving..."):
                try:
                    t_start = time.time()
                    with requests.post(
                        API_QUERY,
                        json={"query": chat_prompt},
                        headers={**auth_headers(), "Accept": "text/event-stream"},
                        timeout=300,
                        stream=True,
                    ) as resp:
                        if resp.status_code == 200:
                            event_name = None
                            event_lines = []
                            for line in resp.iter_lines(decode_unicode=True, chunk_size=1):
                                if isinstance(line, bytes):
                                    line = line.decode("utf-8")
                                if line.startswith("event: "):
                                    event_name = line[7:].strip()
                                elif line.startswith("data: "):
                                    event_lines.append(line[6:])
                                elif not line and event_name:
                                    parsed = json.loads("\n".join(event_lines))
                                    if event_name == "token":
                                        answer_text += parsed["text"]
                                        answer_holder.markdown(answer_text)
                                    elif event_name == "complete":
                                        route_payload = parsed.get("route", "vector")
                                        trace_payload = parsed.get("reasoning_trace", [])
                                        stream_complete = True
                                        break
                                    elif event_name == "error":
                                        stream_error = parsed
                                        break
                                    event_name = None
                                    event_lines = []

                            if stream_complete:
                                msg_idx = len(st.session_state.messages)
                                st.session_state.traces[msg_idx] = trace_payload
                                st.session_state.messages.append({"role": "assistant", "content": answer_text.strip()})
                                st.rerun()
                            elif stream_error:
                                st.error(f"Inference error: {stream_error}")
                        elif resp.status_code == 401:
                            expire_session()
                        else:
                            st.error(f"Engine error ({resp.status_code}): {response_detail(resp)}")
                except requests.ConnectionError:
                    st.error("Connection lost to API backend at localhost:8000.")
                except Exception as ex:
                    st.error(f"Unexpected generation exception: {ex}")


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKSPACE 2: KNOWLEDGE GRAPH EXPLORER (INTERACTIVE CANVAS & TRIPLES)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_graph:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="font-size:1.1rem; margin-bottom:0.2rem;">Interactive Knowledge Graph Visualizer</h3>
        <p style="color:var(--text-secondary); font-size:0.84rem; margin:0;">
            Explore entity concepts, relation topology, and multi-hop triples extracted from academic documents.
        </p>
    </div>
    """, unsafe_allow_html=True)

    graph_data = load_graph_data()
    all_nodes = graph_data.get("nodes", [])
    all_edges = graph_data.get("edges", [])

    if not all_nodes:
        st.info("No knowledge graph available on disk. Ingest a document in the Knowledge Base tab to build one.")
    else:
        ctrl1, ctrl2, ctrl3 = st.columns([1.5, 1.2, 1])
        with ctrl1:
            search_query = st.text_input("Search Concept Entity", placeholder="e.g. Local GraphRAG, Dual-Retrieval...", label_visibility="collapsed")
        with ctrl2:
            all_relations = sorted(list(set(e.get("relation", "") for e in all_edges if e.get("relation"))))
            rel_filter = st.selectbox("Filter Relation Type", ["All Relations"] + all_relations, label_visibility="collapsed")
        with ctrl3:
            st.markdown(f"""
            <div style="padding:0.4rem; font-family:var(--font-mono); font-size:0.75rem; text-align:right; color:var(--text-secondary);">
                Active: <strong style="color:var(--accent-graph);">{len(all_nodes)}</strong> nodes · <strong style="color:var(--accent-graph);">{len(all_edges)}</strong> edges
            </div>
            """, unsafe_allow_html=True)

        # Filter edges based on selection
        filtered_edges = all_edges
        if rel_filter != "All Relations":
            filtered_edges = [e for e in all_edges if e.get("relation") == rel_filter]

        # Prepare Graph Data JSON for HTML5 Canvas
        graph_json_str = json.dumps({
            "nodes": [{"id": n["id"], "label": n["label"], "source": n.get("properties", {}).get("source", "")} for n in all_nodes],
            "edges": [{"source": e["source"], "target": e["target"], "relation": e.get("relation", "")} for e in filtered_edges],
            "search": search_query.strip().lower()
        })

        # Self-contained Force-Directed Graph Canvas (Air-Gapped, Zero CDN Dependencies)
        graph_canvas_template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body { margin: 0; padding: 0; overflow: hidden; background: #121721; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; user-select: none; }
        #graphCanvas { display: block; width: 100vw; height: 540px; cursor: grab; }
        #graphCanvas:active { cursor: grabbing; }
        #hud { position: absolute; top: 12px; left: 14px; background: rgba(18, 23, 33, 0.85); border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 6px; padding: 6px 12px; color: #94a3b8; font-size: 11px; font-family: monospace; pointer-events: none; }
        #details { position: absolute; bottom: 12px; right: 14px; background: rgba(26, 34, 50, 0.95); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 6px; padding: 10px 14px; color: #f1f5f9; font-size: 12px; max-width: 320px; display: none; }
    </style>
</head>
<body>
    <div id="hud">Scroll: Zoom | Drag: Pan | Click Node: Inspect</div>
    <div id="details"></div>
    <canvas id="graphCanvas"></canvas>

    <script>
        const data = __GRAPH_DATA__;
        const canvas = document.getElementById('graphCanvas');
        const ctx = canvas.getContext('2d');
        const detailsBox = document.getElementById('details');

        let width = canvas.width = (window.innerWidth && window.innerWidth > 300) ? window.innerWidth : 960;
        let height = canvas.height = 540;

        const nodes = [];
        const nodeMap = {};
        
        data.nodes.forEach((n, i) => {
            const angle = (i / Math.max(data.nodes.length, 1)) * Math.PI * 2;
            const spread = 150 + (i % 4) * 45;
            const node = {
                id: n.id,
                label: n.label,
                source: n.source,
                x: (width / 2) + Math.cos(angle) * spread,
                y: (height / 2) + Math.sin(angle) * spread,
                vx: 0,
                vy: 0,
                radius: 7,
                connections: 0
            };
            nodes.push(node);
            nodeMap[n.id] = node;
        });

        data.edges.forEach(e => {
            if (nodeMap[e.source]) nodeMap[e.source].connections++;
            if (nodeMap[e.target]) nodeMap[e.target].connections++;
        });

        nodes.forEach(n => {
            n.radius = Math.min(16, 6 + n.connections * 1.5);
        });

        let transform = { x: 0, y: 0, k: 1 };
        let isDragging = false;
        let startPan = { x: 0, y: 0 };
        let selectedNode = null;

        function simulate() {
            if (window.innerWidth && window.innerWidth > 300 && canvas.width !== window.innerWidth) {
                width = canvas.width = window.innerWidth;
            }
            for (let i = 0; i < nodes.length; i++) {
                for (let j = i + 1; j < nodes.length; j++) {
                    const a = nodes[i];
                    const b = nodes[j];
                    let dx = b.x - a.x;
                    let dy = b.y - a.y;
                    let dist = Math.sqrt(dx * dx + dy * dy);
                    if (dist < 1) dist = 1;
                    if (dist < 150) {
                        const force = (150 - dist) / dist * 0.02;
                        a.vx -= dx * force;
                        a.vy -= dy * force;
                        b.vx += dx * force;
                        b.vy += dy * force;
                    }
                }
            }

            data.edges.forEach(e => {
                const a = nodeMap[e.source];
                const b = nodeMap[e.target];
                if (!a || !b) return;
                let dx = b.x - a.x;
                let dy = b.y - a.y;
                let dist = Math.sqrt(dx * dx + dy * dy) || 1;
                let force = (dist - 80) * 0.002;
                a.vx += dx * force;
                a.vy += dy * force;
                b.vx -= dx * force;
                b.vy -= dy * force;
            });

            nodes.forEach(n => {
                n.vx += (width / 2 - n.x) * 0.001;
                n.vy += (height / 2 - n.y) * 0.001;
                n.vx = Math.max(-4, Math.min(4, n.vx * 0.85));
                n.vy = Math.max(-4, Math.min(4, n.vy * 0.85));
                if (n !== selectedNode) {
                    n.x += n.vx;
                    n.y += n.vy;
                    n.x = Math.max(30, Math.min(width - 30, n.x));
                    n.y = Math.max(30, Math.min(height - 30, n.y));
                }
            });
        }

        function render() {
            simulate();
            ctx.save();
            ctx.clearRect(0, 0, width, height);

            ctx.translate(transform.x, transform.y);
            ctx.scale(transform.k, transform.k);

            data.edges.forEach(e => {
                const a = nodeMap[e.source];
                const b = nodeMap[e.target];
                if (!a || !b) return;

                ctx.beginPath();
                ctx.moveTo(a.x, a.y);
                ctx.lineTo(b.x, b.y);
                
                const isConnected = selectedNode && (selectedNode === a || selectedNode === b);
                if (isConnected) {
                    ctx.strokeStyle = '#38bdf8';
                    ctx.lineWidth = 2;
                } else {
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
                    ctx.lineWidth = 1;
                }
                ctx.stroke();

                if (transform.k > 0.8 || isConnected) {
                    const midX = (a.x + b.x) / 2;
                    const midY = (a.y + b.y) / 2;
                    ctx.fillStyle = isConnected ? '#38bdf8' : '#64748b';
                    ctx.font = '9px monospace';
                    ctx.textAlign = 'center';
                    ctx.fillText(e.relation, midX, midY - 3);
                }
            });

            nodes.forEach(n => {
                ctx.beginPath();
                ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
                
                const isMatch = data.search && n.label.toLowerCase().includes(data.search);
                const isSelected = selectedNode === n;

                if (isSelected) {
                    ctx.fillStyle = '#38bdf8';
                    ctx.shadowColor = '#38bdf8';
                    ctx.shadowBlur = 12;
                } else if (isMatch) {
                    ctx.fillStyle = '#fbbf24';
                    ctx.shadowColor = '#fbbf24';
                    ctx.shadowBlur = 10;
                } else if (n.connections > 2) {
                    ctx.fillStyle = '#10b981';
                    ctx.shadowBlur = 0;
                } else {
                    ctx.fillStyle = '#64748b';
                    ctx.shadowBlur = 0;
                }
                ctx.fill();
                ctx.shadowBlur = 0;

                ctx.strokeStyle = '#121721';
                ctx.lineWidth = 2;
                ctx.stroke();

                ctx.fillStyle = isSelected ? '#f1f5f9' : (isMatch ? '#fbbf24' : '#cbd5e1');
                ctx.font = isSelected ? 'bold 11px sans-serif' : '10px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(n.label, n.x, n.y + n.radius + 12);
            });

            ctx.restore();
            requestAnimationFrame(render);
        }

        render();

        function getNodeAt(x, y) {
            const wx = (x - transform.x) / transform.k;
            const wy = (y - transform.y) / transform.k;
            for (let n of nodes) {
                const dx = n.x - wx;
                const dy = n.y - wy;
                if (Math.sqrt(dx * dx + dy * dy) <= n.radius + 4) return n;
            }
            return null;
        }

        canvas.addEventListener('mousedown', e => {
            const rect = canvas.getBoundingClientRect();
            const mx = e.clientX - rect.left;
            const my = e.clientY - rect.top;
            const node = getNodeAt(mx, my);

            if (node) {
                selectedNode = node;
                showDetails(node);
            } else {
                isDragging = true;
                startPan = { x: e.clientX - transform.x, y: e.clientY - transform.y };
            }
        });

        window.addEventListener('mousemove', e => {
            if (selectedNode) {
                const rect = canvas.getBoundingClientRect();
                selectedNode.x = (e.clientX - rect.left - transform.x) / transform.k;
                selectedNode.y = (e.clientY - rect.top - transform.y) / transform.k;
            } else if (isDragging) {
                transform.x = e.clientX - startPan.x;
                transform.y = e.clientY - startPan.y;
            }
        });

        window.addEventListener('mouseup', () => {
            selectedNode = null;
            isDragging = false;
        });

        canvas.addEventListener('wheel', e => {
            e.preventDefault();
            const factor = e.deltaY > 0 ? 0.9 : 1.1;
            transform.k = Math.max(0.2, Math.min(3, transform.k * factor));
        });

        function showDetails(node) {
            const incoming = data.edges.filter(e => e.target === node.id).map(e => (nodeMap[e.source] ? nodeMap[e.source].label : '?') + ' [' + e.relation + ']');
            const outgoing = data.edges.filter(e => e.source === node.id).map(e => '[' + e.relation + '] ➔ ' + (nodeMap[e.target] ? nodeMap[e.target].label : '?'));
            
            detailsBox.innerHTML = '<div style="font-weight:bold; color:#38bdf8; margin-bottom:4px;">' + node.label + '</div>' +
                '<div style="font-size:10px; color:#94a3b8; margin-bottom:6px;">Source Doc: ' + (node.source || 'Local Document') + '</div>' +
                '<div style="font-size:11px; color:#cbd5e1;">Connections: <strong>' + node.connections + '</strong></div>' +
                (outgoing.length ? '<div style="font-size:10px; color:#10b981; margin-top:4px;">Outgoing: ' + outgoing.slice(0,3).join(', ') + '</div>' : '') +
                (incoming.length ? '<div style="font-size:10px; color:#f59e0b; margin-top:2px;">Incoming: ' + incoming.slice(0,3).join(', ') + '</div>' : '');
            detailsBox.style.display = 'block';
        }
    </script>
</body>
</html>"""
        graph_canvas_html = graph_canvas_template.replace("__GRAPH_DATA__", graph_json_str)
        components.html(graph_canvas_html, height=560)

        # Bottom Relationship Triples Catalog Table
        st.markdown("<h4 style='font-size:0.95rem; margin-top:1.5rem;'>Extracted Knowledge Graph Triples Catalog</h4>", unsafe_allow_html=True)
        node_lookup = {n["id"]: n["label"] for n in all_nodes}
        triples_data = []
        for e in filtered_edges:
            s_label = node_lookup.get(e.get("source", ""), e.get("source", ""))
            t_label = node_lookup.get(e.get("target", ""), e.get("target", ""))
            triples_data.append({
                "Subject (Source Entity)": s_label,
                "Predicate (Relation)": e.get("relation", ""),
                "Object (Target Entity)": t_label,
            })
        st.dataframe(triples_data, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKSPACE 3: KNOWLEDGE BASE & INGESTION WORKSPACE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_knowledge:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="font-size:1.1rem; margin-bottom:0.2rem;">Knowledge Base Repository & Dynamic Ingestion</h3>
        <p style="color:var(--text-secondary); font-size:0.84rem; margin:0;">
            Manage ingested documents, monitor the extraction pipeline, and inspect raw vector embeddings stored in ChromaDB.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_up, col_info = st.columns([1.2, 0.8], gap="medium")
    
    with col_up:
        st.markdown('<div class="tech-card">', unsafe_allow_html=True)
        st.markdown("<div class='tech-card-header'><span>Document Upload & Ingestion Dropzone</span><span style='font-size:0.75rem; color:var(--text-muted);'>PDF / DOCX (max 100MB)</span></div>", unsafe_allow_html=True)
        kb_files = st.file_uploader(
            "Select academic files",
            type=["pdf", "docx", "doc"],
            accept_multiple_files=True,
            key="kb_uploader",
        )
        if kb_files:
            for f in kb_files:
                st.caption(f"Staged: {f.name} ({len(f.getvalue())/1024:.1f} KB)")
            if st.button("Start Extraction Pipeline", type="primary", use_container_width=True):
                result = upload_files_to_api(kb_files)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("Files queued for background processing!")
                    st.session_state.upload_triggered = True
                    st.session_state.graph_cache = None
                    time.sleep(0.5)
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown("""
        <div class="tech-card">
            <div class="tech-card-header">Extraction Pipeline Stages</div>
            <div style="font-size:0.8rem; color:var(--text-secondary); line-height:1.7;">
                <div><strong>1. Text Parsing:</strong> Extracts text blocks via PyPDF or python-docx.</div>
                <div><strong>2. Semantic Chunking:</strong> 1000 characters with 200 char overlap.</div>
                <div><strong>3. Vector Embedding:</strong> Batch-embedded via Nomic Embed Text v1.5.</div>
                <div><strong>4. Knowledge Graph:</strong> Gemma 4 extracts entity-relationship triples.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Ingested Documents Inventory & ChromaDB Inspection
    st.markdown("<h4 style='font-size:0.95rem; margin-top:1rem;'>ChromaDB Vector Chunks Catalog</h4>", unsafe_allow_html=True)
    
    # Try reading sample chunks from ChromaDB directly
    try:
        from chromadb import PersistentClient
        chroma_client = PersistentClient(str(CHROMA_DIR_PATH))
        col_chroma = chroma_client.get_collection("local_graphrag")
        total_items = col_chroma.count()
        sample_data = col_chroma.get(limit=10)
        
        st.caption(f"Active collection: 'local_graphrag' ({total_items} total chunks embedded). Showing top 10 chunks:")
        
        chunks_display = []
        for i in range(len(sample_data.get("ids", []))):
            meta = sample_data["metadatas"][i] if sample_data.get("metadatas") else {}
            doc_text = sample_data["documents"][i] if sample_data.get("documents") else ""
            chunks_display.append({
                "Chunk ID": sample_data["ids"][i],
                "Source Document": meta.get("source", "Unknown"),
                "Section": meta.get("section", 1),
                "Length": f"{len(doc_text)} chars",
                "Content Preview": doc_text[:160] + "..." if len(doc_text) > 160 else doc_text
            })
        st.dataframe(chunks_display, use_container_width=True, hide_index=True)
    except Exception as e:
        st.caption(f"Unable to read ChromaDB chunks preview directly: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  WORKSPACE 4: ENGINE DIAGNOSTICS & TELEMETRY WORKSPACE
# ═══════════════════════════════════════════════════════════════════════════════

with tab_diagnostics:
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="font-size:1.1rem; margin-bottom:0.2rem;">Engine Architecture & System Diagnostics</h3>
        <p style="color:var(--text-secondary); font-size:0.84rem; margin:0;">
            Verify LM Studio model endpoints, LangGraph state machine routing, and hardware VRAM allocations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    d1, d2 = st.columns(2, gap="medium")

    with d1:
        st.markdown("""
        <div class="tech-card">
            <div class="tech-card-header">LangGraph Dual-Path Routing State Machine</div>
            <div style="font-family:var(--font-mono); font-size:0.75rem; background:var(--bg-base); padding:1rem; border-radius:4px; border:1px solid var(--border-subtle); line-height:1.7;">
                [User Question]<br>
                &nbsp;&nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;&nbsp;▼<br>
                <strong>classify_query (Gemma 4 Router)</strong><br>
                &nbsp;&nbsp;&nbsp;&nbsp;├─────── "vector" ───────► <strong>retrieve_vector (ChromaDB)</strong><br>
                &nbsp;&nbsp;&nbsp;&nbsp;└─────── "graph"  ───────► <strong>retrieve_graph (graph.json)</strong><br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>generate_answer (Gemma 4 Synthesiser)</strong><br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▼<br>
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[Streaming Academic Answer]
            </div>
        </div>
        """, unsafe_allow_html=True)

    with d2:
        lm_info = health.get("lm_studio", {}) if health else {}
        st.markdown(f"""
        <div class="tech-card">
            <div class="tech-card-header">Model Inventory & Hardware Settings</div>
            <div style="font-family:var(--font-mono); font-size:0.75rem; line-height:1.8;">
                <div>Server Endpoint: <span style="color:var(--text-primary);">http://localhost:1234/v1</span></div>
                <div>Router Model: <span style="color:var(--accent-vector);">google/gemma-4-e4b</span> ({'Ready' if lm_info.get('router_ready') else 'Offline'})</div>
                <div>Synthesiser Model: <span style="color:var(--accent-vector);">google/gemma-4-e4b</span> ({'Ready' if lm_info.get('llm_ready') else 'Offline'})</div>
                <div>Embedding Model: <span style="color:var(--accent-vector);">text-embedding-nomic-embed-text-v1.5</span> ({'Ready' if lm_info.get('embed_ready') else 'Offline'})</div>
                <hr style="border:none; border-top:1px solid var(--border-subtle); margin:0.6rem 0;">
                <div>Target GPU: <span style="color:var(--text-primary);">AMD Radeon RX 6500M (4GB VRAM)</span></div>
                <div>Context Window: <span style="color:var(--text-primary);">4096 tokens</span></div>
                <div>Router Max Tokens: <span style="color:var(--text-primary);">128</span></div>
                <div>Synthesis Max Tokens: <span style="color:var(--text-primary);">2048</span></div>
                <div>Temperature: <span style="color:var(--text-primary);">0.2</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Live Router Classifier Sandbox
    st.markdown("<h4 style='font-size:0.95rem; margin-top:1rem;'>Live Query Router Test Sandbox</h4>", unsafe_allow_html=True)
    st.caption("Test how the Gemma 4 router agent classifies queries into Vector vs Graph without executing the full synthesis.")
    
    with st.form("router_sandbox_form"):
        test_q = st.text_input("Test Query", value="How does vector retrieval compare to graph traversal?")
        run_test = st.form_submit_button("Test Router Decision", type="primary")
        
        if run_test:
            try:
                from core.config import client, ROUTER_MODEL, ROUTER_SYSTEM_PROMPT, ROUTER_MAX_TOKENS, TEMPERATURE
                t0 = time.time()
                res = client.chat.completions.create(
                    model=ROUTER_MODEL,
                    messages=[
                        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                        {"role": "user", "content": test_q},
                    ],
                    max_tokens=ROUTER_MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
                raw = res.choices[0].message.content.strip()
                import re
                matches = re.findall(r'\b(vector|graph)\b', raw, re.IGNORECASE)
                chosen = matches[-1].lower() if matches else "vector"
                t_elapsed = round(time.time() - t0, 3)
                
                badge_color = "var(--accent-vector)" if chosen == "vector" else "var(--accent-graph)"
                st.markdown(f"""
                <div style="background:var(--bg-surface); padding:1rem; border-radius:4px; border:1px solid {badge_color}; margin-top:0.75rem;">
                    <div>Classification: <strong style="color:{badge_color}; font-size:1rem; text-transform:uppercase;">{chosen}</strong> ({t_elapsed}s)</div>
                    <div style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-muted); margin-top:0.3rem;">Raw output: {raw}</div>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Sandbox classification failed: {e}")
