"""
Local GraphRAG — Production UI/UX Overhaul
Air-Gapped Engineering Studio Architecture:
- Two-Workspace Shell (Query Studio, Graph Explorer)
- Interactive Force-Directed Knowledge Graph Explorer (60fps HTML5 Canvas, zero external deps)
- Real-time Dual-Stream Reasoning Telemetry & Source Provenance Drawer
- Air-Gapped Authentication & Session Management
"""

import sys
import json
import html
import time
from datetime import datetime, timedelta
from pathlib import Path

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
API_AUTH_SIGNUP = f"{API_BASE}/api/auth/signup"
API_AUTH_SIGNIN = f"{API_BASE}/api/auth/signin"
API_AUTH_ME = f"{API_BASE}/api/auth/me"
API_AUTH_LOGOUT = f"{API_BASE}/api/auth/logout"
API_CONVERSATIONS = f"{API_BASE}/api/conversations"

GRAPH_FILE_PATH = PROJECT_ROOT / "db" / "graph.json"

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

    /* Keep the 1.56 sidebar controls visible without the default toolbar. */
    header[data-testid="stHeader"] {
        height: 2.5rem;
        min-height: 2.5rem;
        background: transparent;
        box-shadow: none;
        overflow: visible;
        z-index: 1000;
    }
    header[data-testid="stHeader"] [data-testid="stAppDeployButton"],
    header[data-testid="stHeader"] [data-testid="stMainMenu"] {
        display: none !important;
    }
    [data-testid="stSidebarHeader"] {
        display: flex !important;
        height: 2.5rem;
        min-height: 2.5rem;
        padding: 0;
        overflow: visible;
        border: 0;
        position: relative;
    }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarExpandButton"] {
        position: absolute !important;
        top: 0.25rem;
        right: 0.25rem;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 1001;
    }
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarExpandButton"] button,
    button[data-testid="stExpandSidebarButton"],
    [data-testid="collapsedControl"] button,
    [data-testid="stSidebarCollapsedControl"] button {
        visibility: visible !important;
        opacity: 1 !important;
        color: var(--text-secondary) !important;
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-sm) !important;
        z-index: 1001;
    }
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        position: fixed !important;
        top: 0.3rem;
        left: 0.3rem;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 1001;
    }
    button[data-testid="stExpandSidebarButton"] {
        position: fixed !important;
        top: 0.5rem !important;
        left: 0.5rem !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 1001;
    }

    /* Streamlit block container width and spacing */
    .block-container {
        max-width: 1400px;
        padding-top: 1rem !important;
        padding-bottom: 4rem;
        padding-left: 2rem;
        padding-right: 2rem;
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
        display: flex;
        flex-direction: column;
        flex: 1 1 auto;
        height: 100%;
        min-height: calc(100vh - 1rem);
        padding: 0.75rem 0.65rem 0.6rem;
        gap: 0.15rem;
    }
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        overflow: visible !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        display: flex;
        flex-direction: column;
        height: 100vh;
        min-height: 0;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        flex: 1 1 auto;
        display: flex;
        flex-direction: column;
        min-height: 0;
        padding-bottom: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div {
        display: flex;
        flex: 1 1 100%;
        flex-direction: column;
        height: 100% !important;
        min-height: 0;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {
        display: flex;
        flex: 1 1 100% !important;
        flex-direction: column;
        min-height: 100%;
        height: 100%;
    }

    .sidebar-user-card {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.5rem 0.35rem 0.15rem;
    }

    .user-name {
        font-family: var(--font-heading);
        font-weight: 600;
        font-size: 0.8rem;
        color: var(--text-primary);
    }

    .user-email {
        font-family: var(--font-mono);
        font-size: 0.62rem;
        color: var(--text-muted);
        overflow-wrap: anywhere;
        margin-top: 0.08rem;
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

    .sidebar-brand-row {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.15rem 0.25rem 0.7rem;
        font-family: var(--font-heading);
        font-size: 0.9rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text-primary);
    }
    .sidebar-brand-mark {
        color: var(--accent-vector);
        font-size: 1.15rem;
    }
    .sidebar-avatar {
        width: 2rem;
        height: 2rem;
        flex: 0 0 2rem;
        display: grid;
        place-items: center;
        border: 1px solid var(--accent-vector-border);
        border-radius: 50%;
        background: var(--accent-vector-bg);
        color: var(--accent-vector);
        font: 600 0.67rem var(--font-mono);
    }
    .st-key-new_chat button {
        min-height: 2.25rem;
        justify-content: flex-start;
        padding-left: 0.7rem;
        background: transparent !important;
        border: 1px solid var(--border-medium) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-size: 0.83rem;
    }
    .st-key-new_chat button:hover {
        background: var(--bg-elevated) !important;
        border-color: var(--border-medium) !important;
    }
    .st-key-chat_list {
        flex: 1 1 auto;
        min-height: 0;
        max-height: none;
        overflow-y: auto;
        scrollbar-width: thin;
        scrollbar-color: var(--border-medium) transparent;
        padding-right: 0.1rem;
    }
    .st-key-chat_list::-webkit-scrollbar { width: 5px; }
    .st-key-chat_list::-webkit-scrollbar-thumb {
        background: var(--border-medium);
        border-radius: 8px;
    }
    .chat-group-label {
        margin: 0.85rem 0 0.28rem;
        padding-left: 0.35rem;
        color: var(--text-muted);
        font: 500 0.61rem var(--font-mono);
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }
    section[data-testid="stSidebar"] .stButton button {
        min-height: 2rem;
        padding: 0.32rem 0.45rem;
        border: 1px solid transparent;
        background: transparent;
        color: var(--text-secondary);
        font-size: 0.79rem;
        text-align: left;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        box-shadow: none;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background: var(--bg-elevated);
        color: var(--text-primary);
        border-color: transparent;
    }
    section[data-testid="stSidebar"] .stButton button:focus-visible {
        outline: 1px solid var(--accent-vector);
        outline-offset: 1px;
    }
    section[data-testid="stSidebar"] [data-testid="stPopover"] > button {
        min-width: 1.7rem;
        width: 1.7rem;
        min-height: 1.8rem;
        padding: 0;
        font-size: 1rem;
        text-align: center;
    }
    .st-key-chat_active_row {
        padding-left: 0.25rem;
        border-left: 2px solid var(--accent-vector);
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        background: var(--accent-vector-bg);
    }
    .st-key-chat_active_row .stButton button {
        color: var(--text-primary);
    }
    .st-key-sidebar_footer {
        flex: 0 0 auto;
        padding-top: 0.4rem;
        background: var(--bg-surface);
        border-top: 1px solid var(--border-subtle);
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"]
    [data-testid="stLayoutWrapper"]:has(> .stVerticalBlock.st-key-sidebar_footer) {
        margin-top: auto !important;
    }
    .st-key-profile_menu [data-testid="stPopoverButton"] {
        width: 100%;
        min-height: 2.5rem;
        justify-content: flex-start;
        padding: 0.35rem 0.55rem;
        border: 1px solid transparent;
        border-radius: var(--radius-md);
        background: transparent;
        color: var(--text-primary);
        font-family: var(--font-heading);
        font-size: 0.82rem;
        font-weight: 600;
        text-align: left;
        transition: background 120ms ease, border-color 120ms ease;
    }
    .st-key-profile_menu [data-testid="stPopoverButton"] [data-testid="stIconMaterial"] {
        display: none;
    }
    .st-key-profile_menu [data-testid="stPopoverButton"]:hover {
        background: var(--bg-elevated);
        border-color: var(--border-subtle);
    }
    .st-key-profile_menu [data-testid="stPopoverButton"]:focus-visible {
        outline: 1px solid var(--accent-vector);
        outline-offset: 1px;
    }
    .st-key-profile_menu [data-testid="stPopoverButton"]::after {
        content: "⌃";
        margin-left: auto;
        color: var(--text-muted);
        font-size: 0.78rem;
    }
    body [data-testid="stPopoverBody"]:has(.st-key-profile_menu_content),
    body [data-testid="stPopoverBody"]:has(.st-key-system_status_content):not(:has(.st-key-profile_menu_content)) {
        z-index: 1100;
        overflow: visible;
        border: 1px solid var(--border-medium);
        border-radius: var(--radius-md);
        background: var(--bg-elevated);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.38);
        transition:
            opacity 120ms ease,
            transform 120ms ease,
            display 120ms allow-discrete;
    }
    body [data-testid="stPopoverBody"]:has(.st-key-profile_menu_content) {
        min-width: 13rem;
        transform-origin: bottom center;
    }
    body [data-testid="stPopoverBody"]:has(.st-key-system_status_content):not(:has(.st-key-profile_menu_content)) {
        position: fixed !important;
        position-anchor: --system-status-trigger;
        position-area: right;
        position-try-fallbacks: flip-inline;
        min-width: 15rem;
        left: auto !important;
        top: auto !important;
        bottom: auto !important;
        transform: none !important;
        transform-origin: top left;
        margin-left: 0.45rem;
    }
    .st-key-system_status_popup [data-testid="stPopoverButton"] {
        anchor-name: --system-status-trigger;
        width: 100%;
        min-height: 2rem;
        justify-content: flex-start;
        padding: 0.32rem 0.45rem;
        text-align: left;
    }
    .st-key-profile_menu .stButton button {
        width: 100%;
        min-height: 2rem;
        justify-content: flex-start;
        padding: 0.32rem 0.45rem;
    }
    @starting-style {
        body [data-testid="stPopoverBody"]:has(.st-key-profile_menu_content),
        body [data-testid="stPopoverBody"]:has(.st-key-system_status_content):not(:has(.st-key-profile_menu_content)) {
            opacity: 0;
            transform: translateY(4px) scale(0.98);
        }
    }
    @media (prefers-reduced-motion: reduce) {
        body [data-testid="stPopoverBody"]:has(.st-key-profile_menu_content),
        body [data-testid="stPopoverBody"]:has(.st-key-system_status_content):not(:has(.st-key-profile_menu_content)) {
            transition: none;
        }
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
if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = None
if "conversations" not in st.session_state:
    st.session_state.conversations = None
if "conversations_error" not in st.session_state:
    st.session_state.conversations_error = False
if "rename_target" not in st.session_state:
    st.session_state.rename_target = None
if "delete_confirm_target" not in st.session_state:
    st.session_state.delete_confirm_target = None
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
    st.session_state.active_conversation_id = None
    st.session_state.conversations = None
    st.session_state.conversations_error = False
    st.session_state.rename_target = None
    st.session_state.delete_confirm_target = None
    st.session_state.graph_cache = None

def expire_session():
    clear_session()
    st.session_state.auth_error = "Your session expired. Please sign in again."
    st.rerun()

def save_auth_response(response):
    data = response.json()
    st.session_state.access_token = data["access_token"]
    st.session_state.current_user = data["user"]
    st.session_state.active_conversation_id = None
    st.session_state.messages = []
    st.session_state.traces = {}
    st.session_state.conversations = None
    st.session_state.conversations_error = False
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
            return resp.json()
    except Exception:
        pass
    return None

def refresh_conversations():
    try:
        response = requests.get(API_CONVERSATIONS, headers=auth_headers(), timeout=5)
        if response.status_code == 401:
            expire_session()
        if response.status_code == 200:
            st.session_state.conversations = response.json()
            st.session_state.conversations_error = False
        else:
            st.session_state.conversations = []
            st.session_state.conversations_error = True
    except requests.RequestException:
        st.session_state.conversations = []
        st.session_state.conversations_error = True

def load_conversation(conversation_id: str) -> bool:
    try:
        response = requests.get(
            f"{API_CONVERSATIONS}/{conversation_id}",
            headers=auth_headers(),
            timeout=10,
        )
        if response.status_code == 401:
            expire_session()
        if response.status_code != 200:
            st.sidebar.error(response_detail(response))
            return False
        conversation = response.json()
        messages = conversation.get("messages", [])
        st.session_state.messages = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
        ]
        st.session_state.traces = {
            index: message.get("reasoning_trace", [])
            for index, message in enumerate(messages)
            if message.get("role") == "assistant" and message.get("reasoning_trace")
        }
        st.session_state.active_conversation_id = conversation_id
        st.session_state.rename_target = None
        st.session_state.delete_confirm_target = None
        return True
    except requests.RequestException:
        st.sidebar.error("Couldn't load this conversation.")
        return False

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

# Load once per session-state cache invalidation, not on every Streamlit rerun.
if st.session_state.conversations is None:
    refresh_conversations()

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
    st.markdown(
        '<div class="sidebar-brand-row"><span class="sidebar-brand-mark">◈</span>'
        '<span>Local GraphRAG</span></div>',
        unsafe_allow_html=True,
    )
    if st.button("＋  New chat", key="new_chat", use_container_width=True):
        st.session_state.active_conversation_id = None
        st.session_state.messages = []
        st.session_state.traces = {}
        st.session_state.rename_target = None
        st.session_state.delete_confirm_target = None
        st.rerun()

    chat_search = st.text_input(
        "Search chats",
        placeholder="Search chats",
        label_visibility="collapsed",
        key="chat_search",
    )
    conversations = st.session_state.conversations or []
    if st.session_state.conversations_error:
        st.caption("Couldn't load chats")
    elif not conversations:
        st.markdown(
            '<div style="padding:0.7rem 0.35rem;color:var(--text-muted);font-size:0.75rem;">'
            "No conversations yet</div>",
            unsafe_allow_html=True,
        )
    else:
        today = datetime.now().date()
        grouped_conversations = {
            "Today": [],
            "Yesterday": [],
            "Previous 7 Days": [],
            "Older": [],
        }
        for conversation in conversations:
            title = conversation.get("title", "")
            if chat_search and chat_search.casefold() not in title.casefold():
                continue
            updated = datetime.fromtimestamp(conversation["updated_at"]).date()
            if updated == today:
                group = "Today"
            elif updated == today - timedelta(days=1):
                group = "Yesterday"
            elif updated >= today - timedelta(days=7):
                group = "Previous 7 Days"
            else:
                group = "Older"
            grouped_conversations[group].append(conversation)

        with st.container(key="chat_list"):
            for group_name, group_conversations in grouped_conversations.items():
                if not group_conversations:
                    continue
                st.markdown(
                    f'<div class="chat-group-label">{group_name}</div>',
                    unsafe_allow_html=True,
                )
                for conversation in group_conversations:
                    conversation_id = conversation["id"]
                    active = conversation_id == st.session_state.active_conversation_id
                    row_context = st.container(
                        key="chat_active_row" if active else f"chat_row_{conversation_id}"
                    )
                    with row_context:
                        title_col, actions_col = st.columns([0.84, 0.16], gap="small")
                        with title_col:
                            if st.button(
                                conversation["title"],
                                key=f"chat_select_{conversation_id}",
                                use_container_width=True,
                            ):
                                if load_conversation(conversation_id):
                                    st.rerun()
                        with actions_col:
                            with st.popover("⋯", key=f"chat_actions_{conversation_id}"):
                                if st.session_state.rename_target == conversation_id:
                                    renamed_title = st.text_input(
                                        "Rename",
                                        value=conversation["title"],
                                        max_chars=80,
                                        key=f"chat_rename_input_{conversation_id}",
                                    )
                                    if st.button(
                                        "Save",
                                        key=f"chat_rename_save_{conversation_id}",
                                        use_container_width=True,
                                    ):
                                        try:
                                            response = requests.patch(
                                                f"{API_CONVERSATIONS}/{conversation_id}",
                                                json={"title": renamed_title},
                                                headers=auth_headers(),
                                                timeout=5,
                                            )
                                            if response.status_code == 401:
                                                expire_session()
                                            if response.status_code == 200:
                                                st.session_state.rename_target = None
                                                refresh_conversations()
                                                st.rerun()
                                            else:
                                                st.sidebar.error(response_detail(response))
                                        except requests.RequestException:
                                            st.sidebar.error("Couldn't rename this conversation.")
                                elif st.button(
                                    "Rename",
                                    key=f"chat_rename_{conversation_id}",
                                    use_container_width=True,
                                ):
                                    st.session_state.rename_target = conversation_id
                                    st.rerun()

                                if st.session_state.delete_confirm_target == conversation_id:
                                    if st.button(
                                        "Confirm delete",
                                        key=f"chat_delete_confirm_{conversation_id}",
                                        use_container_width=True,
                                    ):
                                        try:
                                            response = requests.delete(
                                                f"{API_CONVERSATIONS}/{conversation_id}",
                                                headers=auth_headers(),
                                                timeout=5,
                                            )
                                            if response.status_code == 401:
                                                expire_session()
                                            if response.status_code == 204:
                                                if st.session_state.active_conversation_id == conversation_id:
                                                    st.session_state.active_conversation_id = None
                                                    st.session_state.messages = []
                                                    st.session_state.traces = {}
                                                st.session_state.delete_confirm_target = None
                                                st.session_state.rename_target = None
                                                refresh_conversations()
                                                st.rerun()
                                            else:
                                                st.sidebar.error(response_detail(response))
                                        except requests.RequestException:
                                            st.sidebar.error("Couldn't delete this conversation.")
                                    if st.button(
                                        "Cancel",
                                        key=f"chat_delete_cancel_{conversation_id}",
                                        use_container_width=True,
                                    ):
                                        st.session_state.delete_confirm_target = None
                                        st.rerun()
                                elif st.button(
                                    "Delete",
                                    key=f"chat_delete_{conversation_id}",
                                    use_container_width=True,
                                ):
                                    st.session_state.delete_confirm_target = conversation_id
                                    st.rerun()

    with st.container(key="sidebar_footer"):
        name = str(user.get("name", "Operator"))
        initials = "".join(part[0] for part in name.split()[:2]).upper() or "OP"
        avatar_letter = initials[0]
        avatar = (
            chr(0x24B6 + ord(avatar_letter) - ord("A"))
            if "A" <= avatar_letter <= "Z"
            else avatar_letter
        )
        profile_label = f"{avatar}  {html.escape(name.upper())}"
        with st.popover(
            profile_label,
            key="profile_menu",
            use_container_width=True,
        ):
            with st.container(key="profile_menu_content"):
                with st.popover(
                    "System status",
                    key="system_status_popup",
                    icon=":material/monitor_heart:",
                    use_container_width=True,
                ):
                    with st.container(key="system_status_content"):
                        st.markdown(
                            f'<div style="font-family:var(--font-mono);font-size:0.72rem;'
                            f'color:var(--text-secondary);line-height:1.8;">'
                            f'<div>LM Studio: <strong style="color:{"var(--accent-graph)" if lm_online else "var(--accent-alert)"};">'
                            f'{"Online" if lm_online else "Offline"}</strong></div>'
                            f'<div>Indexed entities: <strong style="color:var(--text-primary);">{graph_node_count}</strong></div>'
                            f'<div>Relationships: <strong style="color:var(--text-primary);">{graph_edge_count}</strong></div>'
                            "</div>",
                            unsafe_allow_html=True,
                        )
                if st.button(
                    "Sign out",
                    key="sign_out",
                    icon=":material/logout:",
                    use_container_width=True,
                ):
                    try:
                        requests.post(API_AUTH_LOGOUT, headers=auth_headers(), timeout=5)
                    except requests.RequestException:
                        pass
                    clear_session()
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

# ─── Master Workspace Navigation ──────────────────────────────────────────────

tab_query, tab_graph = st.tabs([
    "💬  Query Studio",
    "🕸️  Knowledge Graph Explorer"
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
            <strong style="color:var(--accent-router);">ℹ️ No Indexed Data Available</strong>
            <div style="color:var(--text-secondary); font-size:0.82rem; margin-top:0.3rem;">
                No document chunks or graph nodes were detected. Query results and graph exploration may be limited until data is indexed.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Empty State with Recommended Queries
    if not st.session_state.messages:
        st.markdown("""
        <div class="tech-card" style="text-align:center; padding: 2.5rem 1.5rem; background: var(--bg-surface);">
            <div style="font-size: 1.8rem; color:var(--accent-vector); margin-bottom:0.5rem;">◈</div>
            <h4 style="margin-bottom:0.3rem;">Ready for Investigation</h4>
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
            with st.spinner("Agent routing & retrieving..."):
                try:
                    t_start = time.time()
                    with requests.post(
                        API_QUERY,
                        json={
                            "query": chat_prompt,
                            "conversation_id": st.session_state.active_conversation_id,
                        },
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
                                        trace_payload = parsed.get("reasoning_trace", [])
                                        st.session_state.active_conversation_id = parsed.get(
                                            "conversation_id",
                                            st.session_state.active_conversation_id,
                                        )
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
                                refresh_conversations()
                                st.rerun()
                            elif stream_error:
                                st.error(f"Inference error: {stream_error}")
                                refresh_conversations()
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
        st.info("No knowledge graph is available on disk.")
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
