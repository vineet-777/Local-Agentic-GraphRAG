/**
 * Local Agentic GraphRAG — God-Level Web Application Controller
 * Features:
 * - Multi-Turn Conversation Memory (past turns sent for contextual follow-up)
 * - Server-Sent Events (SSE) Live Token Streaming with Typewriter Effect
 * - Real-Time o1-Style Collapsible Reasoning Trace Updates
 * - Interactive 3D WebGL Knowledge Graph Visualizer Modal
 * - Local Multi-Profile Session Isolation (Default / Researcher / Academic)
 * - One-Click Conversation Share & Export Engine
 * - Neo4j Enterprise Dual-Graph Engine Telemetry
 * - Bottom Paperclip Multi-Format Document Ingestion with Floating Progress
 */

document.addEventListener('DOMContentLoaded', () => {
  // ── DOM References ──────────────────────────────────────────────────────────
  const sidebar = document.getElementById('sidebar');
  const toggleSidebarBtn = document.getElementById('toggleSidebarBtn');
  const newChatBtn = document.getElementById('newChatBtn');
  const historyList = document.getElementById('historyList');
  const profileSelect = document.getElementById('profileSelect');

  // File Upload Elements (Paperclip in Bottom Composer)
  const fileInput = document.getElementById('fileInput');
  const attachBtn = document.getElementById('attachBtn');
  const progressCard = document.getElementById('progressCard');
  const progressStage = document.getElementById('progressStage');
  const progressPercent = document.getElementById('progressPercent');
  const progressBarFill = document.getElementById('progressBarFill');
  const progressDesc = document.getElementById('progressDesc');

  // Header Actions
  const graphModalBtn = document.getElementById('graphModalBtn');
  const shareChatBtn = document.getElementById('shareChatBtn');
  const themeToggleBtn = document.getElementById('themeToggleBtn');
  const themeIcon = document.getElementById('themeIcon');

  // Telemetry Elements (Bottom of Sidebar)
  const lmDot = document.getElementById('lmDot');
  const lmText = document.getElementById('lmText');
  const llmModelTag = document.getElementById('llmModelTag');
  const embedModelTag = document.getElementById('embedModelTag');
  const graphStats = document.getElementById('graphStats');
  const chromaStatus = document.getElementById('chromaStatus');
  const neo4jStatus = document.getElementById('neo4jStatus');

  // Chat Canvas Elements
  const chatViewport = document.getElementById('chatViewport');
  const heroView = document.getElementById('heroView');
  const messagesList = document.getElementById('messagesList');
  const queryInput = document.getElementById('queryInput');
  const sendBtn = document.getElementById('sendBtn');
  const promptCards = document.querySelectorAll('.prompt-card');

  // 3D Graph Modal Elements
  const graphModal = document.getElementById('graphModal');
  const graphCloseBtn = document.getElementById('graphCloseBtn');
  const graphModalBackdrop = document.getElementById('graphModalBackdrop');
  const graphSearchInput = document.getElementById('graphSearchInput');
  const graphResetBtn = document.getElementById('graphResetBtn');
  const graphAutoRotateBtn = document.getElementById('graphAutoRotateBtn');
  const closeInspectorBtn = document.getElementById('closeInspectorBtn');
  const nodeInspector = document.getElementById('nodeInspector');

  // ── State Variables ────────────────────────────────────────────────────────
  let currentSessionId = null;
  let isGenerating = false;
  let statusPollInterval = null;
  let graph3dInstance = null;

  const STORAGE_KEY_PROFILE = 'graphrag_current_profile';
  const STORAGE_KEY_THEME = 'graphrag_theme';
  let currentProfile = localStorage.getItem(STORAGE_KEY_PROFILE) || 'Default';

  function getSessionsKey() {
    return `graphrag_sessions_${currentProfile}`;
  }

  // ── 1. Theme Toggle (Dark & Light Mode) ────────────────────────────────────

  function initTheme() {
    const savedTheme = localStorage.getItem(STORAGE_KEY_THEME) || 'dark';
    applyTheme(savedTheme);
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(STORAGE_KEY_THEME, theme);

    if (theme === 'light') {
      themeIcon.innerHTML = `
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
      `;
    } else {
      themeIcon.innerHTML = `
        <circle cx="12" cy="12" r="5"></circle>
        <line x1="12" y1="1" x2="12" y2="3"></line>
        <line x1="12" y1="21" x2="12" y2="23"></line>
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
        <line x1="1" y1="12" x2="3" y2="12"></line>
        <line x1="21" y1="12" x2="23" y2="12"></line>
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
      `;
    }
  }

  themeToggleBtn.addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    applyTheme(current === 'dark' ? 'light' : 'dark');
  });

  initTheme();

  // ── 2. Local Multi-Profile Switcher ───────────────────────────────────────

  if (profileSelect) {
    profileSelect.value = currentProfile;
    profileSelect.addEventListener('change', () => {
      currentProfile = profileSelect.value;
      localStorage.setItem(STORAGE_KEY_PROFILE, currentProfile);
      currentSessionId = null;
      renderHistoryList();
      showHeroView();
    });
  }

  // ── 3. Chat History Management (localStorage) ─────────────────────────────

  function getSessions() {
    try {
      return JSON.parse(localStorage.getItem(getSessionsKey()) || '[]');
    } catch {
      return [];
    }
  }

  function saveSessions(sessions) {
    localStorage.setItem(getSessionsKey(), JSON.stringify(sessions));
  }

  function renderHistoryList() {
    const sessions = getSessions();
    historyList.innerHTML = '';

    if (sessions.length === 0) {
      historyList.innerHTML = `
        <div style="font-size: 0.74rem; color: var(--text-muted); padding: 8px 10px;">
          No conversations in "${currentProfile}" profile
        </div>
      `;
      return;
    }

    sessions.forEach(sess => {
      const item = document.createElement('div');
      item.className = `history-item ${sess.id === currentSessionId ? 'active' : ''}`;
      item.setAttribute('data-id', sess.id);

      const titleSpan = document.createElement('span');
      titleSpan.className = 'history-title';
      titleSpan.textContent = sess.title || 'Conversation';
      titleSpan.title = sess.title || 'Conversation';

      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'history-delete-btn';
      deleteBtn.title = 'Delete conversation';
      deleteBtn.innerHTML = `
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="3 6 5 6 21 6"></polyline>
          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
        </svg>
      `;

      deleteBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        deleteSession(sess.id);
      });

      item.addEventListener('click', () => {
        loadSession(sess.id);
      });

      item.appendChild(titleSpan);
      item.appendChild(deleteBtn);
      historyList.appendChild(item);
    });
  }

  function createNewSession() {
    currentSessionId = 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6);
    showHeroView();
    renderHistoryList();
    if (queryInput) queryInput.focus();
  }

  function loadSession(id) {
    const sessions = getSessions();
    const session = sessions.find(s => s.id === id);
    if (!session) return;

    currentSessionId = id;
    renderHistoryList();

    if (!session.messages || session.messages.length === 0) {
      showHeroView();
      return;
    }

    heroView.style.display = 'none';
    messagesList.style.display = 'flex';
    messagesList.innerHTML = '';

    session.messages.forEach(msg => {
      if (msg.role === 'user') {
        appendUserMessage(msg.text, false);
      } else {
        appendAssistantMessage(msg.text, msg.reasoningTrace, msg.route, false);
      }
    });

    scrollToBottom();
  }

  function deleteSession(id) {
    let sessions = getSessions();
    sessions = sessions.filter(s => s.id !== id);
    saveSessions(sessions);

    if (currentSessionId === id) {
      createNewSession();
    } else {
      renderHistoryList();
    }
  }

  function saveMessageToCurrentSession(role, text, reasoningTrace = null, route = null) {
    let sessions = getSessions();
    let session = sessions.find(s => s.id === currentSessionId);

    if (!session) {
      session = {
        id: currentSessionId,
        title: text.slice(0, 36) + (text.length > 36 ? '...' : ''),
        createdAt: new Date().toISOString(),
        profile: currentProfile,
        messages: [],
      };
      sessions.unshift(session);
    }

    session.messages.push({
      role,
      text,
      reasoningTrace,
      route,
      timestamp: new Date().toISOString(),
    });

    saveSessions(sessions);
    renderHistoryList();
  }

  function showHeroView() {
    messagesList.innerHTML = '';
    messagesList.style.display = 'none';
    heroView.style.display = 'flex';
  }

  newChatBtn.addEventListener('click', () => {
    createNewSession();
  });

  createNewSession();

  // ── 4. Telemetry Checks (LM Studio, ChromaDB, Neo4j) ──────────────────────

  async function fetchHealth() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error('Health check failed');
      const data = await res.json();

      const lm = data.lm_studio || {};
      const db = data.database || {};
      const neo = data.neo4j || {};

      if (lm.running) {
        lmDot.className = 'dot online';
        lmText.textContent = 'Connected (:1234)';
        llmModelTag.textContent = lm.configured_llm_model ? lm.configured_llm_model.split('/').pop() : 'gemma-4-e4b';
        embedModelTag.textContent = lm.configured_embed_model ? `${lm.configured_embed_model.split('/').pop()} (${lm.embedding_dim || 768}d)` : 'embeddinggemma-2';
      } else {
        lmDot.className = 'dot offline';
        lmText.textContent = 'Offline';
      }

      const nodes = db.graph_nodes || 0;
      const edges = db.graph_edges || 0;
      graphStats.textContent = `${nodes} nodes • ${edges} edges`;
      chromaStatus.textContent = db.chromadb_exists ? 'Indexed (Ready)' : 'Empty';

      if (neo4jStatus) {
        if (neo.connected) {
          neo4jStatus.style.color = 'var(--openai-green)';
          neo4jStatus.textContent = `Connected (:7687)`;
        } else {
          neo4jStatus.style.color = 'var(--text-muted)';
          neo4jStatus.textContent = 'In-Memory (Dual-Mode)';
        }
      }
    } catch {
      lmDot.className = 'dot offline';
      lmText.textContent = 'Offline';
    }
  }

  fetchHealth();
  setInterval(fetchHealth, 10000);

  // ── 5. Sidebar Toggle & Keyboard Shortcuts ────────────────────────────────

  toggleSidebarBtn.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
  });

  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') {
      e.preventDefault();
      sidebar.classList.toggle('collapsed');
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      createNewSession();
    }
  });

  // ── 6. Multi-Format Paperclip Ingestion ────────────────────────────────────

  attachBtn.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleFileUpload(fileInput.files);
    }
  });

  async function handleFileUpload(files) {
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    progressCard.classList.add('active');
    progressStage.textContent = 'Uploading...';
    progressPercent.textContent = '0%';
    progressBarFill.style.width = '5%';
    progressDesc.textContent = `Uploading ${files.length} document(s)...`;

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Upload failed');
      }

      startStatusPolling();
    } catch (err) {
      progressStage.textContent = 'Upload Failed';
      progressDesc.textContent = err.message;
      progressBarFill.style.background = '#ef4444';
      setTimeout(() => {
        progressCard.classList.remove('active');
        progressBarFill.style.background = '';
      }, 5000);
    }
  }

  function startStatusPolling() {
    if (statusPollInterval) clearInterval(statusPollInterval);

    statusPollInterval = setInterval(async () => {
      try {
        const res = await fetch('/api/upload-status');
        if (!res.ok) return;
        const status = await res.json();

        if (status.is_processing) {
          const pct = Math.round(status.progress * 100);
          progressStage.textContent = (status.stage || 'Processing').toUpperCase();
          progressPercent.textContent = `${pct}%`;
          progressBarFill.style.width = `${Math.max(5, pct)}%`;
          progressDesc.textContent = status.message || `Processing ${status.current_file}...`;
        } else if (status.stage === 'done') {
          clearInterval(statusPollInterval);
          progressStage.textContent = 'INGESTION COMPLETE';
          progressPercent.textContent = '100%';
          progressBarFill.style.width = '100%';
          progressDesc.textContent = 'All documents embedded into ChromaDB & Knowledge Graph!';
          fetchHealth();
          setTimeout(() => {
            progressCard.classList.remove('active');
          }, 3500);
        } else if (status.stage === 'error') {
          clearInterval(statusPollInterval);
          progressStage.textContent = 'ERROR';
          progressDesc.textContent = status.error || 'Ingestion encountered an error.';
          progressBarFill.style.background = '#ef4444';
          setTimeout(() => {
            progressCard.classList.remove('active');
            progressBarFill.style.background = '';
          }, 4000);
        }
      } catch {
        // Silent catch
      }
    }, 800);
  }

  // ── 7. Composer & Sending Logic ───────────────────────────────────────────

  function updateSendBtnState() {
    const hasText = queryInput.value.trim().length > 0;
    sendBtn.disabled = !hasText || isGenerating;
    if (hasText && !isGenerating) {
      sendBtn.classList.add('active');
    } else {
      sendBtn.classList.remove('active');
    }
  }

  function adjustTextareaHeight() {
    queryInput.style.height = 'auto';
    queryInput.style.height = Math.min(queryInput.scrollHeight, 160) + 'px';
  }

  queryInput.addEventListener('input', () => {
    adjustTextareaHeight();
    updateSendBtnState();
  });

  queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!sendBtn.disabled) {
        handleUserSubmit(queryInput.value.trim());
      }
    }
  });

  sendBtn.addEventListener('click', () => {
    if (!sendBtn.disabled) {
      handleUserSubmit(queryInput.value.trim());
    }
  });

  promptCards.forEach(card => {
    card.addEventListener('click', () => {
      const prompt = card.getAttribute('data-prompt');
      if (prompt) {
        queryInput.value = prompt;
        handleUserSubmit(prompt);
      }
    });
  });

  // ── 8. SSE Streaming Query Submission & Multi-Turn Memory ─────────────────

  async function handleUserSubmit(query) {
    if (!query || isGenerating) return;

    queryInput.value = '';
    adjustTextareaHeight();
    updateSendBtnState();

    heroView.style.display = 'none';
    messagesList.style.display = 'flex';

    // 1. Append User Message
    appendUserMessage(query, true);

    // 2. Gather conversation history for Multi-Turn Reasoning
    const session = getSessions().find(s => s.id === currentSessionId);
    const conversation_history = (session && session.messages) ? session.messages.slice(-4).map(m => ({
      role: m.role,
      content: m.text,
    })) : [];

    // 3. Create Assistant Row with o1 Thinking Accordion
    const { assistantRow, thinkingSummary, thinkingDetails, bodyEl } = createAssistantRow();
    messagesList.appendChild(assistantRow);
    scrollToBottom();

    isGenerating = true;
    updateSendBtnState();

    const startTime = Date.now();
    const timerInterval = setInterval(() => {
      const elapsed = Math.round((Date.now() - startTime) / 1000);
      thinkingSummary.innerHTML = `
        <div class="o1-left">
          <div class="thinking-pulse-dot"></div>
          <span>Reasoning for ${elapsed}s...</span>
        </div>
      `;
    }, 1000);

    let accumulatedText = '';
    let traces = [];
    let detectedRoute = 'hybrid';

    try {
      // Stream tokens using Server-Sent Events (SSE)
      const response = await fetch('/api/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, conversation_history }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop(); // Keep uncompleted chunk

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const payload = JSON.parse(line.slice(6));
            if (payload.event === 'trace') {
              traces.push(payload.data);
              renderThinkingTraces(thinkingDetails, traces);
            } else if (payload.event === 'start_generation') {
              detectedRoute = payload.route || 'hybrid';
              clearInterval(timerInterval);
              const totalSec = Math.max(1, Math.round((Date.now() - startTime) / 1000));
              thinkingSummary.innerHTML = `
                <div class="o1-left">
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
                  <span>Thought for ${totalSec}s</span>
                  <span class="route-tag ${detectedRoute}">${detectedRoute.toUpperCase()}</span>
                </div>
                <svg class="chevron-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
              `;
              bodyEl.innerHTML = '<span class="streaming-cursor"></span>';
            } else if (payload.event === 'token') {
              accumulatedText += payload.token;
              bodyEl.innerHTML = renderMarkdown(accumulatedText) + '<span class="streaming-cursor"></span>';
              attachCodeCopyButtons(bodyEl);
              scrollToBottom();
            } else if (payload.event === 'done') {
              accumulatedText = payload.response || accumulatedText;
            }
          } catch {
            // Skip unparseable chunks
          }
        }
      }

      // Finalize rendering
      clearInterval(timerInterval);
      const finalSec = Math.max(1, Math.round((Date.now() - startTime) / 1000));
      thinkingSummary.innerHTML = `
        <div class="o1-left">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
          <span>Thought for ${finalSec}s</span>
          <span class="route-tag ${detectedRoute}">${detectedRoute.toUpperCase()}</span>
        </div>
        <svg class="chevron-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
      `;

      bodyEl.innerHTML = renderMarkdown(accumulatedText || 'No response generated.');
      attachCodeCopyButtons(bodyEl);
      saveMessageToCurrentSession('assistant', accumulatedText, traces, detectedRoute);

    } catch (err) {
      console.warn('Streaming encountered issue, falling back to sync query:', err);
      // Fallback to synchronous endpoint
      try {
        const syncRes = await fetch('/api/query', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, conversation_history }),
        });
        clearInterval(timerInterval);
        const data = await syncRes.json();
        const finalSec = Math.max(1, Math.round((Date.now() - startTime) / 1000));
        const route = data.route || 'hybrid';
        thinkingSummary.innerHTML = `
          <div class="o1-left">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
            <span>Thought for ${finalSec}s</span>
            <span class="route-tag ${route}">${route.toUpperCase()}</span>
          </div>
          <svg class="chevron-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
        `;
        renderThinkingTraces(thinkingDetails, data.reasoning_trace || []);
        bodyEl.innerHTML = renderMarkdown(data.response || 'No response generated.');
        attachCodeCopyButtons(bodyEl);
        saveMessageToCurrentSession('assistant', data.response, data.reasoning_trace, route);
      } catch (syncErr) {
        clearInterval(timerInterval);
        bodyEl.innerHTML = `<div style="color: #ef4444;">Error connecting to engine: ${syncErr.message}</div>`;
      }
    } finally {
      clearInterval(timerInterval);
      isGenerating = false;
      updateSendBtnState();
      scrollToBottom();
    }
  }

  // ── 9. Message DOM Creation & Helpers ─────────────────────────────────────

  function appendUserMessage(text, save = true) {
    const row = document.createElement('div');
    row.className = 'message-row user';
    row.innerHTML = `
      <div class="user-bubble">${escapeHtml(text)}</div>
    `;
    messagesList.appendChild(row);
    if (save) saveMessageToCurrentSession('user', text);
  }

  function appendAssistantMessage(text, reasoningTrace, route, save = true) {
    const { assistantRow, thinkingSummary, thinkingDetails, bodyEl } = createAssistantRow();
    const routeClass = route || 'vector';
    thinkingSummary.innerHTML = `
      <div class="o1-left">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
        <span>Reasoning Trace</span>
        <span class="route-tag ${routeClass}">${routeClass.toUpperCase()}</span>
      </div>
      <svg class="chevron-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"></polyline></svg>
    `;

    renderThinkingTraces(thinkingDetails, reasoningTrace || []);
    bodyEl.innerHTML = renderMarkdown(text || '');
    attachCodeCopyButtons(bodyEl);

    messagesList.appendChild(assistantRow);
    if (save) saveMessageToCurrentSession('assistant', text, reasoningTrace, route);
  }

  function createAssistantRow() {
    const row = document.createElement('div');
    row.className = 'message-row assistant';

    const avatarBox = document.createElement('div');
    avatarBox.className = 'assistant-avatar-box';
    avatarBox.innerHTML = `<img src="./assets/logo.svg" alt="GraphRAG">`;

    const contentBox = document.createElement('div');
    contentBox.className = 'assistant-content';

    const thinkingBox = document.createElement('div');
    thinkingBox.className = 'o1-thinking-box';

    const thinkingSummary = document.createElement('div');
    thinkingSummary.className = 'o1-summary';
    thinkingSummary.innerHTML = `
      <div class="o1-left">
        <div class="thinking-pulse-dot"></div>
        <span>Thinking...</span>
      </div>
    `;

    const thinkingDetails = document.createElement('div');
    thinkingDetails.className = 'o1-details';

    thinkingSummary.addEventListener('click', () => {
      thinkingDetails.classList.toggle('open');
      const chevron = thinkingSummary.querySelector('.chevron-icon');
      if (chevron) {
        chevron.style.transform = thinkingDetails.classList.contains('open') ? 'rotate(180deg)' : 'rotate(0deg)';
      }
    });

    thinkingBox.appendChild(thinkingSummary);
    thinkingBox.appendChild(thinkingDetails);

    const bodyEl = document.createElement('div');
    bodyEl.className = 'assistant-body';

    contentBox.appendChild(thinkingBox);
    contentBox.appendChild(bodyEl);

    row.appendChild(avatarBox);
    row.appendChild(contentBox);

    return { assistantRow: row, thinkingSummary, thinkingDetails, bodyEl };
  }

  function renderThinkingTraces(container, traces) {
    if (!traces || traces.length === 0) {
      container.innerHTML = '<div style="font-size:0.75rem; color:var(--text-muted); padding:4px 0;">No trace steps recorded.</div>';
      return;
    }

    let html = '';
    traces.forEach(step => {
      const stepName = (step.step || 'Step').replace(/_/g, ' ').toUpperCase();
      const latency = step.latency_sec ? `${step.latency_sec}s` : '';
      html += `
        <div class="trace-step">
          <div class="trace-step-header">
            <span>🔹 ${stepName}</span>
            <span class="trace-latency">${latency}</span>
          </div>
          <div class="trace-desc">${escapeHtml(step.description || '')}</div>
      `;

      if (step.sources && step.sources.length > 0) {
        html += `<div class="trace-sources">`;
        step.sources.forEach(src => {
          html += `
            <div class="trace-source-item">
              <strong>${escapeHtml(src.source)}</strong> (p.${src.page}): 
              <em>${escapeHtml(src.preview || '')}</em>
            </div>
          `;
        });
        html += `</div>`;
      }

      if (step.triples && step.triples.length > 0) {
        html += `<div class="trace-sources">`;
        step.triples.forEach(t => {
          html += `
            <div class="trace-source-item">
              🕸️ <strong>${escapeHtml(t.source)}</strong> —(${escapeHtml(t.relation)})→ <strong>${escapeHtml(t.target)}</strong>
            </div>
          `;
        });
        html += `</div>`;
      }

      html += `</div>`;
    });

    container.innerHTML = html;
  }

  // ── 10. 3D Knowledge Graph Modal Wiring ───────────────────────────────────

  if (graphModalBtn) {
    graphModalBtn.addEventListener('click', async () => {
      graphModal.style.display = 'flex';
      if (!graph3dInstance && window.KnowledgeGraph3D) {
        graph3dInstance = new window.KnowledgeGraph3D('graph3dCanvas', 'graphCanvasWrapper', 'nodeInspector');
        await graph3dInstance.loadData();
        graph3dInstance.start();
        const badge = document.getElementById('graphModalBadge');
        if (badge) {
          badge.textContent = `${graph3dInstance.nodes.length} Nodes • ${graph3dInstance.links.length} Edges`;
        }
      } else if (graph3dInstance) {
        graph3dInstance.resize();
        graph3dInstance.start();
      }
    });
  }

  function closeGraphModal() {
    if (graphModal) graphModal.style.display = 'none';
    if (graph3dInstance) graph3dInstance.stop();
  }

  if (graphCloseBtn) graphCloseBtn.addEventListener('click', closeGraphModal);
  if (graphModalBackdrop) graphModalBackdrop.addEventListener('click', closeGraphModal);

  if (graphSearchInput) {
    graphSearchInput.addEventListener('input', (e) => {
      if (graph3dInstance) graph3dInstance.filter(e.target.value);
    });
  }

  if (graphResetBtn) {
    graphResetBtn.addEventListener('click', () => {
      if (graph3dInstance) graph3dInstance.reset();
      if (graphSearchInput) graphSearchInput.value = '';
    });
  }

  if (graphAutoRotateBtn) {
    let autoRotate = true;
    graphAutoRotateBtn.addEventListener('click', () => {
      autoRotate = !autoRotate;
      if (graph3dInstance) graph3dInstance.autoRotate = autoRotate;
      graphAutoRotateBtn.textContent = `Auto-Rotate: ${autoRotate ? 'ON' : 'OFF'}`;
    });
  }

  if (closeInspectorBtn && nodeInspector) {
    closeInspectorBtn.addEventListener('click', () => {
      nodeInspector.style.display = 'none';
    });
  }

  // ── 11. Conversation Share / Export Engine ────────────────────────────────

  if (shareChatBtn) {
    shareChatBtn.addEventListener('click', () => {
      const session = getSessions().find(s => s.id === currentSessionId);
      if (!session || !session.messages || session.messages.length === 0) {
        alert('Start a conversation first before exporting or sharing!');
        return;
      }

      const exportPayload = {
        title: session.title || 'GraphRAG Conversation',
        exportedAt: new Date().toISOString(),
        profile: currentProfile,
        messages: session.messages,
        engine: "Local-Agentic-GraphRAG v2.0 (Gemma 4 & EmbeddingGemma-2)",
      };

      const blob = new Blob([JSON.stringify(exportPayload, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${(session.title || 'GraphRAG_Chat').replace(/[^a-zA-Z0-9_-]/g, '_')}_export.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });
  }

  // ── 12. Markdown & Code Copy Helpers ──────────────────────────────────────

  function renderMarkdown(md) {
    if (!md) return '';
    let html = escapeHtml(md);

    // Code blocks
    html = html.replace(/```([a-zA-Z0-9_]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      const language = lang || 'code';
      return `
        <div class="code-block-wrapper">
          <div class="code-block-header">
            <span>${language}</span>
            <button class="copy-code-btn" data-code="${encodeURIComponent(code)}">Copy code</button>
          </div>
          <pre><code>${code}</code></pre>
        </div>
      `;
    });

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold & italic
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Headers
    html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Lists
    html = html.replace(/^\s*[-•]\s+(.*$)/gim, '<li>$1</li>');
    html = html.replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>');

    // Paragraphs
    html = html.replace(/\n\n+/g, '</p><p>');
    html = `<p>${html}</p>`;
    html = html.replace(/<p><\/p>/g, '');

    return html;
  }

  function attachCodeCopyButtons(container) {
    container.querySelectorAll('.copy-code-btn').forEach(btn => {
      btn.onclick = () => {
        const code = decodeURIComponent(btn.getAttribute('data-code'));
        navigator.clipboard.writeText(code).then(() => {
          btn.textContent = 'Copied!';
          setTimeout(() => { btn.textContent = 'Copy code'; }, 2000);
        });
      };
    });
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function scrollToBottom() {
    chatViewport.scrollTop = chatViewport.scrollHeight;
  }
});
