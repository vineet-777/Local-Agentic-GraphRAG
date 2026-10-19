/**
 * Local Agentic GraphRAG — Web Application Controller
 * Features: Chat History (localStorage), Dark/Light Mode,
 * Bottom Paperclip Multi-format Ingestion, Docked Left Telemetry,
 * and OpenAI o1-Style Reasoning Accordion.
 */

document.addEventListener('DOMContentLoaded', () => {
  // ── DOM References ──────────────────────────────────────────────────────────
  const sidebar = document.getElementById('sidebar');
  const toggleSidebarBtn = document.getElementById('toggleSidebarBtn');
  const newChatBtn = document.getElementById('newChatBtn');
  const historyList = document.getElementById('historyList');

  // File Upload Elements (Paperclip in Bottom Composer)
  const fileInput = document.getElementById('fileInput');
  const attachBtn = document.getElementById('attachBtn');
  const progressCard = document.getElementById('progressCard');
  const progressStage = document.getElementById('progressStage');
  const progressPercent = document.getElementById('progressPercent');
  const progressBarFill = document.getElementById('progressBarFill');
  const progressDesc = document.getElementById('progressDesc');

  // Theme Toggle Button
  const themeToggleBtn = document.getElementById('themeToggleBtn');
  const themeIcon = document.getElementById('themeIcon');

  // Docked Left Telemetry Elements (Bottom of Sidebar)
  const lmDot = document.getElementById('lmDot');
  const lmText = document.getElementById('lmText');
  const llmModelTag = document.getElementById('llmModelTag');
  const embedModelTag = document.getElementById('embedModelTag');
  const graphStats = document.getElementById('graphStats');
  const chromaStatus = document.getElementById('chromaStatus');

  // Chat Canvas Elements
  const chatViewport = document.getElementById('chatViewport');
  const heroView = document.getElementById('heroView');
  const messagesList = document.getElementById('messagesList');
  const queryInput = document.getElementById('queryInput');
  const sendBtn = document.getElementById('sendBtn');
  const promptCards = document.querySelectorAll('.prompt-card');

  // ── State Variables ────────────────────────────────────────────────────────
  let currentSessionId = null;
  let isGenerating = false;
  let statusPollInterval = null;

  const STORAGE_KEY_SESSIONS = 'graphrag_sessions';
  const STORAGE_KEY_THEME = 'graphrag_theme';

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

  // ── 2. Chat History Management (localStorage) ─────────────────────────────

  function getSessions() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY_SESSIONS) || '[]');
    } catch {
      return [];
    }
  }

  function saveSessions(sessions) {
    localStorage.setItem(STORAGE_KEY_SESSIONS, JSON.stringify(sessions));
  }

  function renderHistoryList() {
    const sessions = getSessions();
    historyList.innerHTML = '';

    if (sessions.length === 0) {
      historyList.innerHTML = `
        <div style="font-size: 0.74rem; color: var(--text-muted); padding: 6px 10px;">
          No conversations yet
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
    currentSessionId = 'sess_' + Date.now();
    messagesList.innerHTML = '';
    messagesList.style.display = 'none';
    heroView.style.display = 'flex';
    queryInput.value = '';
    adjustTextareaHeight();
    updateSendBtnState();
    renderHistoryList();
    queryInput.focus();
  }

  function deleteSession(id) {
    let sessions = getSessions().filter(s => s.id !== id);
    saveSessions(sessions);
    if (currentSessionId === id) {
      createNewSession();
    } else {
      renderHistoryList();
    }
  }

  function loadSession(id) {
    const sessions = getSessions();
    const session = sessions.find(s => s.id === id);
    if (!session) return;

    currentSessionId = session.id;
    messagesList.innerHTML = '';

    if (session.messages && session.messages.length > 0) {
      heroView.style.display = 'none';
      messagesList.style.display = 'flex';

      session.messages.forEach(msg => {
        if (msg.role === 'user') {
          appendUserMessage(msg.content, false);
        } else {
          appendAssistantMessage(msg.content, msg.route, msg.trace, msg.seconds || 2, false);
        }
      });
      scrollToBottom();
    } else {
      messagesList.style.display = 'none';
      heroView.style.display = 'flex';
    }

    renderHistoryList();
  }

  function saveCurrentMessage(role, content, route = null, trace = null, seconds = 0) {
    let sessions = getSessions();
    let session = sessions.find(s => s.id === currentSessionId);

    if (!session) {
      session = {
        id: currentSessionId,
        title: content.slice(0, 32).trim() + (content.length > 32 ? '...' : ''),
        timestamp: Date.now(),
        messages: [],
      };
      sessions.unshift(session);
    }

    session.messages.push({ role, content, route, trace, seconds });
    saveSessions(sessions);
    renderHistoryList();
  }

  newChatBtn.addEventListener('click', createNewSession);

  // Initialize with latest session or new
  const existingSessions = getSessions();
  if (existingSessions.length > 0) {
    loadSession(existingSessions[0].id);
  } else {
    createNewSession();
  }

  // ── 3. Docked Left Telemetry Checks (LM Studio & Knowledge Graph) ─────────

  async function fetchHealth() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error('Health check failed');
      const data = await res.json();

      const lm = data.lm_studio || {};
      const db = data.database || {};

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
    } catch {
      lmDot.className = 'dot offline';
      lmText.textContent = 'Offline';
    }
  }

  fetchHealth();
  setInterval(fetchHealth, 10000);

  // ── 4. Sidebar Toggle & Shortcuts ─────────────────────────────────────────

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

  // ── 5. Multi-Format Upload (Only Paperclip in Bottom Composer) ─────────────

  attachBtn.addEventListener('click', () => fileInput.click());

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      uploadFiles(fileInput.files);
    }
  });

  // Also support drag-and-drop directly onto chat viewport
  ['dragenter', 'dragover'].forEach(name => {
    chatViewport.addEventListener(name, (e) => {
      e.preventDefault();
      chatViewport.style.backgroundColor = 'rgba(16, 163, 127, 0.04)';
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    chatViewport.addEventListener(name, (e) => {
      e.preventDefault();
      chatViewport.style.backgroundColor = '';
    });
  });

  chatViewport.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) uploadFiles(files);
  });

  async function uploadFiles(files) {
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    try {
      progressCard.classList.add('active');
      progressStage.textContent = 'Uploading...';
      progressPercent.textContent = '0%';
      progressBarFill.style.width = '5%';
      progressDesc.textContent = `Uploading ${files.length} file(s)...`;

      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Upload failed');
      }

      startStatusPolling();
    } catch (err) {
      progressStage.textContent = 'Upload Error';
      progressDesc.textContent = err.message;
      progressBarFill.style.background = '#ef4444';
      setTimeout(() => {
        progressCard.classList.remove('active');
        progressBarFill.style.background = '';
      }, 5000);
    } finally {
      fileInput.value = '';
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
          const pct = Math.round((status.progress || 0) * 100);
          progressStage.textContent = (status.stage || 'Processing').toUpperCase();
          progressPercent.textContent = `${pct}%`;
          progressBarFill.style.width = `${Math.max(pct, 5)}%`;
          progressDesc.textContent = status.message || `Processing ${status.current_file}...`;
        } else {
          clearInterval(statusPollInterval);
          statusPollInterval = null;

          if (status.error) {
            progressStage.textContent = 'FAILED';
            progressDesc.textContent = status.error;
            progressBarFill.style.background = '#ef4444';
          } else {
            progressStage.textContent = 'COMPLETE';
            progressPercent.textContent = '100%';
            progressBarFill.style.width = '100%';
            progressDesc.textContent = 'Knowledge Graph & Embeddings updated!';
            fetchHealth();
          }

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

  // ── 6. Composer & Sending Logic ───────────────────────────────────────────

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

  // ── 7. Query Submission & Reasoning Accordion ─────────────────────────────

  async function handleUserSubmit(query) {
    if (!query || isGenerating) return;

    queryInput.value = '';
    adjustTextareaHeight();
    updateSendBtnState();

    heroView.style.display = 'none';
    messagesList.style.display = 'flex';

    // Append and save user message
    appendUserMessage(query, true);

    // Create Assistant Row
    const { assistantRow, thinkingSummary, thinkingDetails, bodyEl } = createAssistantRow();
    messagesList.appendChild(assistantRow);
    scrollToBottom();

    // Start thinking timer
    isGenerating = true;
    updateSendBtnState();
    const startTime = Date.now();
    const timerInterval = setInterval(() => {
      const elapsed = Math.round((Date.now() - startTime) / 1000);
      thinkingSummary.innerHTML = `
        <div class="o1-left">
          <div class="thinking-pulse-dot"></div>
          <span>Thinking for ${elapsed}s...</span>
        </div>
      `;
    }, 1000);

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      clearInterval(timerInterval);
      const totalSeconds = Math.max(1, Math.round((Date.now() - startTime) / 1000));

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error (${response.status})`);
      }

      const data = await response.json();

      // Render Thinking Trace
      renderThinkingTrace(thinkingSummary, thinkingDetails, data.route, data.reasoning_trace, totalSeconds);

      // Render Response
      bodyEl.innerHTML = formatMarkdown(data.response || 'No response returned.');
      setupCodeCopyButtons(bodyEl);

      // Save to chat session history
      saveCurrentMessage('assistant', data.response, data.route, data.reasoning_trace, totalSeconds);

    } catch (err) {
      clearInterval(timerInterval);
      thinkingSummary.innerHTML = `
        <div class="o1-left">
          <span style="color: #ef4444;">Error occurred</span>
        </div>
      `;
      bodyEl.innerHTML = `<p style="color: #ef4444; background: rgba(239, 68, 68, 0.1); padding: 12px; border-radius: 8px; border: 1px solid rgba(239, 68, 68, 0.25);">
        ⚠️ <strong>Generation Error:</strong> ${escapeHtml(err.message)}
      </p>`;
    } finally {
      isGenerating = false;
      updateSendBtnState();
      scrollToBottom();
      queryInput.focus();
    }
  }

  function appendUserMessage(text, shouldSave = false) {
    const row = document.createElement('div');
    row.className = 'message-row user';
    row.innerHTML = `<div class="user-bubble">${escapeHtml(text)}</div>`;
    messagesList.appendChild(row);
    if (shouldSave) {
      saveCurrentMessage('user', text);
    }
  }

  function appendAssistantMessage(content, route, trace, seconds = 2, shouldSave = false) {
    const { assistantRow, thinkingSummary, thinkingDetails, bodyEl } = createAssistantRow();
    renderThinkingTrace(thinkingSummary, thinkingDetails, route, trace, seconds);
    bodyEl.innerHTML = formatMarkdown(content);
    setupCodeCopyButtons(bodyEl);
    messagesList.appendChild(assistantRow);
    if (shouldSave) {
      saveCurrentMessage('assistant', content, route, trace, seconds);
    }
  }

  function createAssistantRow() {
    const assistantRow = document.createElement('div');
    assistantRow.className = 'message-row assistant';

    const avatar = document.createElement('div');
    avatar.className = 'assistant-avatar';
    avatar.innerHTML = `<img src="./assets/logo.svg" alt="AI" style="width: 28px; height: 28px;">`;

    const assistantContent = document.createElement('div');
    assistantContent.className = 'assistant-body';

    // Thinking Box
    const thinkingBox = document.createElement('div');
    thinkingBox.className = 'o1-thinking-box';

    const thinkingSummary = document.createElement('div');
    thinkingSummary.className = 'o1-thinking-summary';
    thinkingSummary.innerHTML = `
      <div class="o1-left">
        <div class="thinking-pulse-dot"></div>
        <span>Thinking...</span>
      </div>
    `;

    const thinkingDetails = document.createElement('div');
    thinkingDetails.className = 'o1-details';

    thinkingSummary.addEventListener('click', () => {
      thinkingBox.classList.toggle('expanded');
    });

    thinkingBox.appendChild(thinkingSummary);
    thinkingBox.appendChild(thinkingDetails);

    const bodyEl = document.createElement('div');
    bodyEl.className = 'markdown-output';

    assistantContent.appendChild(thinkingBox);
    assistantContent.appendChild(bodyEl);

    assistantRow.appendChild(avatar);
    assistantRow.appendChild(assistantContent);

    return { assistantRow, thinkingSummary, thinkingDetails, bodyEl };
  }

  function renderThinkingTrace(summaryEl, detailsEl, route, trace, seconds) {
    const routeClass = `route-${route || 'hybrid'}`;
    const routeLabel = (route || 'HYBRID').toUpperCase();

    summaryEl.innerHTML = `
      <div class="o1-left">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="color: var(--accent-purple);">
          <circle cx="12" cy="12" r="10"></circle>
          <polyline points="12 6 12 12 16 14"></polyline>
        </svg>
        <span>Thought for ${seconds}s</span>
        <span class="route-tag ${routeClass}">${routeLabel}</span>
      </div>
      <svg class="chevron-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <polyline points="6 9 12 15 18 9"></polyline>
      </svg>
    `;

    if (!trace || trace.length === 0) {
      detailsEl.innerHTML = `<div style="color: var(--text-muted); font-size: 0.76rem;">Direct multi-hop traversal with router node classification.</div>`;
      return;
    }

    let html = '';
    trace.forEach((step, idx) => {
      const stepName = step.step || `Step ${idx + 1}`;
      const details = step.details || step.description || step.content || JSON.stringify(step);
      html += `
        <div class="trace-step">
          <span style="color: var(--openai-green); font-weight: 600;">▸</span>
          <div>
            <strong>${escapeHtml(stepName)}:</strong>
            <span style="color: var(--text-muted);">${escapeHtml(details)}</span>
          </div>
        </div>
      `;
    });
    detailsEl.innerHTML = html;
  }

  // ── 8. Markdown Parser & Code Formatting ───────────────────────────────────

  function formatMarkdown(md) {
    if (!md) return '';

    md = md.replace(/```([a-zA-Z0-9_\-\+]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      const language = lang || 'code';
      return `<pre><div style="display:flex; justify-content:space-between; align-items:center; color:var(--text-muted); font-size:0.75rem; margin-bottom:8px; border-bottom:1px solid rgba(125,125,125,0.15); padding-bottom:4px;"><span>${language}</span><button class="copy-code-btn" style="background:transparent; border:none; color:var(--text-secondary); cursor:pointer; font-size:0.73rem;">Copy code</button></div><code>${escapeHtml(code.trim())}</code></pre>`;
    });

    md = md.replace(/`([^`]+)`/g, '<code>$1</code>');
    md = md.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    md = md.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    md = md.replace(/^# (.*$)/gim, '<h1>$1</h1>');
    md = md.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    md = md.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    md = md.replace(/^\> (.*$)/gim, '<blockquote style="border-left:3px solid var(--openai-green); padding-left:12px; color:var(--text-secondary); margin:8px 0;">$1</blockquote>');
    md = md.replace(/^\- (.*$)/gim, '<li>$1</li>');
    md = md.replace(/(<li>.*<\/li>)/gim, '<ul>$1</ul>');

    md = md.split('\n\n').map(p => {
      p = p.trim();
      if (!p) return '';
      if (p.startsWith('<pre') || p.startsWith('<h') || p.startsWith('<ul') || p.startsWith('<blockquote')) {
        return p;
      }
      return `<p>${p.replace(/\n/g, '<br>')}</p>`;
    }).join('');

    return md;
  }

  function setupCodeCopyButtons(container) {
    container.querySelectorAll('.copy-code-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const pre = e.target.closest('pre');
        const code = pre.querySelector('code');
        if (code) {
          navigator.clipboard.writeText(code.innerText).then(() => {
            btn.textContent = 'Copied!';
            setTimeout(() => { btn.textContent = 'Copy code'; }, 2000);
          });
        }
      });
    });
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function scrollToBottom() {
    chatViewport.scrollTop = chatViewport.scrollHeight;
  }
});
