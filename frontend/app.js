/**
 * Local Agentic GraphRAG — ChatGPT / Codex Web Interface Controller
 * 100% Local, Direct REST Connection to FastAPI backend
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const sidebar = document.getElementById('sidebar');
  const toggleSidebarBtn = document.getElementById('toggleSidebarBtn');
  const newChatBtn = document.getElementById('newChatBtn');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('fileInput');
  const attachBtn = document.getElementById('attachBtn');
  const progressCard = document.getElementById('progressCard');
  const progressStage = document.getElementById('progressStage');
  const progressPercent = document.getElementById('progressPercent');
  const progressBarFill = document.getElementById('progressBarFill');
  const progressDesc = document.getElementById('progressDesc');

  // Telemetry Elements
  const lmDot = document.getElementById('lmDot');
  const lmText = document.getElementById('lmText');
  const llmModelTag = document.getElementById('llmModelTag');
  const embedModelTag = document.getElementById('embedModelTag');
  const graphStats = document.getElementById('graphStats');
  const chromaStatus = document.getElementById('chromaStatus');
  const quickStats = document.getElementById('quickStats');

  // Chat Elements
  const chatViewport = document.getElementById('chatViewport');
  const heroView = document.getElementById('heroView');
  const messagesList = document.getElementById('messagesList');
  const queryInput = document.getElementById('queryInput');
  const sendBtn = document.getElementById('sendBtn');
  const promptCards = document.querySelectorAll('.prompt-card');

  let isGenerating = false;
  let statusPollInterval = null;

  // ── 1. Telemetry & Health Checks ──────────────────────────────────────────

  async function fetchHealth() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error('Health check failed');
      const data = await res.json();

      const lm = data.lm_studio || {};
      const db = data.database || {};

      if (lm.running) {
        lmDot.className = 'dot online';
        lmText.textContent = 'Connected (1234)';
        llmModelTag.textContent = lm.configured_llm_model ? lm.configured_llm_model.split('/').pop() : 'gemma-4-e4b';
        embedModelTag.textContent = lm.configured_embed_model ? `${lm.configured_embed_model.split('/').pop()} (${lm.embedding_dim || 768}d)` : 'embeddinggemma-2';
      } else {
        lmDot.className = 'dot offline';
        lmText.textContent = 'Disconnected';
      }

      const nodes = db.graph_nodes || 0;
      const edges = db.graph_edges || 0;
      graphStats.textContent = `${nodes} nodes • ${edges} edges`;
      quickStats.textContent = `${nodes} Nodes • ${edges} Edges`;

      chromaStatus.textContent = db.chromadb_exists ? 'Indexed (Ready)' : 'Empty';
    } catch (err) {
      lmDot.className = 'dot offline';
      lmText.textContent = 'Offline';
    }
  }

  fetchHealth();
  setInterval(fetchHealth, 10000);

  // ── 2. Sidebar Navigation ─────────────────────────────────────────────────

  toggleSidebarBtn.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
  });

  newChatBtn.addEventListener('click', () => {
    messagesList.innerHTML = '';
    messagesList.style.display = 'none';
    heroView.style.display = 'flex';
    queryInput.value = '';
    adjustTextareaHeight();
    updateSendBtnState();
    queryInput.focus();
  });

  // Shortcut Ctrl+B for sidebar, Ctrl+K for new chat
  window.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') {
      e.preventDefault();
      sidebar.classList.toggle('collapsed');
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      newChatBtn.click();
    }
  });

  // ── 3. File Upload & Ingestion ─────────────────────────────────────────────

  dropzone.addEventListener('click', () => fileInput.click());
  attachBtn.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) uploadFiles(files);
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      uploadFiles(fileInput.files);
    }
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
      progressDesc.textContent = `Uploading ${files.length} file(s) to server...`;

      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
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
            progressDesc.textContent = 'Knowledge Graph & Vector Embeddings updated!';
            fetchHealth();
          }

          setTimeout(() => {
            progressCard.classList.remove('active');
            progressBarFill.style.background = '';
          }, 4000);
        }
      } catch (err) {
        // Silent catch for poll error
      }
    }, 800);
  }

  // ── 4. Chat Composition & Sending ──────────────────────────────────────────

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

  // ── 5. Query Handling & Reasoning Trace ────────────────────────────────────

  async function handleUserSubmit(query) {
    if (!query || isGenerating) return;

    // Reset input
    queryInput.value = '';
    adjustTextareaHeight();
    updateSendBtnState();

    // Show conversation canvas
    heroView.style.display = 'none';
    messagesList.style.display = 'flex';

    // Render User Message
    appendUserMessage(query);

    // Prepare Assistant Placeholder with o1 Thinking Accordion
    const { assistantRow, thinkingSummary, thinkingDetails, bodyEl } = createAssistantRow();
    messagesList.appendChild(assistantRow);
    scrollToBottom();

    // Start timer for o1 thinking
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

      // Finalize o1 Thinking Accordion
      renderThinkingTrace(thinkingSummary, thinkingDetails, data.route, data.reasoning_trace, totalSeconds);

      // Render Markdown Response
      bodyEl.innerHTML = formatMarkdown(data.response || 'No response returned.');
      setupCodeCopyButtons(bodyEl);

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

  function appendUserMessage(text) {
    const row = document.createElement('div');
    row.className = 'message-row user';
    row.innerHTML = `<div class="user-bubble">${escapeHtml(text)}</div>`;
    messagesList.appendChild(row);
  }

  function createAssistantRow() {
    const assistantRow = document.createElement('div');
    assistantRow.className = 'message-row assistant';

    const avatar = document.createElement('div');
    avatar.className = 'assistant-avatar';
    avatar.innerHTML = `
      <svg width="20" height="20" viewBox="0 0 21 21" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M11.6475 18.3409C11.0975 18.3409 10.575 18.2364 10.08 18.0274C9.58502 17.8184 9.14502 17.5269 8.76002 17.1529C8.34202 17.2959 7.90751 17.3674 7.45651 17.3674C6.71951 17.3674 6.03751 17.1859 5.41051 16.8229C4.78351 16.4599 4.27751 15.9649 3.89251 15.3379C3.51851 14.7109 3.33151 14.0124 3.33151 13.2424C3.33151 12.9234 3.37551 12.5769 3.46351 12.2029C3.02351 11.7959 2.68251 11.3284 2.44051 10.8004C2.19851 10.2614 2.07751 9.70044 2.07751 9.11744C2.07751 8.52344 2.20401 7.95144 2.45701 7.40144C2.71001 6.85144 3.06201 6.37844 3.51301 5.98244C3.97501 5.57544 4.50851 5.29494 5.11351 5.14094C5.23451 4.51394 5.48751 3.95294 5.87251 3.45794C6.26851 2.95194 6.75252 2.55594 7.32452 2.26994C7.89652 1.98394 8.50702 1.84094 9.15602 1.84094C9.70602 1.84094 10.2285 1.94544 10.7235 2.15444C11.2185 2.36344 11.6585 2.65494 12.0435 3.02894C12.4615 2.88594 12.896 2.81444 13.347 2.81444C14.084 2.81444 14.766 2.99594 15.393 3.35894C16.02 3.72194 16.5205 4.21694 16.8945 4.84394C17.2795 5.47094 17.472 6.16944 17.472 6.93944C17.472 7.25844 17.428 7.60494 17.34 7.97894C17.78 8.38594 18.121 8.85894 18.363 9.39794C18.605 9.92594 18.726 10.4814 18.726 11.0644C18.726 11.6584 18.5995 12.2304 18.3465 12.7804C18.0935 13.3304 17.736 13.8089 17.274 14.2159C16.823 14.6119 16.295 14.8869 15.69 15.0409C15.569 15.6679 15.3105 16.2289 14.9145 16.7239C14.5295 17.2299 14.051 17.6259 13.479 17.9119C12.907 18.1979 12.2965 18.3409 11.6475 18.3409Z" fill="currentColor"/>
      </svg>
    `;

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
        <span>Thought for ${seconds} seconds</span>
        <span class="route-tag ${routeClass}">${routeLabel}</span>
      </div>
      <svg class="chevron-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <polyline points="6 9 12 15 18 9"></polyline>
      </svg>
    `;

    if (!trace || trace.length === 0) {
      detailsEl.innerHTML = `<div style="color: var(--text-muted);">Direct synthesis executed with router classification.</div>`;
      return;
    }

    let html = '';
    trace.forEach((step, idx) => {
      const stepName = step.step || `Step ${idx + 1}`;
      const details = step.details || step.content || JSON.stringify(step);
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

  // ── 6. Markdown Parser & Code Formatting ───────────────────────────────────

  function formatMarkdown(md) {
    if (!md) return '';

    // Code blocks with syntax badge and copy button
    md = md.replace(/```([a-zA-Z0-9_\-\+]*)\n([\s\S]*?)```/g, (match, lang, code) => {
      const language = lang || 'code';
      return `<pre><div style="display:flex; justify-content:space-between; align-items:center; color:var(--text-muted); font-size:0.75rem; margin-bottom:8px; border-bottom:1px solid rgba(255,255,255,0.08); padding-bottom:4px;"><span>${language}</span><button class="copy-code-btn" style="background:transparent; border:none; color:var(--text-secondary); cursor:pointer; font-size:0.73rem;">Copy code</button></div><code>${escapeHtml(code.trim())}</code></pre>`;
    });

    // Inline code
    md = md.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Headers
    md = md.replace(/^### (.*$)/gim, '<h3>$1</h3>');
    md = md.replace(/^## (.*$)/gim, '<h2>$1</h2>');
    md = md.replace(/^# (.*$)/gim, '<h1>$1</h1>');

    // Bold & Italics
    md = md.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    md = md.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    // Blockquotes
    md = md.replace(/^\> (.*$)/gim, '<blockquote style="border-left:3px solid var(--openai-green); padding-left:12px; color:var(--text-secondary); margin:8px 0;">$1</blockquote>');

    // Bullet lists
    md = md.replace(/^\- (.*$)/gim, '<li>$1</li>');
    md = md.replace(/(<li>.*<\/li>)/gim, '<ul>$1</ul>');

    // Paragraphs
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
