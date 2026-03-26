/**
 * DocuMind \u2013 frontend logic
 *
 * Flow:
 *  1. On load \u2192 POST /api/session \u2192 store sessionId
 *  2. User selects files \u2192 click \"Process & Embed\" \u2192 POST /api/upload/:id
 *     OR clicks \"Try Demo\" \u2192 POST /api/demo/:id
 *  3. User types question \u2192 POST /api/chat/:id/stream \u2192 SSE streaming answer
 */

"use strict";

/* \u2500\u2500 DOM refs \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
const dropZone        = document.getElementById("dropZone");
const fileInput       = document.getElementById("fileInput");
const uploadBtn       = document.getElementById("uploadBtn");
const newSessionBtn   = document.getElementById("newSessionBtn");
const fileListSec     = document.getElementById("fileListSection");
const fileList        = document.getElementById("fileList");
const chunkCount      = document.getElementById("chunkCount");
const messages        = document.getElementById("messages");
const inputForm       = document.getElementById("inputForm");
const questionInput   = document.getElementById("questionInput");
const sendBtn         = document.getElementById("sendBtn");
const sessionMeta     = document.getElementById("sessionMeta");
const overlay         = document.getElementById("overlay");
const overlayMsg      = document.getElementById("overlayMsg");
const pendingFilesEl  = document.getElementById("pendingFilesList");
const sidebarToggle   = document.getElementById("sidebarToggle");
const sidebar         = document.getElementById("sidebar");

/* \u2500\u2500 State \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
let sessionId    = null;
let pendingFiles = [];
let activeProvider = null;
let isStreaming = false;

/* \u2500\u2500 File type helpers \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
const FILE_ICONS = {
  ".pdf": "\ud83d\udcd5",
  ".docx": "\ud83d\udcd8",
  ".doc": "\ud83d\udcd8",
  ".txt": "\ud83d\udcdd",
  ".md": "\ud83d\udcd3",
  ".csv": "\ud83d\udcca",
};

function getFileIcon(filename) {
  const ext = "." + filename.split(".").pop().toLowerCase();
  return FILE_ICONS[ext] || "\ud83d\udcc4";
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

/* \u2500\u2500 Simple Markdown renderer \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
function escapeForHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderMarkdown(text) {
  const codeBlocks = [];
  let html = text.replace(/```(\w*)\n([\s\S]*?)```/g, function(_m, _lang, code) {
    const idx = codeBlocks.length;
    codeBlocks.push('<pre><code>' + escapeForHtml(code.trim()) + '</code></pre>');
    return '\x00CB' + idx + '\x00';
  });

  const inlineCodes = [];
  html = html.replace(/`([^`]+)`/g, function(_m, code) {
    const idx = inlineCodes.length;
    inlineCodes.push('<code>' + escapeForHtml(code) + '</code>');
    return '\x00IC' + idx + '\x00';
  });

  html = escapeForHtml(html);

  html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

  html = html.replace(/^[*\-+] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>\n?)+/g, function(match) {
    return '<ul>' + match + '</ul>';
  });

  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

  html = html.replace(/\n\n/g, '</p><p>');
  html = html.replace(/\n/g, '<br>');

  if (!html.startsWith('<')) {
    html = '<p>' + html + '</p>';
  }

  html = html.replace(/\x00CB(\d+)\x00/g, function(_m, idx) {
    return codeBlocks[parseInt(idx)];
  });
  html = html.replace(/\x00IC(\d+)\x00/g, function(_m, idx) {
    return inlineCodes[parseInt(idx)];
  });

  return html;
}

/* \u2500\u2500 Helpers \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */

function showOverlay(msg) {
  overlayMsg.textContent = msg || "Processing\u2026";
  overlay.classList.remove("hidden");
}

function hideOverlay() {
  overlay.classList.add("hidden");
}

function showWelcome() {
  const providerHint = activeProvider
    ? `<span class="welcome-provider">Powered by ${activeProvider === "gemini" ? "Google Gemini" : "OpenAI"}</span>`
    : "";
  messages.innerHTML = `
    <div class="welcome">
      <div class="welcome-icon">
        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="1.5">
          <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
          <polyline points="14 2 14 8 20 8"/>
          <line x1="16" y1="13" x2="8" y2="13"/>
          <line x1="16" y1="17" x2="8" y2="17"/>
          <polyline points="10 9 9 9 8 9"/>
        </svg>
      </div>
      <h2>Welcome to DocuMind</h2>
      <p>Upload your documents and start asking questions \u2014 or try the instant demo to see it in action. No API key needed.</p>
      ${providerHint}
      <div class="feature-grid">
        <div class="feature-card">
          <div class="feature-card-icon" style="background: rgba(16,185,129,0.1);">\ud83d\ude80</div>
          <h4>No Setup Required</h4>
          <p>Just upload a file and ask \u2014 no API keys, no accounts, nothing to configure</p>
        </div>
        <div class="feature-card">
          <div class="feature-card-icon" style="background: rgba(6,182,212,0.1);">\ud83d\udd0d</div>
          <h4>Smart Retrieval</h4>
          <p>FAISS vector search finds the most relevant passages in your docs</p>
        </div>
        <div class="feature-card">
          <div class="feature-card-icon" style="background: rgba(139,92,246,0.1);">\u26a1</div>
          <h4>Streaming Answers</h4>
          <p>Watch answers appear word-by-word with live pipeline visibility</p>
        </div>
        <div class="feature-card">
          <div class="feature-card-icon" style="background: rgba(16,185,129,0.1);">\ud83d\udcce</div>
          <h4>Source Citations</h4>
          <p>Every answer includes references to the exact source passages</p>
        </div>
      </div>
      <button class="btn btn-demo" id="demoBtn" onclick="loadDemo()">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        Try Instant Demo
      </button>
      <span class="demo-hint">Loads a sample document so you can chat immediately</span>
    </div>`;
}

function addMessage(role, html, sources) {
  const welcome = messages.querySelector(".welcome");
  if (welcome) welcome.remove();

  const div = document.createElement("div");
  div.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.textContent = role === "user" ? "\ud83d\udc64" : "\ud83e\udde0";

  const content = document.createElement("div");
  content.className = "msg-content";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (role === "assistant") {
    bubble.innerHTML = renderMarkdown(html);
  } else {
    bubble.textContent = html;
  }

  content.appendChild(bubble);

  if (role === "assistant" && sources && sources.length > 0) {
    content.appendChild(buildSourcesSection(sources));
  }

  div.appendChild(avatar);
  div.appendChild(content);
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

function buildSourcesSection(sources) {
  const srcSection = document.createElement("div");
  srcSection.className = "sources-section";

  const toggleBtn = document.createElement("button");
  toggleBtn.className = "sources-toggle";
  toggleBtn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg> ${sources.length} source${sources.length > 1 ? "s" : ""} referenced`;

  const srcList = document.createElement("ul");
  srcList.className = "sources-list";

  sources.forEach(src => {
    const li = document.createElement("li");
    li.className = "source-item";
    const truncated = src.snippet.length >= 150;
    li.innerHTML = `
      <div class="source-file">${getFileIcon(src.file)} ${escapeHtml(src.file)}</div>
      <div class="source-snippet">${escapeHtml(src.snippet)}${truncated ? "\u2026" : ""}</div>`;
    srcList.appendChild(li);
  });

  toggleBtn.addEventListener("click", () => {
    toggleBtn.classList.toggle("open");
    srcList.classList.toggle("show");
  });

  srcSection.appendChild(toggleBtn);
  srcSection.appendChild(srcList);
  return srcSection;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function addStreamingMessage() {
  const welcome = messages.querySelector(".welcome");
  if (welcome) welcome.remove();

  const div = document.createElement("div");
  div.className = "message assistant";
  div.id = "streamingMessage";

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.textContent = "\ud83e\udde0";

  const content = document.createElement("div");
  content.className = "msg-content";

  // Pipeline status indicator
  const pipeline = document.createElement("div");
  pipeline.className = "pipeline-status";
  pipeline.id = "pipelineStatus";

  const bubble = document.createElement("div");
  bubble.className = "bubble streaming-bubble";
  bubble.id = "streamingBubble";
  bubble.innerHTML = '<span class="stream-cursor"></span>';

  content.appendChild(pipeline);
  content.appendChild(bubble);
  div.appendChild(avatar);
  div.appendChild(content);
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return { div, content, bubble, pipeline };
}

function updatePipelineStatus(pipeline, message) {
  const step = document.createElement("div");
  step.className = "pipeline-step";
  step.innerHTML = `<span class="pipeline-dot"></span> ${escapeHtml(message)}`;
  pipeline.appendChild(step);

  // Animate previous steps to "completed"
  const steps = pipeline.querySelectorAll(".pipeline-step");
  steps.forEach((s, i) => {
    if (i < steps.length - 1) s.classList.add("done");
  });

  messages.scrollTop = messages.scrollHeight;
}

function finishStreamingMessage(content, fullText, sources) {
  const bubble = document.getElementById("streamingBubble");
  const pipeline = document.getElementById("pipelineStatus");

  if (bubble) {
    bubble.classList.remove("streaming-bubble");
    bubble.innerHTML = renderMarkdown(fullText);
  }

  if (pipeline) {
    // Mark all steps done and fade out
    pipeline.querySelectorAll(".pipeline-step").forEach(s => s.classList.add("done"));
    pipeline.classList.add("finished");
  }

  if (sources && sources.length > 0) {
    content.appendChild(buildSourcesSection(sources));
  }

  const msgDiv = document.getElementById("streamingMessage");
  if (msgDiv) msgDiv.removeAttribute("id");

  messages.scrollTop = messages.scrollHeight;
}

function updateFileList(names, docCount) {
  fileList.innerHTML = "";
  names.forEach(name => {
    const li = document.createElement("li");
    const icon = document.createElement("span");
    icon.className = "file-icon";
    icon.textContent = getFileIcon(name);
    li.appendChild(icon);
    li.appendChild(document.createTextNode(name));
    fileList.appendChild(li);
  });
  fileListSec.style.display = names.length ? "" : "none";
  if (docCount) {
    chunkCount.textContent = `${docCount} chunks indexed`;
  }
}

function showPendingFiles(files) {
  pendingFilesEl.innerHTML = "";
  if (files.length === 0) {
    pendingFilesEl.style.display = "none";
    return;
  }
  pendingFilesEl.style.display = "";
  files.forEach(f => {
    const div = document.createElement("div");
    div.className = "pending-file";
    div.innerHTML = `
      <span class="pending-file-icon">${getFileIcon(f.name)}</span>
      <span class="pending-file-name">${escapeHtml(f.name)}</span>
      <span class="pending-file-size">${formatBytes(f.size)}</span>`;
    pendingFilesEl.appendChild(div);
  });
}

function enableChat() {
  questionInput.disabled = false;
  sendBtn.disabled = false;
  questionInput.focus();
}

/* \u2500\u2500 Demo mode \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */

async function loadDemo() {
  if (!sessionId) return;

  const demoBtn = document.getElementById("demoBtn");
  if (demoBtn) {
    demoBtn.disabled = true;
    demoBtn.innerHTML = `<span class="btn-spinner"></span> Loading demo\u2026`;
  }

  showOverlay("Loading demo document\u2026");

  try {
    const res = await fetch(`/api/demo/${sessionId}`, { method: "POST" });
    const data = await res.json();

    if (!res.ok) {
      addMessage("assistant", `Demo failed: ${data.detail || "Unknown error"}`);
      return;
    }

    const info = data.session;
    updateFileList(info.file_names, info.document_count);
    enableChat();

    addMessage("assistant",
      `**Demo loaded!** I've ingested the DocuMind architecture document.\n\n` +
      `Try asking me:\n` +
      `- *"What tech stack does DocuMind use?"*\n` +
      `- *"How does the RAG pipeline work?"*\n` +
      `- *"What are the key features?"*\n\n` +
      `Or upload your own files from the sidebar.`
    );

    sidebar.classList.remove("open");
  } catch (err) {
    addMessage("assistant", `Network error: ${err.message}`);
  } finally {
    hideOverlay();
  }
}

// Expose to onclick in HTML
window.loadDemo = loadDemo;

/* \u2500\u2500 Session management \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */

async function initSession() {
  const res = await fetch("/api/session", { method: "POST" });
  if (!res.ok) throw new Error("Failed to create session");
  const data = await res.json();
  sessionId = data.session_id;
  sessionMeta.textContent = sessionId.slice(0, 8);

  try {
    const pRes = await fetch("/api/provider");
    if (pRes.ok) {
      const pData = await pRes.json();
      activeProvider = pData.provider;
      updateProviderBadge();
    }
  } catch (_) { /* cosmetic */ }
}

function updateProviderBadge() {
  const existing = document.getElementById("providerBadge");
  if (existing) existing.remove();
  if (!activeProvider) return;

  const badge = document.createElement("div");
  badge.id = "providerBadge";
  badge.className = "provider-badge";
  const label = activeProvider === "gemini" ? "Gemini (Free)" : "OpenAI";
  const icon  = activeProvider === "gemini" ? "\u2726" : "\u2b21";
  badge.innerHTML = `<span class="provider-dot ${activeProvider}"></span>${icon} Powered by ${label}`;
  sessionMeta.parentElement.insertBefore(badge, sessionMeta);
}

async function resetSession() {
  pendingFiles = [];
  fileInput.value = "";
  uploadBtn.disabled = true;
  showPendingFiles([]);
  updateFileList([]);
  questionInput.disabled = true;
  sendBtn.disabled = true;
  showWelcome();
  await initSession();
}

/* \u2500\u2500 File selection \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */

function handleFiles(files) {
  pendingFiles = Array.from(files);
  uploadBtn.disabled = pendingFiles.length === 0;
  showPendingFiles(pendingFiles);
}

dropZone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFiles(fileInput.files);
});

dropZone.addEventListener("dragover", e => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => {
  dropZone.classList.remove("drag-over");
});

dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
});

/* \u2500\u2500 Upload \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */

uploadBtn.addEventListener("click", async () => {
  if (!pendingFiles.length || !sessionId) return;

  showOverlay(`Embedding ${pendingFiles.length} file${pendingFiles.length > 1 ? "s" : ""}\u2026`);
  uploadBtn.disabled = true;

  const formData = new FormData();
  pendingFiles.forEach(f => formData.append("files", f));

  try {
    const res = await fetch(`/api/upload/${sessionId}`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (!res.ok) {
      addMessage("assistant", `Upload failed: ${data.detail || "Unknown error"}`);
      return;
    }

    const names = data.session.file_names;
    updateFileList(names, data.session.document_count);
    enableChat();

    const fileNames = data.uploaded.map(u => u.filename).join(", ");
    addMessage("assistant",
      `**${data.uploaded.length} file${data.uploaded.length > 1 ? "s" : ""} processed successfully!**\n\n` +
      `Files: ${fileNames}\n` +
      `Chunks indexed: **${data.session.document_count}**\n\n` +
      `Your knowledge base is ready. Ask me anything about ${names.length > 1 ? "these documents" : "this document"}.`
    );

    pendingFiles = [];
    fileInput.value = "";
    showPendingFiles([]);
    sidebar.classList.remove("open");

  } catch (err) {
    addMessage("assistant", `Network error during upload: ${err.message}`);
  } finally {
    hideOverlay();
    uploadBtn.disabled = pendingFiles.length === 0;
  }
});

/* \u2500\u2500 Chat (streaming SSE) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */

async function sendQuestion(question) {
  if (isStreaming) return;
  isStreaming = true;
  sendBtn.disabled = true;
  questionInput.disabled = true;

  addMessage("user", question);

  const { div: msgDiv, content, bubble, pipeline } = addStreamingMessage();

  let fullText = "";
  let sources = [];

  try {
    const res = await fetch(`/api/chat/${sessionId}/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      const err = await res.json();
      msgDiv.remove();
      addMessage("assistant", err.detail || "Something went wrong.");
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const jsonStr = line.slice(6).trim();
        if (!jsonStr) continue;

        let event;
        try { event = JSON.parse(jsonStr); } catch { continue; }

        switch (event.type) {
          case "status":
            updatePipelineStatus(pipeline, event.message);
            break;

          case "sources":
            sources = event.sources || [];
            break;

          case "token":
            fullText += event.token;
            // Live render \u2014 re-render markdown every few tokens for smoothness
            bubble.innerHTML = renderMarkdown(fullText) + '<span class="stream-cursor"></span>';
            messages.scrollTop = messages.scrollHeight;
            break;

          case "done":
            finishStreamingMessage(content, fullText, sources);
            break;

          case "error":
            msgDiv.remove();
            addMessage("assistant", event.message || "An error occurred.");
            break;
        }
      }
    }

    // Safety: if stream ended without "done" event
    if (fullText && !document.querySelector(".streaming-bubble")) {
      // Already finalized
    } else if (fullText) {
      finishStreamingMessage(content, fullText, sources);
    }

  } catch (err) {
    msgDiv.remove();
    addMessage("assistant", `Network error: ${err.message}`);
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    questionInput.disabled = false;
    questionInput.focus();
  }
}

inputForm.addEventListener("submit", async e => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question || !sessionId) return;
  questionInput.value = "";
  questionInput.style.height = "auto";
  await sendQuestion(question);
});

/* Auto-resize textarea */
questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 160)}px`;
});

/* Submit on Ctrl+Enter / Cmd+Enter */
questionInput.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    inputForm.dispatchEvent(new Event("submit", { cancelable: true }));
  }
});

/* \u2500\u2500 New session button \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
newSessionBtn.addEventListener("click", async () => {
  if (!confirm("Start a new session? Current conversation will be cleared.")) return;
  await resetSession();
});

/* \u2500\u2500 Mobile sidebar toggle \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
sidebarToggle.addEventListener("click", () => {
  sidebar.classList.toggle("open");
});

document.addEventListener("click", e => {
  if (
    sidebar.classList.contains("open") &&
    !sidebar.contains(e.target) &&
    !sidebarToggle.contains(e.target)
  ) {
    sidebar.classList.remove("open");
  }
});

/* \u2500\u2500 Bootstrap \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 */
(async () => {
  showWelcome();
  try {
    await initSession();
    if (activeProvider) showWelcome();
  } catch (err) {
    addMessage("assistant", `Could not connect to the server: ${err.message}`);
  }
})();
