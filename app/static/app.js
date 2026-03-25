/**
 * DocuMind – frontend logic
 *
 * Flow:
 *  1. On load → POST /api/session → store sessionId
 *  2. User selects files → click "Process & Embed" → POST /api/upload/:id
 *  3. User types question → POST /api/chat/:id → display answer with sources
 */

"use strict";

/* ── DOM refs ─────────────────────────────────────────────────────────── */
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

/* ── State ────────────────────────────────────────────────────────────── */
let sessionId    = null;
let pendingFiles = [];

/* ── File type helpers ────────────────────────────────────────────────── */
const FILE_ICONS = {
  ".pdf": "📕",
  ".docx": "📘",
  ".doc": "📘",
  ".txt": "📝",
  ".md": "📓",
  ".csv": "📊",
};

function getFileIcon(filename) {
  const ext = "." + filename.split(".").pop().toLowerCase();
  return FILE_ICONS[ext] || "📄";
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

/* ── Simple Markdown renderer ─────────────────────────────────────────── */
function renderMarkdown(text) {
  let html = text;

  // Escape HTML
  html = html.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  // Code blocks (``` ... ```)
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, function(_match, _lang, code) {
    return '<pre><code>' + code.trim() + '</code></pre>';
  });

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Headers
  html = html.replace(/^#### (.+)$/gm, '<h4>$1</h4>');
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Bold and italic
  html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Blockquotes
  html = html.replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>');

  // Unordered lists
  html = html.replace(/^[*\-+] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>\n?)+/g, function(match) {
    return '<ul>' + match + '</ul>';
  });

  // Ordered lists
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

  // Links
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

  // Paragraphs (double newline)
  html = html.replace(/\n\n/g, '</p><p>');

  // Single newlines
  html = html.replace(/\n/g, '<br>');

  // Wrap in paragraph if not already wrapped
  if (!html.startsWith('<')) {
    html = '<p>' + html + '</p>';
  }

  return html;
}

/* ── Helpers ──────────────────────────────────────────────────────────── */

function showOverlay(msg) {
  overlayMsg.textContent = msg || "Processing…";
  overlay.classList.remove("hidden");
}

function hideOverlay() {
  overlay.classList.add("hidden");
}

function showWelcome() {
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
      <p>Upload your documents and start asking questions. Your files are analyzed using RAG (Retrieval-Augmented Generation) for precise, context-aware answers.</p>
      <div class="feature-grid">
        <div class="feature-card">
          <div class="feature-card-icon" style="background: rgba(6,182,212,0.1);">🔍</div>
          <h4>Smart Retrieval</h4>
          <p>FAISS vector search finds the most relevant passages in your docs</p>
        </div>
        <div class="feature-card">
          <div class="feature-card-icon" style="background: rgba(139,92,246,0.1);">🧠</div>
          <h4>Context-Aware</h4>
          <p>Conversation memory enables follow-up questions naturally</p>
        </div>
        <div class="feature-card">
          <div class="feature-card-icon" style="background: rgba(16,185,129,0.1);">📎</div>
          <h4>Source Citations</h4>
          <p>Every answer includes references to the exact source passages</p>
        </div>
      </div>
    </div>`;
}

function addMessage(role, html, sources) {
  // Remove welcome screen if present
  const welcome = messages.querySelector(".welcome");
  if (welcome) welcome.remove();

  const div = document.createElement("div");
  div.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.textContent = role === "user" ? "👤" : "🧠";

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

  // Add source citations for assistant messages
  if (role === "assistant" && sources && sources.length > 0) {
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
      li.innerHTML = `
        <div class="source-file">${getFileIcon(src.file)} ${escapeHtml(src.file)}</div>
        <div class="source-snippet">${escapeHtml(src.snippet)}…</div>`;
      srcList.appendChild(li);
    });

    toggleBtn.addEventListener("click", () => {
      toggleBtn.classList.toggle("open");
      srcList.classList.toggle("show");
    });

    srcSection.appendChild(toggleBtn);
    srcSection.appendChild(srcList);
    content.appendChild(srcSection);
  }

  div.appendChild(avatar);
  div.appendChild(content);
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function addTypingIndicator() {
  const welcome = messages.querySelector(".welcome");
  if (welcome) welcome.remove();

  const div = document.createElement("div");
  div.className = "message assistant typing";
  div.id = "typingIndicator";

  const avatar = document.createElement("div");
  avatar.className = "msg-avatar";
  avatar.textContent = "🧠";

  const content = document.createElement("div");
  content.className = "msg-content";
  content.innerHTML = `<div class="bubble">
    <span class="dot"></span><span class="dot"></span><span class="dot"></span>
  </div>`;

  div.appendChild(avatar);
  div.appendChild(content);
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

function removeTypingIndicator() {
  const el = document.getElementById("typingIndicator");
  if (el) el.remove();
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

/* ── Session management ──────────────────────────────────────────────── */

async function initSession() {
  const res = await fetch("/api/session", { method: "POST" });
  if (!res.ok) throw new Error("Failed to create session");
  const data = await res.json();
  sessionId = data.session_id;
  sessionMeta.textContent = sessionId.slice(0, 8);
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

/* ── File selection ───────────────────────────────────────────────────── */

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

/* ── Upload ───────────────────────────────────────────────────────────── */

uploadBtn.addEventListener("click", async () => {
  if (!pendingFiles.length || !sessionId) return;

  showOverlay(`Embedding ${pendingFiles.length} file${pendingFiles.length > 1 ? "s" : ""}…`);
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

    // Reset pending
    pendingFiles = [];
    fileInput.value = "";
    showPendingFiles([]);

    // Close sidebar on mobile
    sidebar.classList.remove("open");

  } catch (err) {
    addMessage("assistant", `Network error during upload: ${err.message}`);
  } finally {
    hideOverlay();
    uploadBtn.disabled = pendingFiles.length === 0;
  }
});

/* ── Chat ─────────────────────────────────────────────────────────────── */

inputForm.addEventListener("submit", async e => {
  e.preventDefault();

  const question = questionInput.value.trim();
  if (!question || !sessionId) return;

  addMessage("user", question);
  questionInput.value = "";
  questionInput.style.height = "auto";

  sendBtn.disabled = true;
  questionInput.disabled = true;
  addTypingIndicator();

  try {
    const res = await fetch(`/api/chat/${sessionId}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await res.json();
    removeTypingIndicator();

    if (!res.ok) {
      addMessage("assistant", data.detail || "Something went wrong.");
    } else {
      addMessage("assistant", data.answer, data.sources);
    }

  } catch (err) {
    removeTypingIndicator();
    addMessage("assistant", `Network error: ${err.message}`);
  } finally {
    sendBtn.disabled = false;
    questionInput.disabled = false;
    questionInput.focus();
  }
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

/* ── New session button ──────────────────────────────────────────────── */
newSessionBtn.addEventListener("click", async () => {
  if (!confirm("Start a new session? Current conversation will be cleared.")) return;
  await resetSession();
});

/* ── Mobile sidebar toggle ───────────────────────────────────────────── */
sidebarToggle.addEventListener("click", () => {
  sidebar.classList.toggle("open");
});

// Close sidebar on outside click (mobile)
document.addEventListener("click", e => {
  if (
    sidebar.classList.contains("open") &&
    !sidebar.contains(e.target) &&
    !sidebarToggle.contains(e.target)
  ) {
    sidebar.classList.remove("open");
  }
});

/* ── Bootstrap ────────────────────────────────────────────────────────── */
(async () => {
  showWelcome();
  try {
    await initSession();
  } catch (err) {
    addMessage("assistant", `Could not connect to the server: ${err.message}`);
  }
})();
