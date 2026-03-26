/**
 * QuillShift - frontend logic
 *
 * Flow:
 *  1. On load -> POST /api/session -> store sessionId
 *  2. User selects files -> click "Add to knowledge base" -> POST /api/upload/:id
 *     OR clicks "Try Demo" -> POST /api/demo/:id
 *  3. User types question -> POST /api/chat/:id/stream -> SSE streaming answer
 */

"use strict";

/* -- DOM refs ----------------------------------------------------------------- */
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
const sidebarScrim    = document.getElementById("sidebarScrim");
const sidebar         = document.getElementById("sidebar");
const toastContainer  = document.getElementById("toastContainer");
const scrollBottomBtn = document.getElementById("scrollBottomBtn");
const themeToggle     = document.getElementById("themeToggle");
const docPreviewPane  = document.getElementById("docPreviewPane");
const docPreviewClose = document.getElementById("docPreviewClose");
const docPreviewFilename = document.getElementById("docPreviewFilename");
const docPreviewMeta  = document.getElementById("docPreviewMeta");
const docPreviewContent = document.getElementById("docPreviewContent");

/* -- State -------------------------------------------------------------------- */
let sessionId    = null;
let pendingFiles = [];
let activeProvider = null;
let isStreaming = false;
let activePreviewFile = null;

/* -- Theme -------------------------------------------------------------------- */
function getPreferredTheme() {
  const stored = localStorage.getItem("theme");
  if (stored === "light" || stored === "dark") return stored;
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

function applyTheme(theme) {
  if (theme === "light") {
    document.documentElement.setAttribute("data-theme", "light");
  } else {
    document.documentElement.removeAttribute("data-theme");
  }
  // Update meta theme-color for mobile browser chrome
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.content = theme === "light" ? "#ffffff" : "#111110";
}

applyTheme(getPreferredTheme());

themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
  const next = current === "light" ? "dark" : "light";
  localStorage.setItem("theme", next);
  applyTheme(next);
});

// Follow OS changes if user hasn't explicitly chosen
window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", e => {
  if (!localStorage.getItem("theme")) {
    applyTheme(e.matches ? "light" : "dark");
  }
});

/* -- File type helpers -------------------------------------------------------- */
const FILE_ICON_SVG = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>`;

function getFileIcon(_filename) {
  return FILE_ICON_SVG;
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

/* -- Simple Markdown renderer ------------------------------------------------- */
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

/* -- Helpers ------------------------------------------------------------------ */

function showToast(message, type = "info") {
  const icons = { error: "\u2715", success: "\u2713", info: "->" };
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span class="toast-icon">${icons[type]}</span><span class="toast-text">${escapeHtml(message)}</span>`;
  toastContainer.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

function getTimestamp() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function copyToClipboard(text, btn) {
  navigator.clipboard.writeText(text).then(() => {
    const prev = btn.innerHTML;
    btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Copied`;
    btn.classList.add("copied");
    showToast("Copied to clipboard", "success");
    setTimeout(() => { btn.innerHTML = prev; btn.classList.remove("copied"); }, 2000);
  });
}

function addCopyBtnToMessage(content, textFn) {
  const btn = document.createElement("button");
  btn.className = "copy-msg-btn";
  btn.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg> Copy`;
  btn.addEventListener("click", () => copyToClipboard(textFn(), btn));
  content.appendChild(btn);
}

function addCopyBtnToCodeBlocks(bubble) {
  bubble.querySelectorAll("pre").forEach(pre => {
    const btn = document.createElement("button");
    btn.className = "copy-code-btn";
    btn.innerHTML = `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg> Copy`;
    btn.addEventListener("click", () => copyToClipboard(pre.textContent, btn));
    pre.style.position = "relative";
    pre.appendChild(btn);
  });
}

function showOverlay(msg) {
  overlayMsg.textContent = msg || "Processing\u2026";
  overlay.classList.remove("hidden");
}

function hideOverlay() {
  overlay.classList.add("hidden");
}

function showWelcome() {
  messages.innerHTML = `
    <div class="welcome">
      <svg class="welcome-icon" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
        <polyline points="14 2 14 8 20 8"/>
        <line x1="16" y1="13" x2="8" y2="13"/>
        <line x1="16" y1="17" x2="8" y2="17"/>
      </svg>
      <h2>QuillShift</h2>
      <p class="welcome-description">Upload a document and ask questions about it.<br>Your files are processed privately and cleared automatically.</p>
      <button class="welcome-demo-link" id="demoBtn" onclick="loadDemo()">Try with a sample document</button>
      <div class="welcome-divider"></div>
      <div class="welcome-examples">
        <span>"What are the key findings in this report?"</span>
        <span>"Summarize section 3 in plain language"</span>
        <span>"What data supports the main conclusion?"</span>
      </div>
    </div>`;
}

function addMessage(role, html, sources) {
  const welcome = messages.querySelector(".welcome");
  if (welcome) welcome.remove();

  const div = document.createElement("div");
  div.className = `message ${role}`;

  const content = document.createElement("div");
  content.className = "msg-content";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  if (role === "assistant") {
    const avatar = document.createElement("div");
    avatar.className = "msg-avatar";
    avatar.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>`;
    div.appendChild(avatar);

    bubble.innerHTML = renderMarkdown(html);
    addCopyBtnToCodeBlocks(bubble);
  } else {
    bubble.textContent = html;
  }

  content.appendChild(bubble);

  // Timestamp
  const time = document.createElement("div");
  time.className = "msg-time";
  time.textContent = getTimestamp();
  content.appendChild(time);

  if (role === "assistant" && sources && sources.length > 0) {
    content.appendChild(buildSourcesSection(sources));
  }

  // Copy button for assistant messages
  if (role === "assistant") {
    addCopyBtnToMessage(content, () => bubble.textContent);
  }

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
  toggleBtn.innerHTML = `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg> ${sources.length} source${sources.length > 1 ? "s" : ""}`;

  const srcList = document.createElement("ul");
  srcList.className = "sources-list";

  sources.forEach(src => {
    const li = document.createElement("li");
    li.className = "source-item";
    const truncated = src.snippet.length >= 150;
    li.innerHTML = `
      <div class="source-file">${escapeHtml(src.file)}</div>
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
  avatar.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>`;

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
    addCopyBtnToCodeBlocks(bubble);
  }

  if (pipeline) {
    // Mark all steps done and fade out
    pipeline.querySelectorAll(".pipeline-step").forEach(s => s.classList.add("done"));
    pipeline.classList.add("finished");
  }

  // Timestamp
  const time = document.createElement("div");
  time.className = "msg-time";
  time.textContent = getTimestamp();
  content.appendChild(time);

  if (sources && sources.length > 0) {
    content.appendChild(buildSourcesSection(sources));
  }

  // Copy button
  addCopyBtnToMessage(content, () => bubble ? bubble.textContent : fullText);

  const msgDiv = document.getElementById("streamingMessage");
  if (msgDiv) msgDiv.removeAttribute("id");

  messages.scrollTop = messages.scrollHeight;

  // Fetch follow-up suggestion chips (non-blocking)
  loadFollowupChips(content);
}

async function loadFollowupChips(content) {
  if (!sessionId) return;
  try {
    const res = await fetch(`/api/followups/${sessionId}`);
    if (!res.ok) return;
    const data = await res.json();
    const followups = data.followups || [];
    if (!followups.length) return;

    const chips = document.createElement("div");
    chips.className = "suggestion-chips followup-chips";
    followups.forEach(q => {
      const chip = document.createElement("button");
      chip.className = "suggestion-chip";
      chip.textContent = q;
      chip.addEventListener("click", () => {
        chips.remove();
        sendQuestion(q);
      });
      chips.appendChild(chip);
    });
    content.appendChild(chips);
    messages.scrollTop = messages.scrollHeight;
  } catch (_) { /* non-critical */ }
}

function updateFileList(names, docCount) {
  fileList.innerHTML = "";
  names.forEach(name => {
    const li = document.createElement("li");
    const icon = document.createElement("span");
    icon.className = "file-icon";
    icon.innerHTML = getFileIcon(name);
    li.appendChild(icon);
    li.appendChild(document.createTextNode(name));
    li.addEventListener("click", () => openDocPreview(name));
    fileList.appendChild(li);
  });
  fileListSec.style.display = names.length ? "" : "none";
  if (docCount) {
    chunkCount.textContent = `${docCount} passages ready`;
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
      <span class="pending-file-icon">${FILE_ICON_SVG}</span>
      <span class="pending-file-name">${escapeHtml(f.name)}</span>
      <span class="pending-file-size">${formatBytes(f.size)}</span>`;
    pendingFilesEl.appendChild(div);
  });
}

function isMobile() {
  return window.matchMedia("(max-width: 768px)").matches || "ontouchstart" in window;
}

function enableChat() {
  questionInput.disabled = false;
  sendBtn.disabled = false;
  if (!isMobile()) questionInput.focus();
}

/* -- Document preview pane ---------------------------------------------------- */

async function openDocPreview(filename) {
  // Toggle off if same file
  if (activePreviewFile === filename && docPreviewPane.classList.contains("open")) {
    closeDocPreview();
    return;
  }

  activePreviewFile = filename;
  docPreviewFilename.textContent = filename;

  // Highlight active file in sidebar
  fileList.querySelectorAll("li").forEach(li => {
    li.classList.toggle("active", li.textContent === filename);
  });

  // Show loading state
  docPreviewContent.innerHTML = '<div class="doc-preview-loading"><div class="spinner"></div> Loading...</div>';
  docPreviewPane.classList.add("open");

  try {
    const res = await fetch(`/api/document/${sessionId}/${encodeURIComponent(filename)}`);
    if (!res.ok) {
      const err = await res.json();
      docPreviewContent.textContent = err.detail || "Could not load document.";
      return;
    }
    const data = await res.json();
    const text = data.text || "";

    // Show character count in meta
    docPreviewMeta.textContent = `${text.length.toLocaleString()} characters extracted \u00b7 This is what the AI sees`;
    docPreviewMeta.classList.add("visible");

    docPreviewContent.textContent = text;
  } catch (err) {
    docPreviewContent.textContent = `Failed to load: ${err.message}`;
  }
}

function closeDocPreview() {
  docPreviewPane.classList.remove("open");
  activePreviewFile = null;
  docPreviewMeta.classList.remove("visible");
  fileList.querySelectorAll("li").forEach(li => li.classList.remove("active"));
}

docPreviewClose.addEventListener("click", closeDocPreview);

// Also expose openDocPreview for file badges in chat messages
window.openDocPreview = openDocPreview;

/* -- File badge helper (for showing attachments in chat messages) ------------- */

function buildFileBadges(filenames) {
  const container = document.createElement("div");
  container.className = "msg-attachments";
  filenames.forEach(name => {
    const badge = document.createElement("button");
    badge.className = "file-badge";
    badge.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>${escapeHtml(name)}`;
    badge.addEventListener("click", () => openDocPreview(name));
    container.appendChild(badge);
  });
  return container;
}

/* -- Demo mode ---------------------------------------------------------------- */

async function loadDemo() {
  if (!sessionId) return;

  const demoBtn = document.getElementById("demoBtn");
  if (demoBtn) {
    demoBtn.disabled = true;
    demoBtn.textContent = "Loading\u2026";
    demoBtn.style.opacity = "0.5";
  }

  showOverlay("Loading demo document\u2026");

  try {
    const res = await fetch(`/api/demo/${sessionId}`, { method: "POST" });
    const data = await res.json();

    if (!res.ok) {
      addMessage("assistant", `Demo failed: ${data.detail || "Unknown error"}`);
      showToast(data.detail || "Demo failed", "error");
      return;
    }

    const info = data.session;
    updateFileList(info.file_names, info.document_count);
    enableChat();

    const msgDiv = addMessage("assistant",
      `**Demo loaded.** I've added a sample document about QuillShift's architecture.\n\n` +
      `Try a suggestion below, or ask your own question.`
    );

    // Add clickable file badge
    const demoContent = msgDiv.querySelector(".msg-content");
    if (demoContent) {
      const demoBubble = demoContent.querySelector(".bubble");
      if (demoBubble) {
        demoBubble.insertBefore(buildFileBadges(info.file_names), demoBubble.firstChild);
      }
    }

    // Fetch AI-generated suggestion chips
    loadSuggestionChips(msgDiv);

    closeSidebar();
  } catch (err) {
    addMessage("assistant", `Network error: ${err.message}`);
  } finally {
    hideOverlay();
  }
}

// Expose to onclick in HTML
window.loadDemo = loadDemo;

async function loadSuggestionChips(msgDiv) {
  if (!sessionId) return;
  try {
    const res = await fetch(`/api/suggestions/${sessionId}`);
    if (!res.ok) return;
    const data = await res.json();
    const suggestions = data.suggestions || [];
    if (!suggestions.length) return;

    const content = msgDiv.querySelector(".msg-content");
    if (!content) return;

    const chips = document.createElement("div");
    chips.className = "suggestion-chips";
    suggestions.forEach(q => {
      const chip = document.createElement("button");
      chip.className = "suggestion-chip";
      chip.textContent = q;
      chip.addEventListener("click", () => {
        chips.remove();
        sendQuestion(q);
      });
      chips.appendChild(chip);
    });
    content.appendChild(chips);
  } catch (_) { /* non-critical */ }
}

/* -- Session management ------------------------------------------------------- */

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
  // Provider badge removed -- technology details belong in README, not the product UI
}

async function resetSession() {
  pendingFiles = [];
  fileInput.value = "";
  uploadBtn.disabled = true;
  showPendingFiles([]);
  updateFileList([]);
  questionInput.disabled = true;
  sendBtn.disabled = true;
  closeDocPreview();
  showWelcome();
  await initSession();
}

/* -- File selection ----------------------------------------------------------- */

function handleFiles(files) {
  pendingFiles = Array.from(files);
  uploadBtn.disabled = pendingFiles.length === 0;
  showPendingFiles(pendingFiles);
}

// dropZone is a <label> wrapping the file input, so clicking it
// already opens the file picker natively — no extra click() needed.

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

/* -- Upload ------------------------------------------------------------------- */

uploadBtn.addEventListener("click", async () => {
  if (!pendingFiles.length || !sessionId) return;

  showOverlay(`Processing ${pendingFiles.length} file${pendingFiles.length > 1 ? "s" : ""}\u2026`);
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

    const uploadedNames = data.uploaded.map(u => u.filename);
    const uploadMsg = addMessage("assistant",
      `**${data.uploaded.length} file${data.uploaded.length > 1 ? "s" : ""} added.**\n\n` +
      `${data.session.document_count} passages are ready for questions.`
    );

    // Add clickable file badges to the message
    const content = uploadMsg.querySelector(".msg-content");
    if (content) {
      const bubble = content.querySelector(".bubble");
      if (bubble) {
        bubble.insertBefore(buildFileBadges(uploadedNames), bubble.firstChild);
      }
    }

    // Fetch AI-generated suggestion chips
    loadSuggestionChips(uploadMsg);

    pendingFiles = [];
    fileInput.value = "";
    showPendingFiles([]);
    closeSidebar();

  } catch (err) {
    addMessage("assistant", `Network error during upload: ${err.message}`);
  } finally {
    hideOverlay();
    uploadBtn.disabled = pendingFiles.length === 0;
  }
});

/* -- Chat (streaming SSE) ----------------------------------------------------- */

async function sendQuestion(question) {
  if (isStreaming) return;
  isStreaming = true;
  sendBtn.disabled = true;
  questionInput.disabled = true;

  // Remove any existing follow-up chips so they don't pile up
  document.querySelectorAll(".followup-chips").forEach(el => el.remove());

  addMessage("user", question);

  const { div: msgDiv, content, bubble, pipeline } = addStreamingMessage();

  let fullText = "";
  let sources = [];
  let renderPending = false;
  let streamDone = false;

  function scheduleRender() {
    if (renderPending) return;
    renderPending = true;
    requestAnimationFrame(() => {
      renderPending = false;
      if (bubble && !streamDone) {
        bubble.innerHTML = renderMarkdown(fullText) + '<span class="stream-cursor"></span>';
        messages.scrollTop = messages.scrollHeight;
      }
    });
  }

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
            scheduleRender();
            break;

          case "done":
            streamDone = true;
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
      streamDone = true;
      finishStreamingMessage(content, fullText, sources);
    }

  } catch (err) {
    msgDiv.remove();
    addMessage("assistant", `Network error: ${err.message}`);
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    questionInput.disabled = false;
    if (!isMobile()) questionInput.focus();
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

/* Submit on Enter (Shift+Enter for newline) */
questionInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    inputForm.dispatchEvent(new Event("submit", { cancelable: true }));
  }
});

/* -- New session button ------------------------------------------------------- */
newSessionBtn.addEventListener("click", async () => {
  if (!confirm("Start a new conversation? Your current session data will be cleared.")) return;
  await resetSession();
});

/* -- Mobile sidebar toggle ---------------------------------------------------- */
function openSidebar() {
  sidebar.classList.add("open");
  if (sidebarScrim) sidebarScrim.classList.add("active");
}

function closeSidebar() {
  sidebar.classList.remove("open");
  if (sidebarScrim) sidebarScrim.classList.remove("active");
}

sidebarToggle.addEventListener("click", () => {
  sidebar.classList.contains("open") ? closeSidebar() : openSidebar();
});

if (sidebarScrim) {
  sidebarScrim.addEventListener("click", closeSidebar);
}

document.addEventListener("click", e => {
  if (
    sidebar.classList.contains("open") &&
    !sidebar.contains(e.target) &&
    !sidebarToggle.contains(e.target) &&
    (!sidebarScrim || e.target !== sidebarScrim)
  ) {
    closeSidebar();
  }
});

/* -- Bootstrap ---------------------------------------------------------------- */
(async () => {
  showWelcome();

  // Scroll-to-bottom button
  messages.addEventListener("scroll", () => {
    const distFromBottom = messages.scrollHeight - messages.scrollTop - messages.clientHeight;
    if (distFromBottom > 200) {
      scrollBottomBtn.style.display = "";
      scrollBottomBtn.classList.add("visible");
    } else {
      scrollBottomBtn.classList.remove("visible");
      setTimeout(() => {
        if (!scrollBottomBtn.classList.contains("visible")) scrollBottomBtn.style.display = "none";
      }, 250);
    }
  });

  scrollBottomBtn.addEventListener("click", () => {
    messages.scrollTo({ top: messages.scrollHeight, behavior: "smooth" });
  });

  try {
    await initSession();
    if (activeProvider) showWelcome();
  } catch (err) {
    addMessage("assistant", `Could not connect to the server: ${err.message}`);
  }
})();
