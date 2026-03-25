/**
 * RAG Bot – frontend logic
 *
 * Flow:
 *  1. On load → POST /api/session → store sessionId
 *  2. User selects files → click "Upload & Process" → POST /api/upload/:id
 *  3. User types question → POST /api/chat/:id → display answer
 */

"use strict";

/* ── DOM refs ─────────────────────────────────────────────────────────── */
const dropZone       = document.getElementById("dropZone");
const fileInput      = document.getElementById("fileInput");
const uploadBtn      = document.getElementById("uploadBtn");
const newSessionBtn  = document.getElementById("newSessionBtn");
const fileListSec    = document.getElementById("fileListSection");
const fileList       = document.getElementById("fileList");
const messages       = document.getElementById("messages");
const inputForm      = document.getElementById("inputForm");
const questionInput  = document.getElementById("questionInput");
const sendBtn        = document.getElementById("sendBtn");
const sessionLabel   = document.getElementById("sessionLabel");
const overlay        = document.getElementById("overlay");
const overlayMsg     = document.getElementById("overlayMsg");

/* ── State ────────────────────────────────────────────────────────────── */
let sessionId    = null;
let pendingFiles = [];          // FileList → array

/* ── Helpers ──────────────────────────────────────────────────────────── */

function showOverlay(msg = "Processing…") {
  overlayMsg.textContent = msg;
  overlay.classList.remove("hidden");
}

function hideOverlay() {
  overlay.classList.add("hidden");
}

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  div.appendChild(bubble);
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

function addTypingIndicator() {
  const div = document.createElement("div");
  div.className = "message assistant typing";
  div.id = "typingIndicator";
  div.innerHTML = `<div class="bubble">
    <span class="dot"></span><span class="dot"></span><span class="dot"></span>
  </div>`;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

function removeTypingIndicator() {
  const el = document.getElementById("typingIndicator");
  if (el) el.remove();
}

function updateFileList(names) {
  fileList.innerHTML = "";
  names.forEach(name => {
    const li = document.createElement("li");
    li.textContent = name;
    fileList.appendChild(li);
  });
  fileListSec.style.display = names.length ? "" : "none";
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
  sessionLabel.textContent = `Session: ${sessionId.slice(0, 8)}…`;
}

async function resetSession() {
  pendingFiles = [];
  fileInput.value = "";
  uploadBtn.disabled = true;
  updateFileList([]);
  questionInput.disabled = true;
  sendBtn.disabled = true;
  messages.innerHTML = `<div class="message assistant">
    <div class="bubble">👋 Hello! Upload one or more documents using the sidebar, then ask me anything about their contents.</div>
  </div>`;
  await initSession();
}

/* ── File selection ───────────────────────────────────────────────────── */

function handleFiles(files) {
  pendingFiles = Array.from(files);
  uploadBtn.disabled = pendingFiles.length === 0;
  if (pendingFiles.length > 0) {
    dropZone.querySelector(".drop-text").textContent =
      pendingFiles.map(f => f.name).join(", ");
  }
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

  showOverlay(`Processing ${pendingFiles.length} file(s)…`);
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
      addMessage("assistant", `⚠️ Upload failed: ${data.detail || "Unknown error"}`);
      return;
    }

    const names = data.session.file_names;
    updateFileList(names);
    enableChat();

    const fileNames = data.uploaded.map(u => u.filename).join(", ");
    const docRef = names.length > 1 ? "these documents" : "this document";
    addMessage("assistant", `✅ Ingested ${data.uploaded.length} file(s): ${fileNames}.\n\nYou can now ask me anything about ${docRef}!`);

    // Reset pending selection
    pendingFiles = [];
    fileInput.value = "";
    dropZone.querySelector(".drop-text").textContent = "Click or drag & drop files here";

  } catch (err) {
    addMessage("assistant", `⚠️ Network error during upload: ${err.message}`);
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
      addMessage("assistant", `⚠️ ${data.detail || "Something went wrong."}`);
    } else {
      addMessage("assistant", data.answer);
    }

  } catch (err) {
    removeTypingIndicator();
    addMessage("assistant", `⚠️ Network error: ${err.message}`);
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
  if (!confirm("Start a new session? The current conversation will be cleared.")) return;
  await resetSession();
});

/* ── Bootstrap ────────────────────────────────────────────────────────── */
(async () => {
  try {
    await initSession();
  } catch (err) {
    addMessage("assistant", `⚠️ Could not connect to the server: ${err.message}`);
  }
})();
