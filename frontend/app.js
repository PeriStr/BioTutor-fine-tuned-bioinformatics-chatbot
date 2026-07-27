// app.js — chat behaviour for BioTutor.
// Talks to the backend at POST /api/chat and renders the conversation.

const chat = document.getElementById("chat");
const welcome = document.getElementById("welcome");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const statusEl = document.getElementById("status");
const statusText = document.getElementById("statusText");
const clearBtn = document.getElementById("clearBtn");

let busy = false;   // prevents sending a second message while one is in flight

// A stable per-browser id, so this browser's history stays separate from others.
let sessionId = localStorage.getItem("biotutor_session");
if (!sessionId){
  sessionId = "s-" + Math.random().toString(36).slice(2) + Date.now().toString(36);
  localStorage.setItem("biotutor_session", sessionId);
}

// ---------- Helpers ----------
function escapeHtml(s){
  return String(s).replace(/[&<>]/g, c => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;" }[c]));
}
function scrollToBottom(){ chat.scrollTop = chat.scrollHeight; }

// Add a message bubble. `who` is "user", "bot", or "error".
function addMessage(who, text){
  const msg = document.createElement("div");
  msg.className = "msg " + who;
  const avatar = who === "user" ? "🧑" : "🧬";
  msg.innerHTML = `
    <div class="avatar">${avatar}</div>
    <div class="bubble">${escapeHtml(text)}</div>`;
  chat.appendChild(msg);
  scrollToBottom();
  return msg;
}

// Add the animated "typing…" bubble and return it so we can remove it later.
function addTyping(){
  const msg = document.createElement("div");
  msg.className = "msg bot typing";
  msg.innerHTML = `
    <div class="avatar">🧬</div>
    <div class="bubble"><span></span><span></span><span></span></div>`;
  chat.appendChild(msg);
  scrollToBottom();
  return msg;
}

// ---------- Sending ----------
async function send(text){
  const message = (text ?? input.value).trim();
  if (!message || busy) return;

  if (welcome) welcome.remove();          // hide the welcome screen on first message
  addMessage("user", message);
  input.value = "";
  autoGrow();
  setBusy(true);

  const typing = addTyping();
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
    });
    const data = await res.json();
    typing.remove();
    if (data.error) addMessage("error", "⚠️ " + data.error);
    else addMessage("bot", data.reply || "(no answer)");
  } catch (e) {
    typing.remove();
    addMessage("error", "⚠️ Could not reach the server. Is the backend running?");
  } finally {
    setBusy(false);
    input.focus();
  }
}

function setBusy(state){
  busy = state;
  sendBtn.disabled = state;
}

// ---------- Input UX ----------
function autoGrow(){
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 140) + "px";
}
input.addEventListener("input", autoGrow);
input.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey){    // Enter sends, Shift+Enter = newline
    e.preventDefault();
    send();
  }
});
sendBtn.addEventListener("click", () => send());

// Suggestion chips
document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => send(chip.textContent));
});

// ---------- Health check (shows device + confirms the model is up) ----------
async function checkHealth(){
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    statusEl.classList.add("online");
    const dev = (data.device || "").toUpperCase();
    statusText.textContent = data.adapter_loaded ? `ready · ${dev}` : `base model · ${dev}`;
  } catch {
    statusEl.classList.add("offline");
    statusText.textContent = "offline";
  }
}

// ---------- Load saved conversation from the database ----------
async function loadHistory(){
  try {
    const res = await fetch("/api/history?session_id=" + encodeURIComponent(sessionId));
    const data = await res.json();
    if (data.history && data.history.length){
      if (welcome) welcome.remove();
      for (const row of data.history){
        addMessage("user", row.question);
        addMessage("bot", row.answer);
      }
    }
  } catch { /* ignore — first run or offline */ }
}

// ---------- Clear the conversation ----------
clearBtn.addEventListener("click", async () => {
  if (!confirm("Clear this conversation?")) return;
  try {
    await fetch("/api/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
  } catch {}
  chat.innerHTML = "";
  location.reload();   // reload to show the fresh welcome screen
});

checkHealth();
loadHistory();
input.focus();
