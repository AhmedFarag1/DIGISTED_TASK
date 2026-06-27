const API = "/api";

const els = {
  messages: document.getElementById("messages"),
  emptyState: document.getElementById("emptyState"),
  composer: document.getElementById("composer"),
  question: document.getElementById("question"),
  sendBtn: document.getElementById("sendBtn"),
  topK: document.getElementById("topK"),
  domain: document.getElementById("domain"),
  difficulty: document.getElementById("difficulty"),
  multiTurn: document.getElementById("multiTurn"),
  tts: document.getElementById("tts"),
  evalBtn: document.getElementById("evalBtn"),
  evalSamples: document.getElementById("evalSamples"),
  evalOutput: document.getElementById("evalOutput"),
  statusDot: document.getElementById("statusDot"),
  statusText: document.getElementById("statusText"),
};

let conversationId = null;

function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function hideEmpty() {
  if (els.emptyState) els.emptyState.style.display = "none";
}

function scrollDown() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function addUser(text) {
  hideEmpty();
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  els.messages.appendChild(el);
  scrollDown();
}

function addTyping() {
  const el = document.createElement("div");
  el.className = "msg bot";
  el.id = "typing";
  el.innerHTML = `<div class="bubble"><span class="typing"><span></span><span></span><span></span></span></div>`;
  els.messages.appendChild(el);
  scrollDown();
  return el;
}

function renderSources(passages) {
  if (!passages || !passages.length) return "";
  const items = passages.map((p, i) => `
    <div class="source-item">
      <div class="src-head">
        <span>[${i + 1}]</span>
        <span class="badge">${escapeHtml(p.domain)}</span>
        <span class="badge">${escapeHtml(p.difficulty)}</span>
        <span>score ${p.score.toFixed(3)}</span>
      </div>
      <div class="src-text">${escapeHtml(p.text.slice(0, 320))}${p.text.length > 320 ? "…" : ""}</div>
    </div>`).join("");
  return `<details class="sources"><summary>▸ ${passages.length} source passage(s)</summary>${items}</details>`;
}

function addBot(data) {
  const confClass = data.confidence >= 0.5 ? "conf-good" : "conf-low";
  const flags = [];
  if (data.cached) flags.push(`<span class="badge">cached</span>`);
  if (data.used_fallback) flags.push(`<span class="badge">fallback</span>`);
  if (data.tts_text) flags.push(`<span class="badge">TTS ready</span>`);

  const bd = data.latency_breakdown || {};
  const breakdown =
  bd.retrieval_ms != null
    ? ` · retrieval ${bd.retrieval_ms}ms · gemini ${bd.generation_ms}ms`
    : "";

  const el = document.createElement("div");
  el.className = "msg bot";
  el.innerHTML = `
    <div class="bubble">${escapeHtml(data.answer)}</div>
    <div class="meta">
      <span class="${confClass}">confidence ${(data.confidence * 100).toFixed(0)}%</span>
      <span>${data.latency_ms.toFixed(0)} ms${breakdown}</span>
      ${flags.join(" ")}
    </div>
    ${renderSources(data.passages)}`;
  els.messages.appendChild(el);
  scrollDown();
}

function addError(msg) {
  const el = document.createElement("div");
  el.className = "msg bot";
  el.innerHTML = `<div class="bubble" style="border-color:#ef4444">⚠️ ${escapeHtml(msg)}</div>`;
  els.messages.appendChild(el);
  scrollDown();
}

async function ask(question) {
  addUser(question);
  els.sendBtn.disabled = true;
  const typing = addTyping();

  if (els.multiTurn.checked && !conversationId) {
    conversationId = "web-" + Math.random().toString(36).slice(2, 10);
  }
  if (!els.multiTurn.checked) conversationId = null;

  const filters = {};
  if (els.domain.value) filters.domain = els.domain.value;
  if (els.difficulty.value) filters.difficulty = els.difficulty.value;

  const payload = {
    question,
    top_k: parseInt(els.topK.value, 10) || 5,
    conversation_id: conversationId,
    prepare_tts: els.tts.checked,
    filters: Object.keys(filters).length ? filters : null,
  };

  try {
    const res = await fetch(`${API}/ask-question`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    typing.remove();
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      addError(err.detail || `Request failed (${res.status})`);
      return;
    }
    const data = await res.json();
    if (data.conversation_id) conversationId = data.conversation_id;
    addBot(data);
  } catch (e) {
    typing.remove();
    addError(e.message || "Network error");
  } finally {
    els.sendBtn.disabled = false;
    els.question.focus();
  }
}

els.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = els.question.value.trim();
  if (!q) return;
  els.question.value = "";
  els.question.style.height = "auto";
  ask(q);
});

els.question.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    els.composer.requestSubmit();
  }
});
els.question.addEventListener("input", () => {
  els.question.style.height = "auto";
  els.question.style.height = Math.min(els.question.scrollHeight, 140) + "px";
});

document.querySelectorAll(".chip").forEach((c) =>
  c.addEventListener("click", () => ask(c.dataset.q))
);

els.evalBtn.addEventListener("click", async () => {
  els.evalBtn.disabled = true;
  els.evalBtn.textContent = "Running…";
  els.evalOutput.hidden = false;
  els.evalOutput.textContent = "Evaluating… this may take 2–10 minutes (E5 + Gemini per sample).";
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 600000); // 10 min
  try {
    const res = await fetch(`${API}/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        num_samples: parseInt(els.evalSamples.value, 10) || 3,
        top_k: parseInt(els.topK.value, 10) || 5,
      }),
      signal: controller.signal,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Evaluation failed");
    els.evalOutput.textContent = JSON.stringify(
      { retrieval: data.retrieval, generation: data.generation, performance: data.performance },
      null, 2
    );
  } catch (e) {
    els.evalOutput.textContent = "Error: " + (e.name === "AbortError" ? "Timed out after 10 minutes." : (e.message || e));
  } finally {
    clearTimeout(timer);
    els.evalBtn.disabled = false;
    els.evalBtn.textContent = "Run evaluation";
  }
});

async function checkHealth() {
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    const dot = els.statusDot;
    dot.className = "dot";
    if (data.status === "ok") {
      dot.classList.add("ok");
      const pts = data.detail?.vector_store?.points ?? 0;
      els.statusText.textContent = `ready · ${pts} chunks`;
    } else if (data.status === "degraded") {
      dot.classList.add("warn");
      const pts = data.detail?.vector_store?.points ?? 0;
      const embed = data.detail?.embedding_ready;
      if (pts > 0 && embed === false) {
        els.statusText.textContent = `${pts} chunks · embedder not loaded`;
      } else {
        els.statusText.textContent = "no index — run ingestion";
      }
    } else {
      dot.classList.add("err");
      els.statusText.textContent = "config error";
    }
  } catch {
    els.statusDot.className = "dot err";
    els.statusText.textContent = "offline";
  }
}

checkHealth();
setInterval(checkHealth, 30000);
