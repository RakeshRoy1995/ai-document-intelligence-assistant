"use strict";

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pages = (a, b) => (!a ? "" : a === b || !b ? `p${a}` : `p${a}–${b}`);
const num = (v) => (typeof v === "number" ? v.toFixed(2) : null);
const fmtScores = (s) =>
  [num(s.score) && `rerank ${num(s.score)}`, num(s.vector_score) && `vec ${num(s.vector_score)}`]
    .filter(Boolean).join(" · ");
const badge = (t) => `<span class="badge ${["text", "table", "figure"].includes(t) ? t : "other"}">${esc(t)}</span>`;

const state = {
  sessionId: localStorage.getItem("ragSession") || crypto.randomUUID(),
  docs: [],
  selected: null,       // source name
  detail: null,         // {items, chunks}
  rawView: false,
  jobs: new Map(),      // job_id -> job
};
localStorage.setItem("ragSession", state.sessionId);

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `${res.status} ${res.statusText}`);
  return body;
}

// ============================================================
// TABS
// ============================================================

document.querySelectorAll(".tab").forEach((tab) =>
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === `panel-${tab.dataset.tab}`));
  })
);

// ============================================================
// UPLOAD
// ============================================================

const dropzone = $("#dropzone");
const fileInput = $("#fileInput");

fileInput.addEventListener("change", () => {
  uploadFiles([...fileInput.files]);
  fileInput.value = "";
});
["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("over"); })
);
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("over"); })
);
dropzone.addEventListener("drop", (e) => uploadFiles([...e.dataTransfer.files]));

async function uploadFiles(files) {
  for (const file of files) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      addLocalJob(file.name, "error", "Not a PDF file");
      continue;
    }
    const tempId = addLocalJob(file.name, "running", "Uploading…");
    try {
      const form = new FormData();
      form.append("file", file);
      const job = await api("/api/upload", { method: "POST", body: form });
      state.jobs.delete(tempId);
      state.jobs.set(job.job_id, job);
      renderJobs();
      pollJob(job.job_id);
    } catch (err) {
      state.jobs.set(tempId, { ...state.jobs.get(tempId), status: "error", stage: err.message });
      renderJobs();
    }
  }
}

function addLocalJob(source, status, stage) {
  const id = "local-" + Math.random().toString(36).slice(2);
  state.jobs.set(id, { job_id: id, source, status, stage, progress: 0 });
  renderJobs();
  return id;
}

async function pollJob(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    state.jobs.set(jobId, job);
    renderJobs();
    if (job.status === "running") return setTimeout(() => pollJob(jobId), 1500);
    if (job.status === "done") {
      await loadDocuments();
      selectDocument(job.source);
      setTimeout(() => { state.jobs.delete(jobId); renderJobs(); }, 6000);
    }
  } catch {
    setTimeout(() => pollJob(jobId), 3000);
  }
}

function renderJobs() {
  $("#jobs").innerHTML = [...state.jobs.values()].map((j) => {
    const detail = j.status === "done"
      ? `Done · ${j.chunks} chunks from ${j.items} items`
      : j.status === "error" ? `Error: ${j.error || j.stage}` : j.stage;
    return `<div class="job ${j.status}">
      <div class="job-name" title="${esc(j.source)}">${esc(j.source)}</div>
      <div class="job-stage">${esc(detail)}</div>
      <div class="bar"><div style="width:${j.status === "error" ? 100 : j.progress || 2}%"></div></div>
    </div>`;
  }).join("");
}

// ============================================================
// DOCUMENTS
// ============================================================

async function loadDocuments() {
  const data = await api("/api/documents");
  state.docs = data.documents;
  for (const j of data.jobs) {           // resume progress bars after a page reload
    if (!state.jobs.has(j.job_id)) { state.jobs.set(j.job_id, j); pollJob(j.job_id); }
  }
  renderJobs();
  renderDocs();
}

function renderDocs() {
  const totalChunks = state.docs.reduce((n, d) => n + d.chunks, 0);
  $("#stats").textContent = state.docs.length
    ? `${state.docs.length} document${state.docs.length > 1 ? "s" : ""} · ${totalChunks} chunks in Chroma`
    : "No documents yet";

  $("#docList").innerHTML = state.docs.length
    ? state.docs.map((d) => {
        const types = Object.entries(d.types).map(([t, n]) => `${n} ${t}`).join(", ");
        return `<li class="doc ${d.source === state.selected ? "active" : ""}" data-source="${esc(d.source)}">
          <div class="doc-name">${esc(d.source)}</div>
          <div class="doc-meta">${d.pages} pages · ${d.chunks} chunks</div>
          <div class="doc-meta" title="${esc(types)}">${esc(types)}</div>
        </li>`;
      }).join("")
    : `<li class="empty">Upload a PDF to get started.</li>`;

  document.querySelectorAll(".doc").forEach((el) =>
    el.addEventListener("click", () => {
      selectDocument(el.dataset.source);
      document.querySelector('.tab[data-tab="chunks"]').click();
    })
  );
}

async function selectDocument(source) {
  state.selected = source;
  renderDocs();
  state.detail = await api(`/api/documents/${encodeURIComponent(source)}`);
  renderChunks();
  renderItems();
}

// ============================================================
// CHUNKS VIEW
// ============================================================

["#chunkFilter", "#chunkSearch"].forEach((s) => $(s).addEventListener("input", renderChunks));

$("#viewToggle").addEventListener("click", () => {
  state.rawView = !state.rawView;
  $("#viewToggle").textContent = state.rawView ? "Cards" : "Raw JSON";
  renderChunks();
});

$("#downloadChunks").addEventListener("click", () => {
  if (!state.detail) return;
  const blob = new Blob([JSON.stringify(filteredChunks(), null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = state.selected.replace(/\.pdf$/i, "") + ".chunks.json";
  a.click();
  URL.revokeObjectURL(a.href);
});

function filteredChunks() {
  if (!state.detail) return [];
  const type = $("#chunkFilter").value;
  const q = $("#chunkSearch").value.trim().toLowerCase();
  return state.detail.chunks.filter((c) =>
    (!type || c.chunk_type === type) &&
    (!q || c.content.toLowerCase().includes(q) || (c.section_path || "").toLowerCase().includes(q))
  );
}

function chunkCard(c) {
  const img = c.image ? `<img class="fig" src="/images/${encodeURIComponent(c.image)}" loading="lazy" alt="">` : "";
  return `<div class="card">
    <div class="card-head">
      ${badge(c.chunk_type)}
      <span>#${c.chunk_index}</span>
      <span>${pages(c.page_start, c.page_end)}</span>
      <span>${c.char_count ?? c.content.length} chars</span>
      ${c.section_path ? `<span class="path">${esc(c.section_path)}</span>` : ""}
      <span class="id" style="margin-left:auto">${esc(c.chunk_id)}</span>
    </div>
    <pre class="content ${c.chunk_type === "table" ? "mono" : ""}">${esc(c.content)}</pre>
    ${img}
  </div>`;
}

function renderChunks() {
  const d = state.detail;
  $("#downloadChunks").disabled = !d;
  if (!d) return;
  const list = filteredChunks();
  const counts = {};
  d.chunks.forEach((c) => (counts[c.chunk_type] = (counts[c.chunk_type] || 0) + 1));

  $("#chunksTitle").textContent = d.source;
  $("#chunksMeta").textContent =
    `${d.chunks.length} chunks (${Object.entries(counts).map(([t, n]) => `${n} ${t}`).join(", ")})` +
    (list.length !== d.chunks.length ? ` · showing ${list.length}` : "");

  $("#chunksBody").hidden = state.rawView;
  $("#chunksJson").hidden = !state.rawView;
  if (state.rawView) {
    $("#chunksJson").textContent = JSON.stringify(list, null, 2);
  } else {
    $("#chunksBody").innerHTML = list.map(chunkCard).join("") || `<div class="muted">No chunks match.</div>`;
  }
}

// ============================================================
// EXTRACTED ITEMS VIEW
// ============================================================

function renderItems() {
  const d = state.detail;
  if (!d) return;
  const counts = {};
  d.items.forEach((it) => (counts[it.type] = (counts[it.type] || 0) + 1));
  $("#itemsTitle").textContent = d.source;
  $("#itemsMeta").textContent = `${d.items.length} layout items · ` +
    Object.entries(counts).map(([t, n]) => `${n} ${t}`).join(", ");

  $("#itemsBody").innerHTML = d.items.map((it) => {
    let body = esc(it.text);
    if (it.type === "heading") body = `${"#".repeat(Math.max(1, (it.level || 0) + 1))} ${body}`;
    if (it.image) body += `<br><img class="fig" src="/images/${encodeURIComponent(it.image)}" loading="lazy" alt="">`;
    return `<div class="item ${esc(it.type)}">
      <span class="type">${esc(it.type)}</span>
      <span class="pg">p${it.page}</span>
      <span class="txt ${it.type === "table" ? "content mono" : ""}">${body}</span>
    </div>`;
  }).join("");
}

// ============================================================
// ASK
// ============================================================

const chat = $("#chat");

function addMessage(role, html) {
  chat.querySelector(".chat-empty")?.remove();
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.innerHTML = html;
  chat.appendChild(el);
  chat.scrollTop = chat.scrollHeight;
  return el;
}

function renderAnswer(result) {
  const answer = esc(result.answer).replace(/\[(\d+)\]/g, (m, n) => `<span class="cite" data-n="${n}">${n}</span>`);
  const sources = result.sources.map((s) => {
    const img = s.image ? `<img class="fig" src="/images/${encodeURIComponent(s.image)}" loading="lazy" alt="">` : "";
    return `<details class="source" data-n="${s.n}">
      <summary>
        <span class="cite">${s.n}</span>
        ${badge(s.chunk_type)}
        <span>${esc(s.source)} · ${pages(s.page_start, s.page_end)}</span>
        <span class="score" title="cross-encoder rerank score · cosine similarity">${fmtScores(s)}</span>
      </summary>
      <div class="src-body">
        ${s.section_path ? `<div class="muted">${esc(s.section_path)}</div>` : ""}
        <pre class="content ${s.chunk_type === "table" ? "mono" : ""}">${esc(s.content)}</pre>
        ${img}
      </div>
    </details>`;
  }).join("");
  const cached = result.from_cache
    ? `<span class="badge other" title="Reused from an earlier question with the same meaning">cached</span>`
    : "";
  const feedback = result.answer_id
    ? `<div class="feedback" data-id="${result.answer_id}">
        ${cached}<span class="muted">Helpful?</span>
        <button class="ghost fb" data-up="1" title="Helpful">👍</button>
        <button class="ghost fb" data-up="0" title="Not helpful">👎</button>
      </div>`
    : "";
  return `<div class="bubble">${answer}</div>${feedback}<div class="sources">${sources}</div>`;
}

chat.addEventListener("click", async (e) => {
  const vote = e.target.closest(".fb");
  if (vote) {
    const row = vote.closest(".feedback");
    const note = row.querySelector(".muted");
    row.querySelectorAll(".fb").forEach((b) => (b.disabled = true));
    try {
      await api("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ answer_id: Number(row.dataset.id), helpful: vote.dataset.up === "1" }),
      });
      note.textContent = "Thanks for the feedback";
    } catch (err) {
      note.textContent = err.message;
    }
    return;
  }
  const cite = e.target.closest(".bubble .cite");
  if (!cite) return;
  const src = cite.closest(".msg").querySelector(`.source[data-n="${cite.dataset.n}"]`);
  if (src) { src.open = true; src.scrollIntoView({ behavior: "smooth", block: "nearest" }); }
});

$("#askForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("#question").value.trim();
  if (!question) return;
  $("#question").value = "";
  $("#askBtn").disabled = true;

  addMessage("user", `<div class="bubble">${esc(question)}</div>`);
  const pending = addMessage("assistant", `<div class="bubble typing">Searching the vector DB…</div>`);
  try {
    const result = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        session_id: state.sessionId,
        k: Number($("#topK").value),
        chunk_type: $("#chunkType").value || null,
      }),
    });
    pending.innerHTML = renderAnswer(result);
  } catch (err) {
    pending.innerHTML = `<div class="bubble" style="color:var(--err)">${esc(err.message)}</div>`;
  } finally {
    $("#askBtn").disabled = false;
    $("#question").focus();
    chat.scrollTop = chat.scrollHeight;
  }
});

$("#resetBtn").addEventListener("click", async () => {
  await api("/api/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: state.sessionId }),
  }).catch(() => {});
  chat.innerHTML = `<div class="chat-empty"><h2>Ask your documents</h2><p>Conversation cleared.</p></div>`;
});

loadDocuments().catch((err) => ($("#stats").textContent = err.message));
