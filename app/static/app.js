/* Second Brain — Document Test Bench (vanilla JS, no frameworks) */
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const byName = (n) => $(`input[name="mode"][value="${n}"]`);

const state = { documents: 0, allowAI: false, sources: [], lastAnswer: null };

const el = {
  healthDot: $('#health-dot'),
  healthLabel: $('#health-label'),
  indexBadge: $('#index-badge'),
  themeToggle: $('#theme-toggle'),

  dropzone: $('#upload-form'),
  fileInput: $('#file-input'),
  selectedFile: $('#selected-file'),
  uploadBtn: $('#upload-btn'),
  uploadMsg: $('#upload-message'),
  uploadList: $('#upload-list'),
  baseList: $('#base-list'),
  clearUploads: $('#clear-uploads'),

  questionForm: $('#question-form'),
  question: $('#question'),
  topK: $('#top-k'),
  sourceFilter: $('#source-filter'),
  askBtn: $('#ask-btn'),
  askMsg: $('#ask-message'),
  results: $('#results'),
  emptyState: $('#empty-state'),
  hitTemplate: $('#hit-template'),
  skeleton: $('#skeleton'),

  progressPanel: $('#ask-progress'),
  progressStage: $('#progress-stage'),
  progressElapsed: $('#progress-elapsed'),
  progressFill: $('#progress-fill'),
  progressHint: $('#progress-hint'),

  allowAi: $('#allow-ai'),
  answerBtn: $('#answer-btn'),
  answerPanel: $('#answer-panel'),
  answerText: $('#answer-text'),
  answerModel: $('#answer-model'),
  sentList: $('#sent-list'),
  copyAnswer: $('#copy-answer'),
  copyCitations: $('#copy-citations'),
  copyFeedback: $('#copy-feedback'),
};

const HINTS = [
  'Edit one file → only that file is re-embedded. Nothing else is touched.',
  'Press / to jump straight to the question box.',
  'Benchmark on this corpus: hit@3 = 33/33 (baseline, 12 missing-source tests skipped).',
  'Vector measured best here — the other modes are under Advanced retrieval.',
  'Uploaded documents merge with data/ — the corpus grows, locally.',
];

const fmtBytes = (n) =>
  n >= 1 << 20 ? `${(n / (1 << 20)).toFixed(1)} MB`
    : n >= 1024 ? `${Math.round(n / 1024)} KB`
                : `${n} B`;

const modeOf = () => (document.querySelector('input[name="mode"]:checked') || { value: 'vector' }).value;

function showNotice(target, text, isError = false) {
  target.hidden = false;
  target.textContent = text;
  target.classList.toggle('error', isError);
}
const clearNotice = (target) => { target.hidden = true; target.textContent = ''; };

const isEditable = (t) =>
  t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable);

/* ---------- request lifecycle / progress ---------- */

let pollTimer = null, elapsedTimer = null, hintTimer = null;
let queryStartedAt = 0, hintIndex = 0;

function stageText(p) {
  if (!p.active) return null;
  if (p.stage === 'reading') return `Reading ${p.current_file || 'files'} ${p.files_total ? `(${p.file_index}/${p.files_total})` : ''}`;
  if (p.stage === 'embedding') return `Embedding ${p.current_file || 'documents'} ${p.file_index}/${p.files_total}`;
  return 'Working…';
}
function stageFraction(p) {
  if (p.stage === 'embedding' && p.chunks_total) return p.chunks_done / p.chunks_total;
  if (p.stage === 'reading' && p.files_total) return (p.file_index / p.files_total) * 0.35;
  return null;
}
async function pollIndexStatus() {
  try {
    const res = await fetch('/api/index-status');
    if (!res.ok) return;
    const p = await res.json();
    if (!p.active) return;
    const label = stageText(p);
    if (label) el.progressStage.textContent = label;
    const frac = stageFraction(p);
    el.progressFill.style.width = frac == null ? '8%' : `${Math.max(2, Math.round(frac * 100))}%`;
  } catch { /* polling is best-effort; the bar just stays put */ }
}
function beginProgress() {
  el.progressPanel.hidden = false;
  el.skeleton.hidden = false;
  el.results.replaceChildren();
  el.emptyState.hidden = true;
  el.progressFill.style.width = '2%';
  el.progressStage.textContent = 'Preparing question…';
  el.progressElapsed.textContent = '0.0s';
  el.progressHint.textContent = HINTS[0];
  hintIndex = 0;
  queryStartedAt = Date.now();

  elapsedTimer = setInterval(() => {
    el.progressElapsed.textContent = `${((Date.now() - queryStartedAt) / 1000).toFixed(1)}s`;
  }, 100);
  hintTimer = setInterval(() => {
    hintIndex = (hintIndex + 1) % HINTS.length;
    el.progressHint.textContent = HINTS[hintIndex];
  }, 4500);
  pollTimer = setInterval(pollIndexStatus, 1200);
  pollIndexStatus();
}
function endProgress() {
  clearInterval(elapsedTimer); clearInterval(hintTimer); clearInterval(pollTimer);
  el.progressPanel.hidden = true;
  el.skeleton.hidden = true;
}

/* ---------- status / lists ---------- */

async function refreshStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) throw new Error(`status ${res.status}`);
    const s = await res.json();
    state.documents = s.documents;
    state.sources = s.sources || [];
    renderSourceFilter();

    renderFileList(el.uploadList, s.uploads, true);
    renderFileList(el.baseList, s.base, false);
    el.clearUploads.disabled = s.uploads.length === 0;

    if (s.documents > 0) {
      setBusy('ok', `Ready · ${s.documents} document${s.documents === 1 ? '' : 's'} · ${s.model}`, 'Service up');
    } else {
      setBusy('idle', 'Waiting for documents', 'Service up');
    }
    return s;
  } catch {
    el.healthDot.classList.remove('ok', 'busy');
    el.healthLabel.textContent = 'Status unknown';
    el.indexBadge.className = 'workbench-status';
    el.indexBadge.textContent = 'Status unknown';
  }
}

function renderSourceFilter() {
  const previous = el.sourceFilter.value;
  const options = ['<option value="">All files</option>',
    ...state.sources.map((n) => `<option value="${escapeAttr(n)}">${escapeHtml(n)}</option>`)];
  el.sourceFilter.innerHTML = options.join('');
  el.sourceFilter.value = state.sources.includes(previous) ? previous : '';
}
const escapeHtml = (s) => s.replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const escapeAttr = escapeHtml;

function renderFileList(list, files, removable) {
  list.replaceChildren();
  if (!files.length) {
    const li = document.createElement('li');
    li.className = 'file-list-empty';
    li.textContent = removable ? 'Nothing uploaded yet.' : 'The data/ folder is empty.';
    list.appendChild(li);
    return;
  }
  for (const file of files) {
    const li = document.createElement('li');
    li.className = 'file-row';

    const icon = document.createElement('span');
    icon.className = 'file-icon';
    icon.textContent = (file.name.split('.').pop() || 'file').slice(0, 4);

    const meta = document.createElement('span');
    meta.className = 'file-meta';
    const name = document.createElement('span');
    name.className = 'file-name';
    name.textContent = file.name;
    name.title = file.name;
    const sub = document.createElement('span');
    sub.className = 'file-sub';
    sub.textContent = fmtBytes(file.bytes);
    meta.append(name, sub);
    li.append(icon, meta);

    if (removable) {
      const btn = document.createElement('button');
      btn.className = 'file-remove';
      btn.type = 'button';
      btn.textContent = 'Remove';
      btn.setAttribute('aria-label', `Remove ${file.name}`);
      btn.addEventListener('click', () => removeUpload(file.name));
      li.append(btn);
    }
    list.appendChild(li);
  }
}

function setBusy(kind, badgeText, healthText) {
  el.healthDot.classList.toggle('busy', kind === 'busy');
  el.healthDot.classList.toggle('ok', kind === 'ok');
  el.healthLabel.textContent = healthText;
  el.indexBadge.className = 'workbench-status' + (kind === 'busy' ? ' busy' : kind === 'ok' ? ' ready' : '');
  el.indexBadge.textContent = badgeText;
}

async function removeUpload(name) {
  const res = await fetch(`/api/uploads/${encodeURIComponent(name)}`, { method: 'DELETE' });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    showNotice(el.uploadMsg, body.detail || 'Could not remove that file.', true);
    return;
  }
  showNotice(el.uploadMsg, `Removed ${name}. The index rebuilds on the next search.`);
  await refreshStatus();
}

document.addEventListener('keydown', (e) => {
  if (e.key === '/' && !isEditable(e.target)) {
    e.preventDefault();
    el.question.focus();
  }
});

el.questionForm.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    el.questionForm.requestSubmit();
  }
});

/* ---------- theme ---------- */

function applyTheme(mode) {
  document.documentElement.setAttribute('data-theme', mode);
  el.themeToggle.textContent = mode === 'dark' ? 'Light' : 'Dark';
  el.themeToggle.setAttribute('aria-pressed', String(mode === 'dark'));
}
el.themeToggle.addEventListener('click', () => {
  const next = (document.documentElement.getAttribute('data-theme') === 'dark') ? 'light' : 'dark';
  try { localStorage.setItem('sb-theme', next); } catch { /* private mode */ }
  applyTheme(next);
});

/* ---------- upload ---------- */

el.fileInput.addEventListener('change', () => {
  const file = el.fileInput.files && el.fileInput.files[0];
  el.selectedFile.textContent = file ? `${file.name} · ${fmtBytes(file.size)}` : 'No file selected';
  clearNotice(el.uploadMsg);
});

el.dropzone.addEventListener('dragover', (e) => { e.preventDefault(); el.dropzone.classList.add('dragging'); });
el.dropzone.addEventListener('dragleave', () => el.dropzone.classList.remove('dragging'));
el.dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  el.dropzone.classList.remove('dragging');
  if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
    el.fileInput.files = e.dataTransfer.files;
    el.fileInput.dispatchEvent(new Event('change'));
  }
});

el.dropzone.addEventListener('submit', async (event) => {
  event.preventDefault();
  const file = el.fileInput.files && el.fileInput.files[0];
  if (!file) return;
  clearNotice(el.uploadMsg);
  el.uploadBtn.disabled = true;
  el.uploadBtn.textContent = 'Saving…';
  setBusy('busy', 'Saving file…', 'Saving…');

  const body = new FormData();
  body.append('file', file);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(payload.detail || `upload failed (${res.status})`);
    showNotice(el.uploadMsg,
      `Saved ${payload.name}${payload.replaced ? ' (replaced existing file)' : ''}. The index rebuilds on the next search — only this file is re-embedded.`);
    el.fileInput.value = '';
    el.selectedFile.textContent = 'No file selected';
  } catch (err) {
    showNotice(el.uploadMsg, err.message || 'Upload failed.', true);
  } finally {
    el.uploadBtn.disabled = false;
    el.uploadBtn.textContent = 'Save & index';
    await refreshStatus();
  }
});

el.clearUploads.addEventListener('click', async () => {
  if (!window.confirm('Remove every uploaded file? The base corpus in data/ is untouched.')) return;
  const res = await fetch('/api/uploads/clear', { method: 'POST' });
  const payload = await res.json().catch(() => ({}));
  showNotice(el.uploadMsg, `Removed ${payload.removed ?? 0} uploaded file(s). The index updates on the next search.`);
  await refreshStatus();
});

/* ---------- search & answers ---------- */

const queryBody = (question) => ({
  question,
  top_k: Number(el.topK.value),
  mode: modeOf(),
  source: el.sourceFilter.value || null,
});

function renderHits(hits) {
  el.results.replaceChildren();
  el.emptyState.hidden = true;
  for (const hit of hits) {
    const node = el.hitTemplate.content.cloneNode(true);
    const li = node.querySelector('.hit');
    node.querySelector('.hit-rank').textContent = String(hit.rank).padStart(2, '0');
    const source = node.querySelector('.hit-source');
    source.textContent = hit.source;
    source.title = `${hit.chunk_id} · origin ${hit.origin || 'n/a'}`;
    node.querySelector('.hit-page').textContent = hit.page;

    const score = node.querySelector('.hit-score');
    const extras = [];
    if (typeof hit.bm25_score === 'number') extras.push(`bm25 ${hit.bm25_score.toFixed(2)}`);
    if (typeof hit.rerank_score === 'number') extras.push(`rerank ${hit.rerank_score.toFixed(3)}`);
    score.textContent = hit.score.toFixed(3) + (extras.length ? ` · ${extras.join(' · ')}` : '');

    node.querySelector('.hit-text').textContent = hit.text;
    const toggle = node.querySelector('.toggle-text');
    toggle.addEventListener('click', () => {
      const expanded = li.classList.toggle('expanded');
      toggle.textContent = expanded ? 'Collapse passage' : 'Expand passage';
      toggle.setAttribute('aria-expanded', String(expanded));
    });
    el.results.appendChild(node);
  }
}

async function requestJson(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const payload = await res.json().catch(() => ({}));
  return { res, payload };
}

async function handleQueryError(res, payload, noticeTarget) {
  if (res.status === 409) {
    el.results.replaceChildren();
    el.emptyState.hidden = false;
    showNotice(noticeTarget, payload.detail || 'No documents yet.', true);
    return;
  }
  throw new Error(payload.detail || `request failed (${res.status})`);
}

el.questionForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = el.question.value.trim();
  if (question.length < 3) {
    showNotice(el.askMsg, 'Ask at least three characters.', true);
    return;
  }
  clearNotice(el.askMsg);
  el.askBtn.disabled = true;
  el.askBtn.textContent = 'Searching…';
  setBusy('busy', 'Working (cold index rebuilds once)', 'Working…');
  beginProgress();

  try {
    const { res, payload } = await requestJson('/api/search', queryBody(question));
    if (!res.ok) await handleQueryError(res, payload, el.askMsg);
    else {
      renderHits(payload.hits);
      showNotice(el.askMsg,
        `${payload.hits.length} passage(s) from ${payload.documents} document(s) · ${payload.model}` +
        (el.sourceFilter.value ? ` · filtered to ${el.sourceFilter.value}` : ''));
    }
  } catch (err) {
    showNotice(el.askMsg, err.message || 'Search failed.', true);
  } finally {
    endProgress();
    el.askBtn.disabled = false;
    el.askBtn.textContent = 'Show evidence';
    await refreshStatus();
  }
});

el.allowAi.addEventListener('change', () => {
  if (el.allowAi.checked) {
    const ok = window.confirm(
      'Allow AI answers for this session?\n\n' +
      'Passages retrieved for your question will be sent to api.groq.com\n' +
      "(your GROQ_API_KEY — your quota). Full documents are never sent.");
    if (!ok) { el.allowAi.checked = false; return; }
    state.allowAI = true;
  } else {
    state.allowAI = false;
  }
  el.answerBtn.disabled = !state.allowAI;
});

function citationsText() {
  const answer = state.lastAnswer;
  if (!answer) return '';
  return (answer.sent || [])
    .map((s) => `[${s.rank}] ${s.source} — page ${s.page} (sim ${s.score.toFixed(3)})`)
    .join('\n');
}
function answerTextWithCitations() {
  const a = state.lastAnswer;
  if (!a) return '';
  return `${a.answer}\n\nSources:\n${citationsText()}`;
}
async function copyText(text) {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const area = document.createElement('textarea');
  area.value = text;
  area.setAttribute('readonly', '');
  area.style.position = 'fixed';
  area.style.opacity = '0';
  document.body.appendChild(area);
  area.select();
  document.execCommand('copy');
  area.remove();
}
async function handleCopy(which) {
  const text = which === 'answer' ? answerTextWithCitations() : citationsText();
  if (!text) return;
  try {
    await copyText(text);
    el.copyFeedback.textContent = 'Copied';
    setTimeout(() => { el.copyFeedback.textContent = ''; }, 1600);
  } catch {
    el.copyFeedback.textContent = 'Copy failed';
  }
}
el.copyAnswer.addEventListener('click', () => handleCopy('answer'));
el.copyCitations.addEventListener('click', () => handleCopy('citations'));

function renderAnswer(payload) {
  state.lastAnswer = payload;
  el.answerPanel.hidden = false;
  el.answerPanel.classList.toggle('refused', !!payload.refused);
  el.answerText.textContent = payload.answer;
  el.answerModel.textContent = payload.model || 'groq';
  el.sentList.replaceChildren();
  for (const item of payload.sent || []) {
    const li = document.createElement('li');
    li.textContent = `${item.rank}. ${item.source} · page ${item.page} · ${item.chars} chars · sim ${item.score.toFixed(3)}`;
    el.sentList.appendChild(li);
  }
}

el.answerBtn.addEventListener('click', async () => {
  const question = el.question.value.trim();
  if (question.length < 3) {
    showNotice(el.askMsg, 'Ask at least three characters before requesting an answer.', true);
    el.question.focus();
    return;
  }
  clearNotice(el.askMsg);
  el.answerBtn.disabled = true;
  el.answerBtn.textContent = 'Asking Groq…';
  setBusy('busy', 'Generating grounded answer', 'Answering…');
  beginProgress();

  try {
    const { res, payload } = await requestJson('/api/answer', { ...queryBody(question), consent: true });
    if (!res.ok) await handleQueryError(res, payload, el.askMsg);
    else renderAnswer(payload);
  } catch (err) {
    showNotice(el.askMsg, err.message || 'Could not generate an answer.', true);
  } finally {
    endProgress();
    el.answerBtn.textContent = 'Grounded answer';
    el.answerBtn.disabled = !state.allowAI;
    await refreshStatus();
  }
});

/* ---------- boot ---------- */
if (el.themeToggle) {
  const current = document.documentElement.getAttribute('data-theme') || 'light';
  el.themeToggle.textContent = current === 'dark' ? 'Light' : 'Dark';
  el.themeToggle.setAttribute('aria-pressed', String(current === 'dark'));
}
refreshStatus();
