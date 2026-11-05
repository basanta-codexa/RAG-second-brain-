/* Second Brain — Document Test Bench (vanilla JS, no frameworks) */
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = { documents: 0 };

const el = {
  healthDot: $('#health-dot'),
  healthLabel: $('#health-label'),
  indexBadge: $('#index-badge'),
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
  askBtn: $('#ask-btn'),
  askMsg: $('#ask-message'),
  results: $('#results'),
  emptyState: $('#empty-state'),
  hitTemplate: $('#hit-template'),
};

const fmtBytes = (n) =>
  n >= 1 << 20 ? `${(n / (1 << 20)).toFixed(1)} MB`
    : n >= 1024 ? `${Math.round(n / 1024)} KB`
                : `${n} B`;

function showNotice(target, text, isError = false) {
  target.hidden = false;
  target.textContent = text;
  target.classList.toggle('error', isError);
}

function clearNotice(target) { target.hidden = true; target.textContent = ''; }

function setBusy(dot, label, busyText) {
  el.healthDot.classList.toggle('busy', dot === 'busy');
  el.healthDot.classList.toggle('ok', dot === 'ok');
  el.healthLabel.textContent = busyText;
  el.indexBadge.className = 'workbench-status' + (dot === 'busy' ? ' busy' : dot === 'ok' ? ' ready' : '');
  el.indexBadge.textContent = label;
}

async function refreshStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) throw new Error(`status ${res.status}`);
    const s = await res.json();
    state.documents = s.documents;

    renderFileList(el.uploadList, s.uploads, true);
    renderFileList(el.baseList, s.base, false);
    el.clearUploads.disabled = s.uploads.length === 0;

    if (s.documents > 0) {
      setBusy('ok', `Ready · ${s.documents} document${s.documents === 1 ? '' : 's'} · ${s.model}`, 'Service up');
    } else {
      setBusy('idle', 'Waiting for documents', 'Service up');
    }
    return s;
  } catch (err) {
    el.healthDot.classList.remove('ok', 'busy');
    el.healthLabel.textContent = 'Status unknown';
    el.indexBadge.className = 'workbench-status';
    el.indexBadge.textContent = 'Status unknown';
    return null;
  }
}

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
      `Saved ${payload.name}${payload.replaced ? ' (replaced existing file)' : ''}. Ask a question — the index rebuilds automatically.`);
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
  showNotice(el.uploadMsg, `Removed ${payload.removed ?? 0} uploaded file(s).`);
  await refreshStatus();
});

/* ---------- search ---------- */

function renderHits(hits) {
  el.results.replaceChildren();
  el.emptyState.hidden = true;
  for (const hit of hits) {
    const node = el.hitTemplate.content.cloneNode(true);
    const li = node.querySelector('.hit');
    node.querySelector('.hit-rank').textContent = String(hit.rank).padStart(2, '0');
    node.querySelector('.hit-source').textContent = hit.source;
    node.querySelector('.hit-source').title = hit.chunk_id;
    node.querySelector('.hit-page').textContent = hit.page;
    node.querySelector('.hit-score').textContent = hit.score.toFixed(3);
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

el.questionForm.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = el.question.value.trim();
  if (question.length < 3) { showNotice(el.askMsg, 'Ask at least three characters.', true); return; }

  clearNotice(el.askMsg);
  el.askBtn.disabled = true;
  el.askBtn.textContent = 'Searching…';
  setBusy('busy', 'Embedding query (first run rebuilds the index)', 'Working…');

  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, top_k: Number(el.topK.value) }),
    });
    const payload = await res.json().catch(() => ({}));
    if (res.status === 409) {
      el.results.replaceChildren();
      el.emptyState.hidden = false;
      showNotice(el.askMsg, payload.detail || 'No documents yet.', true);
      return;
    }
    if (!res.ok) throw new Error(payload.detail || `search failed (${res.status})`);

    renderHits(payload.hits);
    showNotice(el.askMsg,
      `${payload.hits.length} passage(s) from ${payload.documents} document(s) · index ${payload.cached_index ? 'reused from cache' : 'rebuilt'} · ${payload.model}`);
  } catch (err) {
    showNotice(el.askMsg, err.message || 'Search failed.', true);
  } finally {
    el.askBtn.disabled = false;
    el.askBtn.textContent = 'Show evidence';
    await refreshStatus();
  }
});

/* ---------- boot ---------- */

refreshStatus();
