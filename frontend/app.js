/**
 * app.js — Industrial OCR Monitor Frontend Logic
 *
 * Responsibilities:
 * - File selection preview (shows uploaded image immediately).
 * - POST /api/upload — runs OCR pipeline via the backend.
 * - Displays detected text, length, confidence bar, timestamp, status.
 * - Fetches and displays the annotated detection image.
 * - Loads history from GET /api/history.
 * - Clears history via DELETE /api/history.
 * - Polling: refreshes the latest result every 5 seconds (Phase 1 — no WS yet).
 *
 * Phase 3 note: The polling block at the bottom will be replaced with a
 * WebSocket connection once the /ws endpoint is implemented.
 */

'use strict';

// ── API base URL (same origin when served by FastAPI) ────────────────────
const API = '/api';

// ── DOM references ───────────────────────────────────────────────────────
const fileInput        = document.getElementById('fileInput');
const processBtn       = document.getElementById('processBtn');
const spinner          = document.getElementById('spinner');
const uploadedImage    = document.getElementById('uploadedImage');
const detectionImage   = document.getElementById('detectionImage');
const detPlaceholder   = document.getElementById('detPlaceholder');
const uploadedFilename = document.getElementById('uploadedFilename');
const detectionBadge   = document.getElementById('detectionBadge');
const detectedText     = document.getElementById('detectedText');
const textLength       = document.getElementById('textLength');
const confidenceBar    = document.getElementById('confidenceBar');
const confidenceValue  = document.getElementById('confidenceValue');
const resultTimestamp  = document.getElementById('resultTimestamp');
const pipelineStatus   = document.getElementById('pipelineStatus');
const historyList      = document.getElementById('historyList');
const refreshHistory   = document.getElementById('refreshHistory');
const clearHistory     = document.getElementById('clearHistory');
const statusDot        = document.getElementById('statusDot');
const statusText       = document.getElementById('statusText');

// ── State ────────────────────────────────────────────────────────────────
let selectedFile = null;

// ── File selection ───────────────────────────────────────────────────────
fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;

  selectedFile = file;
  uploadedFilename.textContent = file.name;
  processBtn.disabled = false;

  // Show preview immediately
  const reader = new FileReader();
  reader.onload = e => {
    uploadedImage.src = e.target.result;
    uploadedImage.classList.remove('hidden');
    document.querySelector('#uploadedImageBox .placeholder').style.display = 'none';
  };
  reader.readAsDataURL(file);

  // Reset detection panel
  detectionImage.classList.add('hidden');
  detPlaceholder.style.display = '';
  resetResultPanel();
});

// ── Run OCR ──────────────────────────────────────────────────────────────
processBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  setStatus('busy', 'Processing…');
  processBtn.disabled = true;
  spinner.classList.remove('hidden');
  detectionImage.classList.add('hidden');
  detPlaceholder.style.display = '';

  const formData = new FormData();
  formData.append('file', selectedFile);

  try {
    const resp = await fetch(`${API}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }

    const result = await resp.json();
    renderResult(result);
    loadDetectionImage(result.filename);
    loadHistory();
    setStatus('online', 'Done');

  } catch (e) {
    console.error('Upload error:', e);
    setStatus('error', 'Error');
    pipelineStatus.textContent = 'error';
    pipelineStatus.className = 'field-value status-badge error';
    alert(`OCR failed: ${e.message}`);
  } finally {
    processBtn.disabled = false;
    spinner.classList.add('hidden');
  }
});

// ── Render result panel ──────────────────────────────────────────────────
function renderResult(r) {
  detectedText.textContent   = r.text   || '(no text detected)';
  textLength.textContent     = r.length  ?? '—';
  confidenceValue.textContent = r.confidence
    ? `${(r.confidence * 100).toFixed(1)}%`
    : '—';
  confidenceBar.style.width  = r.confidence
    ? `${Math.min(r.confidence * 100, 100)}%`
    : '0%';
  resultTimestamp.textContent = r.timestamp
    ? new Date(r.timestamp).toLocaleString()
    : '—';

  const s = r.status || 'unknown';
  pipelineStatus.textContent = s;
  pipelineStatus.className   = `field-value status-badge ${s}`;

  detectionBadge.textContent = r.status === 'success'
    ? `✓ ${r.length} chars`
    : r.status;
}

function resetResultPanel() {
  detectedText.textContent    = '—';
  textLength.textContent      = '—';
  confidenceBar.style.width   = '0%';
  confidenceValue.textContent = '—';
  resultTimestamp.textContent = '—';
  pipelineStatus.textContent  = '—';
  pipelineStatus.className    = 'field-value status-badge';
  detectionBadge.textContent  = '—';
}

// ── Load annotated detection image ───────────────────────────────────────
function loadDetectionImage(filename) {
  // Backend serves annotated images at /api/images/result/<stem>_det.jpg
  const stem    = filename.replace(/\.[^.]+$/, '');  // strip extension if present
  const imgUrl  = `${API}/images/result/${stem}_det.jpg?t=${Date.now()}`;
  detectionImage.src = imgUrl;
  detectionImage.onload = () => {
    detectionImage.classList.remove('hidden');
    detPlaceholder.style.display = 'none';
  };
  detectionImage.onerror = () => {
    // Annotated image may not exist yet — keep placeholder
    detectionImage.classList.add('hidden');
    detPlaceholder.style.display = '';
  };
}

// ── History ──────────────────────────────────────────────────────────────
async function loadHistory() {
  try {
    const resp = await fetch(`${API}/history?limit=20`);
    if (!resp.ok) return;
    const data = await resp.json();
    renderHistory(data.results);
  } catch (e) {
    console.warn('History load failed:', e);
  }
}

function renderHistory(items) {
  if (!items || items.length === 0) {
    historyList.innerHTML = '<li class="history-empty">No history yet</li>';
    return;
  }
  historyList.innerHTML = items.map(item => `
    <li class="history-item" data-filename="${item.filename}" title="${item.text}">
      <div class="history-filename">${escHtml(item.filename)}</div>
      <div class="history-text">${escHtml(item.text) || '<em style="opacity:.5">no text</em>'}</div>
      <div class="history-meta">
        conf: ${item.confidence ? (item.confidence * 100).toFixed(1) + '%' : '—'}
        &nbsp;·&nbsp;
        ${item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : ''}
      </div>
    </li>
  `).join('');

  // Click a history item to re-load its detection image
  historyList.querySelectorAll('.history-item').forEach(li => {
    li.addEventListener('click', () => {
      const fname = li.dataset.filename;
      loadDetectionImage(fname);
    });
  });
}

refreshHistory.addEventListener('click', loadHistory);

clearHistory.addEventListener('click', async () => {
  if (!confirm('Delete all OCR history?')) return;
  try {
    const resp = await fetch(`${API}/history`, { method: 'DELETE' });
    const data = await resp.json();
    alert(data.message);
    loadHistory();
  } catch (e) {
    alert('Failed to clear history.');
  }
});

// ── Status helpers ───────────────────────────────────────────────────────
function setStatus(state, text) {
  statusDot.className = `status-dot ${state}`;
  statusText.textContent = text;
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// ── Polling (Phase 1 — replace with WebSocket in Phase 3) ────────────────
async function pollLatest() {
  try {
    const resp = await fetch(`${API}/latest`);
    if (resp.status === 404) return;   // no results yet
    if (!resp.ok) return;
    // Only update if nothing is being processed (spinner hidden)
    if (spinner.classList.contains('hidden')) {
      // Don't auto-refresh the panel if a local result is already shown
    }
  } catch (_) { /* network error — ignore silently */ }
}

// ── Init ─────────────────────────────────────────────────────────────────
(function init() {
  setStatus('online', 'Ready');
  loadHistory();
  setInterval(pollLatest, 5000);
})();
