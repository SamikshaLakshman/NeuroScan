/**
 * NeuraScan – Brain Tumor Detection Frontend
 * ============================================
 * Handles:
 *   • Drag-and-drop + browse file selection (multiple)
 *   • Thumbnail preview strip with individual removal
 *   • POST /predict API call (multipart/form-data)
 *   • Result cards with Grad-CAM overlay
 *   • Session history (localStorage)
 *   • Loading, error states, toast notifications
 */

'use strict';

/* ── Config ──────────────────────────────────────────────── */
const API_BASE = 'http://localhost:5000';   // Flask backend URL

/* ── DOM refs ─────────────────────────────────────────────── */
const dropzone       = document.getElementById('dropzone');
const fileInput      = document.getElementById('file-input');
const browseBtn      = document.getElementById('browse-btn');
const thumbsStrip    = document.getElementById('thumbs-strip');
const actionBar      = document.getElementById('action-bar');
const analyzeBtn     = document.getElementById('analyze-btn');
const clearBtn       = document.getElementById('clear-btn');
const btnCount       = document.getElementById('btn-count');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingCount   = document.getElementById('loading-count');
const resultsSection = document.getElementById('results-section');
const resultsGrid    = document.getElementById('results-grid');
const resultsCount   = document.getElementById('results-count');
const historySection = document.getElementById('history-section');
const historyList    = document.getElementById('history-list');
const clearHistBtn   = document.getElementById('clear-history-btn');
const toast          = document.getElementById('toast');
const toastMsg       = document.getElementById('toast-msg');
const resultCardTpl  = document.getElementById('result-card-tpl');

/* ── State ────────────────────────────────────────────────── */
let selectedFiles = [];          // File objects
let toastTimer    = null;

/* ── Tumour class meta ────────────────────────────────────── */
const CLASS_META = {
  'Glioma'          : { badge: 'TUMOR',    cls: 'badge-tumor' },
  'Meningioma'      : { badge: 'TUMOR',    cls: 'badge-tumor' },
  'Pituitary Tumor' : { badge: 'TUMOR',    cls: 'badge-tumor' },
  'No Tumor'        : { badge: 'NEGATIVE', cls: 'badge-no-tumor' },
};

/* ════════════════════════════════════════════════════════════
   FILE SELECTION
   ════════════════════════════════════════════════════════════ */

/** Merge new files into selectedFiles (avoid duplicates by name+size). */
function addFiles(incoming) {
  for (const file of incoming) {
    if (!isAllowed(file)) {
      showToast(`"${file.name}" is not a supported image type.`);
      continue;
    }
    const dupe = selectedFiles.some(f => f.name === file.name && f.size === file.size);
    if (!dupe) selectedFiles.push(file);
  }
  renderThumbs();
}

function isAllowed(file) {
  return /\.(jpe?g|png|bmp|webp)$/i.test(file.name);
}

function removeFile(idx) {
  selectedFiles.splice(idx, 1);
  renderThumbs();
}

/** Re-render the thumbnail strip from selectedFiles. */
function renderThumbs() {
  thumbsStrip.innerHTML = '';

  if (selectedFiles.length === 0) {
    thumbsStrip.hidden = true;
    actionBar.hidden   = true;
    return;
  }

  thumbsStrip.hidden = false;
  actionBar.hidden   = false;
  btnCount.textContent = selectedFiles.length;

  selectedFiles.forEach((file, idx) => {
    const url  = URL.createObjectURL(file);
    const item = document.createElement('div');
    item.className = 'thumb-item';
    item.innerHTML = `
      <img class="thumb-img" src="${url}" alt="${file.name}" />
      <button class="thumb-remove" title="Remove" aria-label="Remove ${file.name}">✕</button>
    `;
    item.querySelector('.thumb-remove').addEventListener('click', () => removeFile(idx));
    thumbsStrip.appendChild(item);
  });
}

/* ── Drag & drop ──────────────────────────────────────────── */
dropzone.addEventListener('dragover', e => {
  e.preventDefault();
  dropzone.classList.add('drag-over');
});
['dragleave', 'dragend'].forEach(ev =>
  dropzone.addEventListener(ev, () => dropzone.classList.remove('drag-over'))
);
dropzone.addEventListener('drop', e => {
  e.preventDefault();
  dropzone.classList.remove('drag-over');
  addFiles(Array.from(e.dataTransfer.files));
});

/* ── Click to browse ──────────────────────────────────────── */
dropzone.addEventListener('click', e => {
  if (e.target !== browseBtn) fileInput.click();
});
dropzone.addEventListener('keydown', e => {
  if (e.key === 'Enter' || e.key === ' ') fileInput.click();
});
browseBtn.addEventListener('click', e => {
  e.stopPropagation();
  fileInput.click();
});
fileInput.addEventListener('change', () => {
  addFiles(Array.from(fileInput.files));
  fileInput.value = '';       // reset so same file can be re-added
});

/* ── Clear ────────────────────────────────────────────────── */
clearBtn.addEventListener('click', () => {
  selectedFiles = [];
  renderThumbs();
  resultsSection.hidden = true;
});

/* ════════════════════════════════════════════════════════════
   ANALYSE – API CALL
   ════════════════════════════════════════════════════════════ */

analyzeBtn.addEventListener('click', runAnalysis);

async function runAnalysis() {
  if (selectedFiles.length === 0) return;

  // Build FormData
  const fd = new FormData();
  selectedFiles.forEach(f => fd.append('images', f));

  // UI: loading
  setLoading(true);
  loadingCount.textContent = selectedFiles.length;
  resultsSection.hidden = true;

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      body:   fd,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
      throw new Error(err.error || `Server returned ${res.status}`);
    }

    const data = await res.json();

    if (data.errors) {
      data.errors.forEach(e => showToast(`${e.filename}: ${e.error}`));
    }

    if (data.results && data.results.length > 0) {
      renderResults(data.results);
      saveToHistory(data.results);
    } else {
      showToast('No results returned from server.');
    }

  } catch (err) {
    if (err.name === 'TypeError') {
      showToast('Cannot reach the backend. Is Flask running on port 5000?');
    } else {
      showToast(err.message);
    }
  } finally {
    setLoading(false);
  }
}

/* ════════════════════════════════════════════════════════════
   RENDER RESULTS
   ════════════════════════════════════════════════════════════ */

function renderResults(results) {
  resultsGrid.innerHTML = '';
  resultsCount.textContent = `${results.length} scan${results.length > 1 ? 's' : ''}`;

  results.forEach((r, i) => {
    const card = buildResultCard(r, i);
    resultsGrid.appendChild(card);
  });

  resultsSection.hidden = false;
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function buildResultCard(result, delay = 0) {
  const clone  = resultCardTpl.content.cloneNode(true);
  const card   = clone.querySelector('.result-card');

  // Stagger animation
  card.style.animationDelay = `${delay * 80}ms`;

  // Images
  const imgOrig   = card.querySelector('[data-type="original"]');
  const imgGradcam = card.querySelector('[data-type="gradcam"]');
  imgOrig.src     = result.original_image;
  imgGradcam.src  = result.gradcam_overlay;

  // Filename
  card.querySelector('.card-filename').textContent = result.filename;

  // Verdict
  const meta = CLASS_META[result.predicted] || { badge: 'UNKNOWN', cls: 'badge-tumor' };
  card.querySelector('.verdict-label').textContent = result.predicted;
  const badge = card.querySelector('.verdict-badge');
  badge.textContent = meta.badge;
  badge.className   = `verdict-badge ${meta.cls}`;

  // Confidence bar
  const pct = Math.round(result.confidence * 100);
  const fill = card.querySelector('.confidence-bar-fill');
  const pctEl = card.querySelector('.confidence-pct');
  pctEl.textContent = `${pct}%`;
  // Trigger animation after paint
  requestAnimationFrame(() => {
    requestAnimationFrame(() => { fill.style.width = `${pct}%`; });
  });

  // Probability breakdown
  const breakdownEl = card.querySelector('.prob-breakdown');
  if (result.probabilities) {
    const sorted = Object.entries(result.probabilities)
      .sort(([,a],[,b]) => b - a);
    const top = sorted[0][0];     // highest class

    sorted.forEach(([cls, prob]) => {
      const row = document.createElement('div');
      row.className = 'prob-row';
      const pctRaw = Math.round(prob * 100);
      const isTop  = cls === top;
      row.innerHTML = `
        <span class="prob-class">${cls}</span>
        <div class="prob-track">
          <div class="prob-fill ${isTop ? 'top' : ''}" style="width:0%"></div>
        </div>
        <span class="prob-val">${pctRaw}%</span>
      `;
      breakdownEl.appendChild(row);
      // Animate
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          row.querySelector('.prob-fill').style.width = `${pctRaw}%`;
        });
      });
    });
  }

  return card;
}

/* ════════════════════════════════════════════════════════════
   SESSION HISTORY  (localStorage)
   ════════════════════════════════════════════════════════════ */

const HISTORY_KEY = 'neurascan_history';

function loadHistory() {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
  catch { return []; }
}
function saveHistory(history) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 50)));
}

function saveToHistory(results) {
  const history = loadHistory();
  const ts = new Date().toLocaleTimeString();
  results.forEach(r => {
    history.unshift({
      filename  : r.filename,
      predicted : r.predicted,
      confidence: r.confidence,
      thumbnail : r.original_image,
      ts,
    });
  });
  saveHistory(history);
  renderHistory();
}

function renderHistory() {
  const history = loadHistory();
  if (history.length === 0) {
    historySection.hidden = true;
    return;
  }
  historySection.hidden = false;
  historyList.innerHTML = '';

  history.forEach(entry => {
    const item = document.createElement('div');
    item.className = 'history-item';
    item.innerHTML = `
      <img class="history-thumb" src="${entry.thumbnail}" alt="${entry.filename}" />
      <div class="history-info">
        <p class="history-fname">${entry.filename} · ${entry.ts}</p>
        <p class="history-result">${entry.predicted}</p>
      </div>
      <span class="history-conf">${Math.round(entry.confidence * 100)}%</span>
    `;
    historyList.appendChild(item);
  });
}

clearHistBtn.addEventListener('click', () => {
  localStorage.removeItem(HISTORY_KEY);
  historySection.hidden = true;
});

/* ════════════════════════════════════════════════════════════
   UI HELPERS
   ════════════════════════════════════════════════════════════ */

function setLoading(on) {
  loadingOverlay.hidden = !on;
  analyzeBtn.disabled   = on;
  if (on) loadingOverlay.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

let toastHideTimer = null;
function showToast(msg, duration = 4500) {
  clearTimeout(toastHideTimer);
  toastMsg.textContent = msg;
  toast.hidden = false;
  toastHideTimer = setTimeout(() => { toast.hidden = true; }, duration);
}

/* ── Init ─────────────────────────────────────────────────── */
renderHistory();

// Health check on load (non-blocking)
fetch(`${API_BASE}/health`)
  .then(r => r.json())
  .then(d => console.log('[NeuraScan] Backend ready:', d))
  .catch(() => console.warn('[NeuraScan] Backend not reachable. Start Flask first.'));
