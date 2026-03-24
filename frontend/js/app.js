/* app.js — AI PDF Agent frontend logic */
'use strict';

const API = '';  // same origin
let currentJobId = null;
let totalDone = 0;
let totalItems = 0;

// ── Init ─────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  setupDropZone();
  setupFileInput();
  loadStats();
  checkHealth();
});

// ── Section Navigation ────────────────────────────────────────
function showSection(name) {
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

  const section = document.getElementById(`section-${name}`);
  if (section) section.classList.add('active');
  const navItem = document.querySelector(`[data-section="${name}"]`);
  if (navItem) navItem.classList.add('active');

  const titles = {
    upload: ['Tải Lên & Xử Lý', 'Hỗ trợ PDF mềm, bản scan, ảnh chụp điện thoại'],
    history: ['Lịch Sử Xử Lý', 'Tất cả các file đã được xử lý'],
    docs: ['Hướng Dẫn Sử Dụng', 'Cách sử dụng AI PDF Agent'],
  };
  if (titles[name]) {
    document.getElementById('pageTitle').textContent = titles[name][0];
    document.getElementById('pageSubtitle').textContent = titles[name][1];
  }
  if (name === 'history') loadHistory();
}

// ── Health check ──────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(`${API}/api/health`);
    const data = await r.json();
    const badge = document.getElementById('apiBadge');
    if (data.status === 'ok') {
      badge.innerHTML = `<span class="dot dot-green"></span><span>Gemini Flash</span>`;
    }
  } catch (_) {}
}

// ── Stats ─────────────────────────────────────────────────────
async function loadStats() {
  try {
    const r = await fetch(`${API}/api/history?limit=100`);
    const jobs = await r.json();
    totalDone = jobs.filter(j => j.status === 'done').length;
    document.getElementById('countDone').textContent = totalDone;
  } catch (_) {}
}

// ── Drag & Drop ───────────────────────────────────────────────
function setupDropZone() {
  const zone = document.getElementById('uploadZone');
  zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('drag-over');
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) uploadFiles(files);
  });
  zone.addEventListener('click', e => {
    if (e.target.closest('.upload-btn') || e.target.closest('#fileInput')) return;
    document.getElementById('fileInput').click();
  });
}

function setupFileInput() {
  document.getElementById('fileInput').addEventListener('change', e => {
    const files = Array.from(e.target.files);
    if (files.length > 0) uploadFiles(files);
    e.target.value = '';
  });
}

// ── Upload ────────────────────────────────────────────────────
async function uploadFiles(files) {
  // Upload all files in parallel (no await) so the UI stays responsive
  await Promise.allSettled(files.map(file => uploadSingleFile(file)));
}

async function uploadSingleFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  showToast(`📤 Đang tải lên: ${file.name}`, 'info');

  try {
    const r = await fetch(`${API}/api/upload`, { method: 'POST', body: formData });
    if (!r.ok) {
      const err = await r.json();
      showToast(`❌ Lỗi: ${err.detail}`, 'error');
      return;
    }
    const data = await r.json();
    currentJobId = data.job_id;
    showToast(`✅ Đã nhận file (${data.size_mb}MB) — đang xử lý...`, 'success');

    showJobsPanel();
    addJobToList(data.job_id, file.name);
    showPipeline();

    // Prefer WebSocket / SSE; fall back to polling
    if (data.ws_url) {
      startWebSocket(data.ws_url, data.job_id);
    } else {
      startPolling(data.job_id);
    }
  } catch (e) {
    showToast(`❌ Không thể kết nối đến server: ${e.message}`, 'error');
  }
}

// ── WebSocket (real-time) ──────────────────────────────────────
function startWebSocket(wsPath, jobId) {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${proto}//${location.host}${wsPath}`;
  let ws;
  let reconnectDelay = 1000;
  let reconnectTimer;

  function connect() {
    ws = new WebSocket(wsUrl);

    ws.addEventListener('open', () => {
      console.log('[WS] Connected to', wsUrl);
      reconnectDelay = 1000;   // reset backoff on success
    });

    ws.addEventListener('message', (event) => {
      try {
        const raw = event.data;
        let msg;
        // SSE-formatted message: "data: {...}\n\n"
        if (typeof raw === 'string' && raw.startsWith('data:')) {
          const jsonStr = raw.replace(/^data:\s*/, '').trim();
          msg = JSON.parse(jsonStr);
        } else {
          msg = JSON.parse(raw);
        }
        if (msg.job_id === jobId) {
          updateJobUI(jobId, msg.status, msg.current_step, msg.progress_pct);
          if (msg.status === 'done') {
            showToast('🎉 Trích xuất hoàn thành!', 'success');
            loadResults(jobId);
            loadStats();
            closeWs();
          } else if (msg.status === 'failed') {
            showToast(`❌ Xử lý thất bại: ${msg.error || 'Lỗi không xác định'}`, 'error');
            closeWs();
          }
        }
      } catch (_) {}
    });

    ws.addEventListener('close', () => {
      // Fall back to polling if WS drops
      console.warn('[WS] Disconnected, falling back to polling');
      clearTimeout(reconnectTimer);
      startPolling(jobId);
    });

    ws.addEventListener('error', () => {
      ws.close();
    });
  }

  function closeWs() {
    clearTimeout(reconnectTimer);
    if (ws) ws.close();
  }

  connect();
}

// ── Jobs panel ────────────────────────────────────────────────
function showJobsPanel() {
  document.getElementById('jobsPanel').style.display = 'block';
  document.getElementById('pipelineCard').classList.remove('hidden');
  document.getElementById('resultsCard').classList.add('hidden');
}

function addJobToList(jobId, filename) {
  const list = document.getElementById('jobsList');
  const item = document.createElement('div');
  item.className = 'job-item';
  item.id = `job-${jobId}`;
  item.innerHTML = `
    <div class="job-file-icon">📄</div>
    <div class="job-info">
      <div class="job-name">${escHtml(filename)}</div>
      <div class="job-step" id="jobStep-${jobId}">🔍 Phân loại tài liệu...</div>
    </div>
    <div class="job-right">
      <div class="progress-ring" title="Tiến độ">
        <svg width="44" height="44" viewBox="0 0 44 44">
          <circle class="track" cx="22" cy="22" r="18"/>
          <circle class="fill" cx="22" cy="22" r="18" id="progressCircle-${jobId}"
            stroke-dasharray="113" stroke-dashoffset="113"/>
        </svg>
        <span class="progress-pct" id="progressPct-${jobId}">0%</span>
      </div>
    </div>`;
  list.prepend(item);
}

function updateJobUI(jobId, status, step, pct) {
  const stepEl = document.getElementById(`jobStep-${jobId}`);
  if (stepEl && step) stepEl.textContent = step;

  const circle = document.getElementById(`progressCircle-${jobId}`);
  const pctEl  = document.getElementById(`progressPct-${jobId}`);
  if (circle && pct != null) circle.style.strokeDashoffset = 113 - (pct / 100) * 113;
  if (pctEl  && pct != null) pctEl.textContent = pct + '%';

  // Update icon color by status
  const jobEl = document.getElementById(`job-${jobId}`);
  if (jobEl && status === 'done')   jobEl.style.borderColor = 'var(--accent)';
  if (jobEl && status === 'failed') jobEl.style.borderColor = 'var(--accent-red)';

  // Pipeline steps
  updatePipelineSteps(status);
}

function updatePipelineSteps(status) {
  const stateMap = {
    detecting:     { detect: 'active' },
    preprocessing: { detect: 'done', preprocess: 'active' },
    ocr:           { detect: 'done', preprocess: 'done', ocr: 'active' },
    extracting:    { detect: 'done', preprocess: 'done', ocr: 'done', ai: 'active' },
    done:          { detect: 'done', preprocess: 'done', ocr: 'done', ai: 'done' },
  };
  const emojiMap = { active: '⚙️', done: '✅', '': '⏸' };
  const steps = stateMap[status] || {};
  ['detect', 'preprocess', 'ocr', 'ai'].forEach(name => {
    const el = document.getElementById(`step-${name}`);
    const statusEl = document.getElementById(`stepStatus-${name}`);
    if (!el) return;
    el.className = `pipeline-step ${steps[name] || ''}`;
    if (statusEl) statusEl.textContent = emojiMap[steps[name] || ''] || '⏸';
  });
}

function showPipeline() {
  document.getElementById('pipelineCard').classList.remove('hidden');
}

// ── Polling (fallback) ──────────────────────────────────────────
function startPolling(jobId) {
  // Per-job polling interval — stored by jobId
  if (!window._pollTimers) window._pollTimers = {};
  if (window._pollTimers[jobId]) clearInterval(window._pollTimers[jobId]);
  window._pollTimers[jobId] = setInterval(() => pollStatus(jobId), 1500);
}

async function pollStatus(jobId) {
  try {
    const r = await fetch(`${API}/api/jobs/${jobId}/status`);
    if (!r.ok) return;
    const data = await r.json();
    updateJobUI(jobId, data.status, data.current_step, data.progress_pct);

    if (data.status === 'done') {
      clearInterval(window._pollTimers[jobId]);
      delete window._pollTimers[jobId];
      showToast('🎉 Trích xuất hoàn thành!', 'success');
      await loadResults(jobId);
      loadStats();
    } else if (data.status === 'failed') {
      clearInterval(window._pollTimers[jobId]);
      delete window._pollTimers[jobId];
      showToast(`❌ Xử lý thất bại: ${data.error || 'Lỗi không xác định'}`, 'error');
    }
  } catch (_) {}
}

// ── Load Results ──────────────────────────────────────────────
async function loadResults(jobId) {
  try {
    const r = await fetch(`${API}/api/results/${jobId}`);
    if (!r.ok) {
      showToast('Không thể tải kết quả', 'error');
      return;
    }
    const data = await r.json();
    renderResults(jobId, data);
    currentJobId = jobId;
  } catch (e) {
    showToast(`Lỗi: ${e.message}`, 'error');
  }
}

function renderResults(jobId, data) {
  const doc = data.result;
  const card = document.getElementById('resultsCard');
  card.classList.remove('hidden');

  // Meta tags
  const methodLabel = {
    pdfplumber: '📄 PDF Mềm', camelot: '📊 Camelot',
    ocr_paddleocr: '🔤 PaddleOCR', ocr_easyocr: '🔤 EasyOCR',
    ocr_tesseract: '🔤 Tesseract', gemini_vision: '👁️ Gemini Vision',
  };
  const typeLabel = {
    soft_pdf: '📄 PDF Mềm', scanned_pdf: '📷 PDF Scan',
    image_scan: '🖼️ Ảnh Scan', image_photo: '📱 Ảnh Điện Thoại',
    image_photo_glare: '🌟 Ảnh Lóa', image_skewed: '📐 Ảnh Nghiêng',
  };

  document.getElementById('resultsMeta').innerHTML = `
    <span class="meta-tag">${typeLabel[data.document_type] || data.document_type}</span>
    <span class="meta-tag">${methodLabel[data.extraction_method] || data.extraction_method}</span>
    <span class="meta-tag">Độ tin cậy: ${Math.round((data.overall_confidence || 0) * 100)}%</span>
    <span class="meta-tag">${doc.items.length} sản phẩm</span>`;

  // Summary grid
  const fields = [
    ['Nhà cung cấp', doc.nha_cung_cap], ['Số báo giá', doc.so_bao_gia],
    ['Ngày báo giá', doc.ngay_bao_gia], ['Khách hàng', doc.khach_hang],
    ['Đơn vị tiền', doc.don_vi_tien], ['Bảo hành', doc.bao_hanh],
  ].filter(([, v]) => v);

  document.getElementById('docSummary').innerHTML = fields.map(([k, v]) => `
    <div class="summary-field">
      <div class="summary-label">${k}</div>
      <div class="summary-value">${escHtml(String(v))}</div>
    </div>`).join('');

  // Table
  const tbody = document.getElementById('tableBody');
  tbody.innerHTML = '';
  let calculatedTotal = 0;

  doc.items.forEach((item, idx) => {
    const conf = item.confidence || 0;
    const confClass = conf >= 0.85 ? 'conf-high' : conf >= 0.6 ? 'conf-medium' : 'conf-low';
    const confLabel = Math.round(conf * 100) + '%';

    if (item.thanh_tien) calculatedTotal += item.thanh_tien;

    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${item.stt || idx + 1}</td>
      <td><code>${escHtml(item.ma_sp || '—')}</code></td>
      <td>${escHtml(item.ten_sp)}</td>
      <td>${escHtml(item.dvt || '—')}</td>
      <td class="money">${fmtNum(item.so_luong)}</td>
      <td class="money">${fmtMoney(item.don_gia, doc.don_vi_tien)}</td>
      <td class="money">${fmtMoney(item.thanh_tien, doc.don_vi_tien)}</td>
      <td style="max-width:140px;color:var(--text-secondary)">${escHtml(item.ghi_chu || '')}</td>
      <td><span class="conf-badge ${confClass}">${confLabel}</span></td>`;
    tbody.appendChild(tr);
  });

  totalItems = doc.items.length;
  document.getElementById('countItems').textContent = totalItems;

  // Totals bar
  const total = doc.tong_sau_vat || doc.tong_chua_vat || calculatedTotal;
  const vatPct = doc.thue_vat_pct || 0;
  document.getElementById('totalsBar').innerHTML = `
    ${doc.tong_chua_vat ? `<div class="total-item"><div class="total-label">Tổng chưa VAT</div><div class="total-value money">${fmtMoney(doc.tong_chua_vat, doc.don_vi_tien)}</div></div>` : ''}
    ${vatPct ? `<div class="total-item"><div class="total-label">VAT (${vatPct}%)</div><div class="total-value money">${fmtMoney(doc.thue_vat_tien, doc.don_vi_tien)}</div></div>` : ''}
    <div class="total-item"><div class="total-label">TỔNG CỘNG</div><div class="total-value highlight money">${fmtMoney(total || calculatedTotal, doc.don_vi_tien)}</div></div>`;
}

// ── Export ────────────────────────────────────────────────────
async function doExport(format) {
  if (!currentJobId) { showToast('Chưa có kết quả để export', 'error'); return; }

  if (format === 'json') {
    const r = await fetch(`${API}/api/export/${currentJobId}?format=json`);
    const blob = await r.blob();
    downloadBlob(blob, `baogia_${currentJobId}.json`);
  } else {
    const link = document.createElement('a');
    link.href = `${API}/api/export/${currentJobId}?format=${format}`;
    link.download = `baogia_${currentJobId}.${format === 'excel' ? 'xlsx' : 'csv'}`;
    link.click();
  }
  showToast(`📥 Đang tải xuống ${format.toUpperCase()}...`, 'success');
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
}

// ── History ───────────────────────────────────────────────────
async function loadHistory() {
  const list = document.getElementById('historyList');
  list.innerHTML = '<div class="empty-state"><div class="spinner"></div></div>';
  try {
    const r = await fetch(`${API}/api/history?limit=30`);
    const jobs = await r.json();
    if (!jobs.length) {
      list.innerHTML = '<div class="empty-state">Chưa có file nào được xử lý</div>';
      return;
    }
    list.innerHTML = '';
    jobs.forEach(job => {
      const confPct = Math.round((job.overall_confidence || 0) * 100);
      const statusCls = {
        done: 'status-done', failed: 'status-failed',
        pending: 'status-pending',
      }[job.status] || 'status-processing';
      const statusLabel = {
        done: '✓ Hoàn thành', failed: '✗ Thất bại',
        pending: '⏳ Chờ', preprocessing: '⚙️ Xử lý', ocr: '🔤 OCR', extracting: '🤖 AI',
      }[job.status] || job.status;

      const div = document.createElement('div');
      div.className = 'history-item';
      div.onclick = () => {
        if (job.status === 'done') {
          showSection('upload');
          loadResults(job.job_id);
        }
      };
      div.innerHTML = `
        <div class="job-file-icon">📄</div>
        <div class="history-info">
          <div class="history-name">${escHtml(job.filename)}</div>
          <div class="history-meta">
            ${job.document_type ? `${job.document_type} • ` : ''}
            ${job.status === 'done' ? `Độ tin cậy: ${confPct}% • ` : ''}
            ${formatDate(job.created_at)}
          </div>
        </div>
        <div style="display:flex;gap:8px;align-items:center">
          <span class="status-badge ${statusCls}">${statusLabel}</span>
          ${job.status === 'done' ? `<button class="btn btn-sm btn-success" onclick="event.stopPropagation();doExportFor('${job.job_id}','excel')">Excel</button>` : ''}
          <button class="btn btn-sm btn-danger" onclick="event.stopPropagation();deleteJob('${job.job_id}', this)">🗑</button>
        </div>`;
      list.appendChild(div);
    });
  } catch (e) {
    list.innerHTML = `<div class="empty-state">Lỗi tải lịch sử: ${e.message}</div>`;
  }
}

async function deleteJob(jobId, btn) {
  if (!confirm('Xóa job này?')) return;
  try {
    await fetch(`${API}/api/jobs/${jobId}`, { method: 'DELETE' });
    btn.closest('.history-item').remove();
    showToast('Đã xóa', 'success');
  } catch (e) {
    showToast('Lỗi xóa: ' + e.message, 'error');
  }
}

async function doExportFor(jobId, format) {
  const link = document.createElement('a');
  link.href = `${API}/api/export/${jobId}?format=${format}`;
  link.download = `baogia_${jobId}.xlsx`;
  link.click();
}

// ── Toast ─────────────────────────────────────────────────────
function showToast(msg, type = 'info') {
  const container = document.getElementById('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${msg}</span>`;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.animation = 'slideOut 0.3s ease forwards';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// ── Helpers ───────────────────────────────────────────────────
function escHtml(str) {
  if (!str) return '';
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function fmtNum(n) {
  if (n == null) return '—';
  return Number(n).toLocaleString('vi-VN');
}

function fmtMoney(n, currency = 'VND') {
  if (n == null) return '—';
  if (currency === 'VND') return Number(n).toLocaleString('vi-VN') + ' đ';
  return Number(n).toLocaleString('en-US', { minimumFractionDigits: 2 }) + ' ' + currency;
}

function formatDate(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
}
