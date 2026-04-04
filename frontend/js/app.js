/* app.js — AI PDF Agent frontend logic */
'use strict';

const API = '';  // same origin
let currentJobId = null;
let totalDone = 0;
let totalItems = 0;

// Stage audit state — reset per job
let _audit = { raw: null, merge: null, deduped: 0, pass2: null };

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
  let wsClosed = false;

  // Safety: if WebSocket fails to connect within 3s, fall back to polling
  const connectTimer = setTimeout(() => {
    if (ws && ws.readyState !== WebSocket.OPEN) {
      ws.close();
    }
  }, 3000);

  ws = new WebSocket(wsUrl);

  ws.addEventListener('open', () => {
    console.log('[WS] Connected to', wsUrl);
    clearTimeout(connectTimer);
  });

  ws.addEventListener('message', (event) => {
    try {
      const raw = event.data;
      let msg;
      // SSE-formatted: "data: {...}\n\n"
      if (typeof raw === 'string' && raw.startsWith('data:')) {
        const jsonStr = raw.replace(/^data:\s*/, '').trim();
        msg = JSON.parse(jsonStr);
      } else {
        msg = JSON.parse(raw);
      }
      if (msg.job_id === jobId) {
        updateJobUI(jobId, msg.status, msg.current_step, msg.progress_pct);
        if (msg.status === 'done') {
          wsClosed = true;
          ws.close();
          showToast('🎉 Trích xuất hoàn thành!', 'success');
          loadResults(jobId);
          loadStats();
          stopPolling(jobId);
        } else if (msg.status === 'failed') {
          wsClosed = true;
          ws.close();
          stopPolling(jobId);
          showToast(`❌ Xử lý thất bại: ${msg.error || 'Lỗi không xác định'}`, 'error');
        }
      }
    } catch (_) {}
  });

  ws.addEventListener('close', () => {
    clearTimeout(connectTimer);
    if (!wsClosed) {
      // WS dropped unexpectedly — fall back to polling
      console.warn('[WS] Disconnected, falling back to polling');
      startPolling(jobId);
    }
  });

  ws.addEventListener('error', () => {
    ws.close();
  });
}

// ── Jobs panel ────────────────────────────────────────────────
function showJobsPanel() {
  _detectedPath = null; // reset path detection for new job
  _audit = { raw: null, merge: null, deduped: 0, pass2: null }; // reset audit state
  const auditBar = document.getElementById('auditBar');
  if (auditBar) auditBar.classList.add('hidden');
  document.getElementById('jobsPanel').style.display = 'block';
  document.getElementById('pipelineCard').classList.remove('hidden');
  document.getElementById('resultsCard').classList.add('hidden');
  // Reset step descriptions to defaults for new job
  const preDesc = document.querySelector('#step-preprocess .step-desc');
  const ocrDesc = document.querySelector('#step-ocr .step-desc');
  const aiDesc  = document.querySelector('#step-ai .step-desc');
  if (preDesc) preDesc.textContent = 'Glare / Deskew / Deblur';
  if (ocrDesc) ocrDesc.textContent = 'PaddleOCR → EasyOCR → Gemini';
  if (aiDesc)  aiDesc.textContent  = 'Gemini Flash phân tích';
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

// Detected processing path — updated live from step messages
let _detectedPath = null; // 'soft_pdf' | 'scanned_pdf' | 'image'

function updateJobUI(jobId, status, step, pct) {
  const stepEl = document.getElementById(`jobStep-${jobId}`);
  if (stepEl && step) stepEl.textContent = step;

  const circle = document.getElementById(`progressCircle-${jobId}`);
  const pctEl  = document.getElementById(`progressPct-${jobId}`);
  if (circle && pct != null) circle.style.strokeDashoffset = 113 - (pct / 100) * 113;
  if (pctEl  && pct != null) pctEl.textContent = pct + '%';

  const jobEl = document.getElementById(`job-${jobId}`);
  if (jobEl && status === 'done')   jobEl.style.borderColor = 'var(--accent)';
  if (jobEl && status === 'failed') jobEl.style.borderColor = 'var(--accent-red)';

  updatePipelineSteps(status, step);
}

function updatePipelineSteps(status, step) {
  // ── Detect path from step message ────────────────────────────
  if (step) {
    if (step.includes('PDF sang ảnh') || step.includes('Đang đọc trang')) {
      _detectedPath = 'scanned_pdf';
    } else if (step.includes('Đang đọc text') || step.includes('trích xuất bảng')) {
      _detectedPath = 'soft_pdf';
    } else if (step.includes('tiền xử lý') || step.includes('nhận dạng văn bản')) {
      _detectedPath = 'image';
    }
  }

  // ── Update step descriptions based on detected path ──────────
  const preDesc = document.querySelector('#step-preprocess .step-desc');
  const ocrDesc = document.querySelector('#step-ocr .step-desc');
  const aiDesc  = document.querySelector('#step-ai .step-desc');
  if (_detectedPath === 'scanned_pdf') {
    if (preDesc) preDesc.textContent = 'PDF → ảnh JPEG (200 DPI)';
    if (ocrDesc) ocrDesc.textContent = 'Gemini Vision / trang';
    if (aiDesc)  aiDesc.textContent  = 'Hoàn thành trong bước Vision';
  } else if (_detectedPath === 'soft_pdf') {
    if (preDesc) preDesc.textContent = 'Đọc text + bảng pdfplumber';
    if (ocrDesc) ocrDesc.textContent = 'Markdown table → Gemini Flash';
    if (aiDesc)  aiDesc.textContent  = 'Gemini Flash phân tích';
  } else if (_detectedPath === 'image') {
    if (preDesc) preDesc.textContent = 'Glare / Deskew / Deblur';
    if (ocrDesc) ocrDesc.textContent = 'PaddleOCR → EasyOCR → Gemini';
    if (aiDesc)  aiDesc.textContent  = 'Gemini Flash phân tích';
  }

  // ── Parse audit counts from step message ─────────────────────
  if (step) {
    // "🔗 Đã đọc N trang — M dòng thô"
    const rawMatch = step.match(/(\d+)\s*dòng thô/);
    if (rawMatch) {
      _audit.raw = parseInt(rawMatch[1], 10);
      _updateAuditBar();
    }
    // "🔗 Ghép trang → N sản phẩm (loại D trùng)"
    const mergeMatch = step.match(/Ghép trang.*?→\s*(\d+)\s*sản phẩm/);
    if (mergeMatch) {
      _audit.merge = parseInt(mergeMatch[1], 10);
      const dedupMatch = step.match(/loại\s*(\d+)\s*trùng/);
      _audit.deduped = dedupMatch ? parseInt(dedupMatch[1], 10) : 0;
      _updateAuditBar();
    }
    // "✅ Pass 2 xong — N sản phẩm"
    const pass2Match = step.match(/Pass 2 xong.*?(\d+)\s*sản phẩm/);
    if (pass2Match) {
      _audit.pass2 = parseInt(pass2Match[1], 10);
      _updateAuditBar();
    }
  }

  // ── Step state (active / done) ────────────────────────────────
  // pass2 step activates when step text mentions "Pass 2"
  const isPass2Active = step && step.includes('Pass 2') && !step.includes('Pass 2 xong');
  const isPass2Done   = step && step.includes('Pass 2 xong');

  const stateMap = {
    detecting:     { detect: 'active' },
    preprocessing: { detect: 'done', preprocess: 'active' },
    ocr:           { detect: 'done', preprocess: 'done', ocr: 'active' },
    extracting:    { detect: 'done', preprocess: 'done', ocr: 'done', ai: 'done',
                     pass2: isPass2Done ? 'done' : isPass2Active ? 'active' : '' },
    done:          { detect: 'done', preprocess: 'done', ocr: 'done', ai: 'done', pass2: 'done' },
  };
  const emojiMap = { active: '⚙️', done: '✅', '': '⏸' };
  const steps = stateMap[status] || {};

  // ai step: stays active until pass2 takes over
  if (status === 'extracting' && !isPass2Active && !isPass2Done) {
    steps.ai = 'active';
  }

  ['detect', 'preprocess', 'ocr', 'ai', 'pass2'].forEach(name => {
    const el = document.getElementById(`step-${name}`);
    const statusEl = document.getElementById(`stepStatus-${name}`);
    if (!el) return;
    el.className = `pipeline-step ${steps[name] || ''}`;
    if (statusEl) statusEl.textContent = emojiMap[steps[name] || ''] || '⏸';
  });
}

function _updateAuditBar() {
  const bar = document.getElementById('auditBar');
  if (!bar) return;
  bar.classList.remove('hidden');

  const rawEl    = document.getElementById('auditRaw');
  const mergeEl  = document.getElementById('auditMerge');
  const pass2El  = document.getElementById('auditPass2');

  if (_audit.raw != null && rawEl) {
    rawEl.innerHTML = `🔍 Thô: <strong>${_audit.raw}</strong> dòng`;
  }
  if (_audit.merge != null && mergeEl) {
    const dedupTxt = _audit.deduped > 0 ? ` <span style="color:var(--accent-yellow,#d29922)">(−${_audit.deduped} trùng)</span>` : '';
    mergeEl.innerHTML = `🔗 Sau ghép: <strong>${_audit.merge}</strong>${dedupTxt}`;
    if (_audit.deduped > 0) mergeEl.classList.add('has-dedup');
  }
  if (_audit.pass2 != null && pass2El) {
    pass2El.innerHTML = `✅ Sau Pass 2: <strong>${_audit.pass2}</strong>`;
    pass2El.classList.add('pass2-done');
  }
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

function stopPolling(jobId) {
  if (!window._pollTimers || !window._pollTimers[jobId]) return;
  clearInterval(window._pollTimers[jobId]);
  delete window._pollTimers[jobId];
}

async function pollStatus(jobId) {
  try {
    const r = await fetch(`${API}/api/jobs/${jobId}/status`);
    if (!r.ok) return;
    const data = await r.json();
    updateJobUI(jobId, data.status, data.current_step, data.progress_pct);

    if (data.status === 'done') {
      stopPolling(jobId);
      showToast('🎉 Trích xuất hoàn thành!', 'success');
      await loadResults(jobId);
      loadStats();
    } else if (data.status === 'failed') {
      stopPolling(jobId);
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

  const hasDvt2        = doc.items.some(i => i.dvt_2 || i.don_gia_2 != null);
  const hasVatPct      = doc.items.some(i => i.vat_pct != null);
  const hasQuiCach     = doc.items.some(i => i.qui_cach);
  const hasDonGiaCoVat = doc.items.some(i => i.don_gia_co_vat != null);

  // ── Meta tags ───────────────────────────────────────────────────
  const methodLabel = {
    pdfplumber: '📄 pdfplumber', camelot: '📊 Camelot',
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

  // ── Summary grid ────────────────────────────────────────────────
  const vatInfo = doc.gia_da_bao_gom_vat ? 'Đã bao gồm VAT' : 'Chưa bao gồm VAT';
  const fields = [
    ['Nhà cung cấp',   doc.nha_cung_cap],
    ['Ngày hiệu lực',  doc.ngay_hieu_luc],
    ['Đơn vị tiền',    doc.don_vi_tien],
    ['Giá VAT',        vatInfo],
  ].filter(([, v]) => v);

  document.getElementById('docSummary').innerHTML = fields.map(([k, v]) => `
    <div class="summary-field">
      <div class="summary-label">${k}</div>
      <div class="summary-value">${escHtml(String(v))}</div>
    </div>`).join('');

  // ── Update pipeline step descriptions by document type ──────────
  const docType = data.document_type;
  const preDesc = document.querySelector('#step-preprocess .step-desc');
  const ocrDesc = document.querySelector('#step-ocr .step-desc');
  if (preDesc && ocrDesc) {
    if (docType === 'soft_pdf') {
      preDesc.textContent = 'Đọc text + bảng pdfplumber';
      ocrDesc.textContent = 'Markdown table → Gemini Flash';
    } else if (docType === 'scanned_pdf') {
      preDesc.textContent = 'PDF → ảnh JPEG (200 DPI)';
      ocrDesc.textContent = 'Gemini Vision / trang';
    } else {
      preDesc.textContent = 'Glare / Deskew / Deblur';
      ocrDesc.textContent = 'PaddleOCR → EasyOCR → Gemini';
    }
  }

  // ── Build table header ──────────────────────────────────────────
  const table = document.getElementById('dataTable');
  const thead = table.querySelector('thead');
  const headers = ['STT', 'Mã SP', 'Tên sản phẩm'];
  if (hasQuiCach)     { headers.push('Quy cách'); }
  headers.push('ĐVT', 'Đơn giá chưa VAT');
  if (hasVatPct)      { headers.push('VAT %'); }
  if (hasDonGiaCoVat) { headers.push('Đơn giá có VAT'); }
  if (hasDvt2)        { headers.push('ĐVT 2', 'Đơn giá 2'); }
  headers.push('Ghi chú', 'Tin cậy');
  thead.innerHTML = `<tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>`;
  const colCount = headers.length;

  // ── Build table body ────────────────────────────────────────────
  const tbody = document.getElementById('tableBody');
  tbody.innerHTML = '';

  if (!doc.items || doc.items.length === 0) {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td colspan="${colCount}" style="text-align:center;color:var(--text-secondary);padding:32px">
      ⚠️ Không tìm thấy sản phẩm nào trong file này.<br>
      <small>Vui lòng thử file báo giá/bảng giá có cấu trúc bảng.</small>
    </td>`;
    tbody.appendChild(tr);
  } else {
    let lastNhomSp = null;
    doc.items.forEach((item, idx) => {
      // Group header row when nhom_sp changes
      if (item.nhom_sp && item.nhom_sp !== lastNhomSp) {
        lastNhomSp = item.nhom_sp;
        const groupTr = document.createElement('tr');
        groupTr.className = 'group-header-row';
        groupTr.innerHTML = `<td colspan="${colCount}" class="group-header-cell">${escHtml(item.nhom_sp)}</td>`;
        tbody.appendChild(groupTr);
      }

      const conf = item.confidence || 0;
      const confClass = conf >= 0.85 ? 'conf-high' : conf >= 0.6 ? 'conf-medium' : 'conf-low';
      const hasWarning = item.ghi_chu && item.ghi_chu.includes('⚠️');

      const tr = document.createElement('tr');
      if (hasWarning) tr.classList.add('row-warning');

      let cells = `
        <td>${item.stt || idx + 1}</td>
        <td><code>${escHtml(item.ma_sp || '—')}</code></td>
        <td>${escHtml(item.ten_sp)}</td>`;

      if (hasQuiCach) cells += `
        <td>${escHtml(item.qui_cach || '—')}</td>`;

      cells += `
        <td>${escHtml(item.dvt || '—')}</td>
        <td class="money">${fmtMoney(item.don_gia, doc.don_vi_tien)}</td>`;

      if (hasVatPct) cells += `
        <td class="money">${item.vat_pct != null ? item.vat_pct + '%' : '—'}</td>`;

      if (hasDonGiaCoVat) cells += `
        <td class="money">${fmtMoney(item.don_gia_co_vat, doc.don_vi_tien)}</td>`;

      if (hasDvt2) cells += `
        <td>${escHtml(item.dvt_2 || '—')}</td>
        <td class="money">${fmtMoney(item.don_gia_2, doc.don_vi_tien)}</td>`;

      const noteHtml = hasWarning
        ? `<span class="warning-note">${escHtml(item.ghi_chu)}</span>`
        : escHtml(item.ghi_chu || '');

      cells += `
        <td class="note-cell">${noteHtml}</td>
        <td><span class="conf-badge ${confClass}">${Math.round(conf * 100)}%</span></td>`;

      tr.innerHTML = cells;
      tbody.appendChild(tr);
    });
  }

  totalItems = doc.items.length;
  document.getElementById('countItems').textContent = totalItems;

  // ── Pipeline audit summary ──────────────────────────────────────
  const auditDiv = document.getElementById('resultAudit');
  if (auditDiv) {
    const corrected = doc.items.filter(i => i.ghi_chu && i.ghi_chu.includes('[Pass2:')).length;
    const warned    = doc.items.filter(i => i.ghi_chu && i.ghi_chu.includes('⚠️')).length;
    const noQC      = doc.items.filter(i => !i.qui_cach).length;
    const noDVT     = doc.items.filter(i => !i.dvt).length;

    const chips = [
      { label: 'Tổng sản phẩm', num: doc.items.length, cls: 'chip-ok' },
    ];
    if (_audit.raw != null)    chips.push({ label: 'Dòng thô', num: _audit.raw });
    if (_audit.deduped > 0)    chips.push({ label: 'Trùng loại bỏ', num: _audit.deduped, cls: 'chip-warn' });
    if (corrected > 0)         chips.push({ label: 'Pass 2 chỉnh sửa', num: corrected, cls: 'chip-ok' });
    if (warned > 0)            chips.push({ label: 'Cảnh báo', num: warned, cls: 'chip-warn' });
    if (noQC > 0)              chips.push({ label: 'Thiếu qui cách', num: noQC, cls: 'chip-warn' });
    if (noDVT > 0)             chips.push({ label: 'Thiếu ĐVT', num: noDVT, cls: 'chip-warn' });

    auditDiv.innerHTML = chips.map(c => `
      <span class="result-audit-chip ${c.cls || ''}">
        <span class="chip-num">${c.num}</span> ${c.label}
      </span>`).join('');
    auditDiv.classList.remove('hidden');
  }

  // ── Totals bar ──────────────────────────────────────────────────
  const totalsBar = document.getElementById('totalsBar');
  const vatStatus = doc.gia_da_bao_gom_vat ? '✓ Đã bao gồm VAT' : '✗ Chưa bao gồm VAT';
  totalsBar.innerHTML = `
    <div class="total-item"><div class="total-label">Giá VAT</div><div class="total-value">${vatStatus}</div></div>`;
}

// ── Export ────────────────────────────────────────────────────
async function doExport(format) {
  if (!currentJobId) { showToast('Chưa có kết quả để export', 'error'); return; }

  const ext = format === 'excel' ? 'xlsx' : format === 'csv' ? 'csv' : 'json';
  const filename = `baogia_${currentJobId}.${ext}`;

  try {
    const r = await fetch(`${API}/api/export/${currentJobId}?format=${format}`);
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: 'Lỗi không xác định' }));
      showToast(`❌ Export thất bại: ${err.detail}`, 'error');
      return;
    }
    const blob = await r.blob();
    downloadBlob(blob, filename);
    showToast(`📥 Đã tải ${filename}`, 'success');
  } catch (e) {
    showToast(`❌ Export lỗi: ${e.message}`, 'error');
  }
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
        pending: '⏳ Chờ', detecting: '🔍 Detect',
        preprocessing: '⚙️ Xử lý', ocr: '🔤 OCR', extracting: '🤖 AI',
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
