/* ============================================================
   SentimentIQ · Batch Analytics Page
   ============================================================ */

'use strict';

// ── Chart defaults (dark theme) ────────────────────────────────
Chart.defaults.color           = '#64748b';
Chart.defaults.borderColor     = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family     = "'IBM Plex Sans', sans-serif";
Chart.defaults.plugins.legend.display = false;

const COLORS = {
  positive: '#10b981',
  neutral:  '#a78bfa',
  negative: '#f43f5e',
  sarcasm:  '#fb923c',
  primary:  '#3b82f6',
  cyan:     '#22d3ee',
};

// ── State ──────────────────────────────────────────────────────
let allResults    = null;
let filteredRows  = [];
let currentFilter = 'all';
let trendMode     = 'raw';

// Chart instances (kept to allow destroy on re-render)
let donutChart  = null;
let trendChartI = null;
let confChartI  = null;
let sarChartI   = null;

// ── File Input Wiring ──────────────────────────────────────────
const fileInput    = document.getElementById('fileInput');
const uploadZone   = document.getElementById('uploadZone');
const analyseBtn   = document.getElementById('analyseBtn');
const selectedBar  = document.getElementById('fileSelectedBar');

fileInput.addEventListener('change', onFileSelected);

uploadZone.addEventListener('dragover',  e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) { fileInput.files = e.dataTransfer.files; onFileSelected(); }
});

function onFileSelected() {
  const file = fileInput.files[0];
  if (!file) return;
  if (!file.name.endsWith('.csv')) { alert('Please select a .csv file.'); return; }

  document.getElementById('fileSelectedName').textContent = file.name;
  document.getElementById('fileSelectedSize').textContent = formatBytes(file.size);
  selectedBar.style.display = 'flex';
  analyseBtn.disabled       = false;
}

function clearFile() {
  fileInput.value           = '';
  selectedBar.style.display = 'none';
  analyseBtn.disabled       = true;
}

// ── Run Batch ──────────────────────────────────────────────────
async function runBatchAnalysis() {
  const file = fileInput.files[0];
  if (!file) return;

  showBatchLoading();

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/predict/csv', { method: 'POST', body: formData });
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Analysis failed'); }

    const data = await res.json();
    allResults = enrichWithSarcasm(data);
    renderDashboard(file.name);
  } catch (err) {
    console.error(err);
    hideBatchLoading();
    alert('Error: ' + err.message + '\n\nCheck that the SageMaker endpoint is running.');
  }
}

// ── Enrich predictions with sarcasm scores ─────────────────────
function enrichWithSarcasm(data) {
  data.predictions = data.predictions.map(p => {
    p.sarcasm = computeSarcasmScore(p.text, p.probabilities);
    return p;
  });
  return data;
}

// ── Sarcasm heuristic (same logic as single page) ──────────────
function computeSarcasmScore(text, probs) {
  const lower     = text.toLowerCase();
  const posWords  = ['amazing','great','excellent','wonderful','fantastic','awesome','love','best',
                     'perfect','brilliant','superb','outstanding','delightful','incredible','terrific'];
  const negWords  = ['terrible','awful','worst','hate','horrible','dreadful','disgusting','pathetic'];
  const rhetoric  = ['as always','of course','naturally','obviously','just what i needed',
                     'thanks for nothing','what a surprise','so helpful','so efficient'];
  const politeNeg = ['thanks','thank you','appreciate','lovely'];

  const hasPosWord  = posWords.some(w => lower.includes(w));
  const hasNegWord  = negWords.some(w => lower.includes(w));
  const hasRhetoric = rhetoric.some(p => lower.includes(p));
  const hasPolitNeg = politeNeg.some(w => lower.includes(w));
  const exclaims    = (text.match(/!/g) || []).length;

  const isNeg = probs.negative > 0.45;
  const isPos = probs.positive > 0.45;
  const conf  = Math.max(probs.positive, probs.neutral, probs.negative);

  let s = 0;
  if (hasPosWord && isNeg)  s += 0.55;
  if (hasNegWord && isPos)  s += 0.30;
  if (hasRhetoric)          s += 0.35;
  if (hasPolitNeg && isNeg) s += 0.25;
  if (exclaims >= 2 && isNeg) s += 0.15;
  s += (1 - conf) * 0.25;

  return Math.min(Math.max(s, 0), 0.95);
}

// ── Render Full Dashboard ──────────────────────────────────────
function renderDashboard(filename) {
  hideBatchLoading();
  const dash = document.getElementById('resultsDashboard');
  dash.style.display = 'block';
  document.getElementById('uploadSection').style.display = 'none';

  const preds      = allResults.predictions;
  const total      = allResults.total_texts;
  const posCount   = preds.filter(p => p.prediction === 'positive').length;
  const neuCount   = preds.filter(p => p.prediction === 'neutral').length;
  const negCount   = preds.filter(p => p.prediction === 'negative').length;
  const highSar    = preds.filter(p => p.sarcasm >= 0.6).length;
  const avgSar     = preds.reduce((a, p) => a + p.sarcasm, 0) / total;

  // KPI cards
  animateCounter('kpiTotal', total);
  document.getElementById('kpiFileName').textContent  = filename;
  animateCounter('kpiPos', Math.round((posCount / total) * 100), '%');
  document.getElementById('kpiPosCount').textContent  = `${posCount} records`;
  animateCounter('kpiNeg', Math.round((negCount / total) * 100), '%');
  document.getElementById('kpiNegCount').textContent  = `${negCount} records`;
  animateCounter('kpiSar', Math.round(avgSar * 100), '%');
  document.getElementById('kpiHighSarCount').textContent = `${highSar} high-risk`;

  // Filter counts
  document.getElementById('cntAll').textContent = total;
  document.getElementById('cntPos').textContent = posCount;
  document.getElementById('cntNeu').textContent = neuCount;
  document.getElementById('cntNeg').textContent = negCount;
  document.getElementById('cntSar').textContent = highSar;

  // Charts
  renderDonut(posCount, neuCount, negCount);
  renderTrend(preds, 'raw');
  renderConfHistogram(preds);
  renderSarcasmBar(preds);

  // Table
  filteredRows  = preds;
  currentFilter = 'all';
  setActiveFilter('filterAll');
  renderTable(preds);

  // Scroll
  setTimeout(() => dash.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);

  // Save to sessionStorage so Intelligence Hub can read it
  try { sessionStorage.setItem('sentimentResults', JSON.stringify(allResults)); } catch(e) {}
}

// ── Chart 1: Sentiment Donut ───────────────────────────────────
function renderDonut(pos, neu, neg) {
  if (donutChart) { donutChart.destroy(); }

  const ctx = document.getElementById('donutChart').getContext('2d');
  donutChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Positive', 'Neutral', 'Negative'],
      datasets: [{
        data: [pos, neu, neg],
        backgroundColor: [COLORS.positive, COLORS.neutral, COLORS.negative],
        borderColor: '#091425',
        borderWidth: 3,
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      cutout: '72%',
      plugins: {
        tooltip: { callbacks: {
          label: ctx => {
            const total = ctx.dataset.data.reduce((a, b) => a + b, 0);
            return ` ${ctx.label}: ${ctx.raw} (${((ctx.raw / total) * 100).toFixed(1)}%)`;
          }
        }}
      }
    }
  });

  // Legend
  const legend = document.getElementById('donutLegend');
  const total = pos + neu + neg;
  legend.innerHTML = [
    ['Positive', COLORS.positive, pos],
    ['Neutral',  COLORS.neutral,  neu],
    ['Negative', COLORS.negative, neg],
  ].map(([label, color, count]) => `
    <div class="legend-item">
      <div class="legend-dot" style="background:${color};"></div>
      <span>${label} — <strong style="color:var(--text-1)">${count}</strong>
      <span style="color:var(--text-3);">(${((count/total)*100).toFixed(1)}%)</span></span>
    </div>
  `).join('');
}

// ── Chart 2: Sentiment Trend Line ─────────────────────────────
function renderTrend(preds, mode) {
  if (trendChartI) { trendChartI.destroy(); }

  trendMode = mode;

  // Convert prediction to numeric score: pos=1, neu=0, neg=-1
  const scores = preds.map(p =>
    p.prediction === 'positive' ? 1 : p.prediction === 'negative' ? -1 : 0
  );

  let labels, data;

  if (mode === 'raw') {
    // Rolling average with window = ~10% of data (min 5, max 50)
    const win = Math.min(50, Math.max(5, Math.floor(preds.length * 0.1)));
    data   = rollingAvg(scores, win);
    labels = data.map((_, i) => i + 1);
  } else {
    // Chunk into groups of ~20 rows, label as "Chunk N"
    const chunkSize = Math.max(10, Math.floor(preds.length / 15));
    const chunks = chunkArray(scores, chunkSize);
    data   = chunks.map(c => (c.reduce((a, b) => a + b, 0) / c.length).toFixed(3));
    labels = chunks.map((_, i) => `G${i + 1}`);
  }

  const ctx = document.getElementById('trendChart').getContext('2d');
  trendChartI = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Sentiment Score',
        data,
        borderColor: COLORS.cyan,
        backgroundColor: 'rgba(34,211,238,0.07)',
        fill: true,
        tension: 0.45,
        pointRadius: data.length < 60 ? 2 : 0,
        pointHoverRadius: 5,
        borderWidth: 2,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: {
          ticks: { maxTicksLimit: 8, color: '#4b6080', font: { size: 10 } },
          grid: { color: 'rgba(255,255,255,0.04)' }
        },
        y: {
          min: -1.1, max: 1.1,
          ticks: { stepSize: 0.5, color: '#4b6080', font: { size: 10 },
                   callback: v => v === 1 ? '+1 Pos' : v === 0 ? '0 Neu' : v === -1 ? '-1 Neg' : v },
          grid: { color: 'rgba(255,255,255,0.04)' }
        }
      },
      plugins: {
        tooltip: { callbacks: {
          label: ctx => ` Score: ${parseFloat(ctx.raw).toFixed(2)}`
        }}
      }
    }
  });
}

function setTrend(mode) {
  if (!allResults) return;
  document.getElementById('trendRaw').classList.toggle('active',  mode === 'raw');
  document.getElementById('trendWeek').classList.toggle('active', mode === 'week');
  renderTrend(allResults.predictions, mode);
}

// ── Chart 3: Confidence Histogram ─────────────────────────────
function renderConfHistogram(preds) {
  if (confChartI) { confChartI.destroy(); }

  // Bucket into 10% bins
  const bins   = Array(10).fill(0);
  const labels = ['0–10%','10–20%','20–30%','30–40%','40–50%',
                  '50–60%','60–70%','70–80%','80–90%','90–100%'];

  preds.forEach(p => {
    const bin = Math.min(9, Math.floor(p.confidence * 10));
    bins[bin]++;
  });

  const ctx = document.getElementById('confChart').getContext('2d');
  confChartI = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Records',
        data: bins,
        backgroundColor: bins.map((_, i) => {
          const t = i / 9;
          const r = Math.round(244 * (1 - t) + 59  * t);
          const g = Math.round(63  * (1 - t) + 130 * t);
          const b = Math.round(94  * (1 - t) + 246 * t);
          return `rgba(${r},${g},${b},0.75)`;
        }),
        borderRadius: 4,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: '#4b6080', font: { size: 9 } }, grid: { display: false } },
        y: { ticks: { color: '#4b6080', font: { size: 10 } }, grid: { color: 'rgba(255,255,255,0.04)' } }
      },
      plugins: { tooltip: { callbacks: { label: ctx => ` ${ctx.raw} records` }}}
    }
  });
}

// ── Chart 4: Sarcasm Risk Profile ─────────────────────────────
function renderSarcasmBar(preds) {
  if (sarChartI) { sarChartI.destroy(); }

  const low    = preds.filter(p => p.sarcasm < 0.30).length;
  const medium = preds.filter(p => p.sarcasm >= 0.30 && p.sarcasm < 0.60).length;
  const high   = preds.filter(p => p.sarcasm >= 0.60).length;

  const ctx = document.getElementById('sarChart').getContext('2d');
  sarChartI = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Low Risk\n(< 30%)', 'Medium Risk\n(30–60%)', 'High Risk\n(> 60%)'],
      datasets: [{
        data: [low, medium, high],
        backgroundColor: [COLORS.positive, '#fbbf24', COLORS.sarcasm],
        borderRadius: 6,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      indexAxis: 'y',
      scales: {
        x: { ticks: { color: '#4b6080' }, grid: { color: 'rgba(255,255,255,0.04)' } },
        y: { ticks: { color: '#94a3b8', font: { size: 11 } }, grid: { display: false } }
      },
      plugins: { tooltip: { callbacks: { label: ctx => ` ${ctx.raw} records` }}}
    }
  });
}

// ── Table Rendering ────────────────────────────────────────────
function renderTable(rows) {
  const tbody = document.getElementById('resultsBody');
  tbody.innerHTML = '';

  rows.forEach((p, i) => {
    const sarLevel = p.sarcasm < 0.30 ? 'low' : p.sarcasm < 0.60 ? 'medium' : 'high';
    const sarLabel = p.sarcasm < 0.30 ? 'Low'  : p.sarcasm < 0.60 ? 'Med'   : 'High';
    const posW     = (p.probabilities.positive * 100).toFixed(0);
    const neuW     = (p.probabilities.neutral  * 100).toFixed(0);
    const negW     = (p.probabilities.negative * 100).toFixed(0);
    const confPct  = (p.confidence * 100).toFixed(1);
    const emoji    = p.prediction === 'positive' ? '😊' : p.prediction === 'negative' ? '😞' : '😐';

    const row = document.createElement('tr');
    row.dataset.prediction = p.prediction;
    row.dataset.sarcasm    = p.sarcasm;

    row.innerHTML = `
      <td class="mono text-3" style="font-size:0.78rem;">${p.row}</td>
      <td><div class="text-clip" title="${escapeHtml(p.text)}">${escapeHtml(p.text)}</div></td>
      <td>
        <span class="badge ${p.prediction}">${emoji} ${capitalize(p.prediction)}</span>
      </td>
      <td>
        <div class="inline-bar">
          <div class="inline-bar-track">
            <div class="inline-bar-fill ${p.prediction === 'positive' ? 'pos' : p.prediction === 'negative' ? 'neg' : 'neu'}"
                 style="width:${confPct}%;"></div>
          </div>
          <span class="inline-val">${confPct}%</span>
        </div>
      </td>
      <td><span class="mono" style="color:var(--positive);font-size:0.78rem;">${posW}%</span></td>
      <td><span class="mono" style="color:var(--neutral); font-size:0.78rem;">${neuW}%</span></td>
      <td><span class="mono" style="color:var(--negative);font-size:0.78rem;">${negW}%</span></td>
      <td><span class="sarcasm-pill ${sarLevel}">${sarLabel} · ${Math.round(p.sarcasm * 100)}%</span></td>
    `;
    tbody.appendChild(row);
  });
}

// ── Filter ─────────────────────────────────────────────────────
function filterTable(filter) {
  currentFilter = filter;
  setActiveFilter(`filter${capitalize(filter === 'sarcasm' ? 'Sar' : filter === 'all' ? 'All' : filter)}`);

  const preds = allResults.predictions;
  let rows;

  if (filter === 'all')     rows = preds;
  else if (filter === 'sarcasm') rows = preds.filter(p => p.sarcasm >= 0.6);
  else                      rows = preds.filter(p => p.prediction === filter);

  filteredRows = rows;
  renderTable(rows);
}

function setActiveFilter(activeId) {
  ['filterAll','filterPos','filterNeu','filterNeg','filterSar'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle('active', el.id === activeId);
  });
}

// ── Download CSV ───────────────────────────────────────────────
function downloadResults() {
  if (!allResults) return;

  const rows = filteredRows.length ? filteredRows : allResults.predictions;
  let csv = 'Row,Text,Sentiment,Confidence,Positive%,Neutral%,Negative%,Sarcasm_Risk,Sarcasm_Level\n';

  rows.forEach(p => {
    const sarLevel = p.sarcasm < 0.30 ? 'Low' : p.sarcasm < 0.60 ? 'Medium' : 'High';
    csv += [
      p.row,
      `"${p.text.replace(/"/g, '""')}"`,
      p.prediction,
      (p.confidence * 100).toFixed(2),
      (p.probabilities.positive * 100).toFixed(2),
      (p.probabilities.neutral  * 100).toFixed(2),
      (p.probabilities.negative * 100).toFixed(2),
      (p.sarcasm * 100).toFixed(2),
      sarLevel,
    ].join(',') + '\n';
  });

  downloadBlob(csv, 'sentiment_analysis_results.csv', 'text/csv');
}

function downloadSampleCSV() {
  const csv = `text
"Absolutely outstanding service from gate to landing, crew was exceptional!"
"Flight departed on time. Arrival was on schedule. Standard experience."
"Delayed 3 hours, zero communication, no compensation offered. Disgraceful."
"Oh brilliant, another mechanical delay. Because this always happens to me."
"The meal was decent, nothing special but nothing terrible either."
"Thank you for losing my luggage again. Truly a world-class operation."
"Boarding process was smooth and staff were genuinely friendly."
"Would not recommend this airline to anyone. Worst experience of my life."
"Flight was fine. Average in every way. Got from A to B."
"Wow amazing service as usual! Only 2 hours late with no explanation!"`;

  downloadBlob(csv, 'sample_airline_feedback.csv', 'text/csv');
}

function downloadBlob(content, filename, type) {
  const blob = new Blob([content], { type });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click();
  document.body.removeChild(a); URL.revokeObjectURL(url);
}

// ── Reset Dashboard ────────────────────────────────────────────
function resetDashboard() {
  allResults    = null;
  filteredRows  = [];
  currentFilter = 'all';

  document.getElementById('resultsDashboard').style.display = 'none';
  document.getElementById('uploadSection').style.display    = 'block';
  clearFile();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Loading States ─────────────────────────────────────────────
function showBatchLoading() {
  document.getElementById('uploadSection').style.display = 'none';
  document.getElementById('batchLoading').style.display  = 'flex';
}

function hideBatchLoading() {
  document.getElementById('batchLoading').style.display = 'none';
}

// ── Utilities ──────────────────────────────────────────────────
function animateCounter(id, target, suffix = '') {
  const el   = document.getElementById(id);
  const dur  = 900;
  const fps  = 60;
  const step = target / (dur / (1000 / fps));
  let current = 0;

  const timer = setInterval(() => {
    current += step;
    if (current >= target) { current = target; clearInterval(timer); }
    el.textContent = Math.round(current) + suffix;
  }, 1000 / fps);
}

function rollingAvg(arr, win) {
  return arr.map((_, i) => {
    const start = Math.max(0, i - win + 1);
    const slice = arr.slice(start, i + 1);
    return (slice.reduce((a, b) => a + b, 0) / slice.length).toFixed(3);
  });
}

function chunkArray(arr, size) {
  const chunks = [];
  for (let i = 0; i < arr.length; i += size) chunks.push(arr.slice(i, i + size));
  return chunks;
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}
