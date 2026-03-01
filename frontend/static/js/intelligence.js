/* ============================================================
   SentimentIQ · Intelligence Hub
   Client-side NLP intelligence engine — zero backend changes
   ============================================================ */

'use strict';

Chart.defaults.color       = '#64748b';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'IBM Plex Sans', sans-serif";
Chart.defaults.plugins.legend.display = false;

// ── Airline Detection Dictionary ──────────────────────────────
const AIRLINES = {
  'United':         ['united', '@united', 'unitedairlines', 'united airlines'],
  'American':       ['american', '@americanair', 'american airlines', 'americanairlines'],
  'US Airways':     ['us airways', 'usairways', '@usairways', 'usair'],
  'Southwest':      ['southwest', '@southwestair', 'southwest airlines'],
  'Delta':          ['delta', '@delta', 'delta airlines'],
  'Virgin America': ['virgin america', '@virginamerica', 'virgin'],
};

// ── Issue Category Dictionary ─────────────────────────────────
const ISSUES = {
  'Flight Delays':      ['delay','delayed','late','behind schedule','hours wait','waiting','on time','overdue','tarmac','depart','departure'],
  'Lost / Baggage':     ['bag','luggage','baggage','lost','missing','suitcase','belongings','checked bag','claim'],
  'Customer Service':   ['staff','crew','agent','rude','service','unhelpful','helpful','representative','attendant','gate agent','employee'],
  'Cancellations':      ['cancel','cancelled','cancellation','cancellations','reschedule','rebook','no flight'],
  'Booking / Seating':  ['book','booking','ticket','reservation','seat','upgrade','check-in','checkin','boarding pass','overbooked'],
  'Food & Comfort':     ['food','meal','drink','water','snack','hungry','comfort','legroom','cramped','temperature'],
  'Communication':      ['inform','update','notification','told','communication','information','gate change','announcement','no info','no update'],
};

const STOPWORDS = new Set([
  'i','me','my','we','our','you','your','he','she','they','them','their',
  'a','an','the','and','or','but','in','on','at','to','for','of','with',
  'is','are','was','were','be','been','being','have','has','had','do','does',
  'did','will','would','could','should','may','might','shall','can','not',
  'no','nor','so','yet','both','either','just','also','very','too','then',
  'than','that','this','these','those','it','its','as','if','by','up','out',
  'from','about','into','through','during','after','before','between','here',
  'there','when','where','while','how','all','each','more','most','other',
  'some','such','what','which','who','because','until','although','since',
  'amp','get','got','im','ive','dont','cant','wont','didnt','flight','airline',
  'plane','fly','flying','flyer','air','trip','travel','passengers','passenger',
  'rt','via','http','https','co','com','www','gt','lt',
]);

// ── State ─────────────────────────────────────────────────────
let intelData       = null;
let airlineView     = 'negative';
let issueView       = 'count';
let keywordView     = 'all';
let simInterval     = null;
let simIndex        = 0;
let simAlerts       = [];

// Chart instances
let airlineChartI, issueChartI, keywordChartI,
    severityChartI, pulseChartI;

// ── Boot: try to load from sessionStorage ─────────────────────
window.addEventListener('DOMContentLoaded', () => {
  const stored = sessionStorage.getItem('sentimentResults');
  if (stored) {
    try {
      const parsed = JSON.parse(stored);
      if (parsed && parsed.predictions && parsed.predictions.length > 0) {
        buildIntelligence(parsed, 'Batch Results (from previous analysis)');
        return;
      }
    } catch (_) {}
  }
  // Nothing in session — show upload state
  document.getElementById('noDataState').style.display = 'block';
  wireUpload();
});

// ── File upload wiring ────────────────────────────────────────
function wireUpload() {
  const input   = document.getElementById('intelFileInput');
  const zone    = document.getElementById('uploadZone');
  const btn     = document.getElementById('intelAnalyseBtn');

  input.addEventListener('change', () => {
    const file = input.files[0];
    if (!file) return;
    document.getElementById('intelFileName').textContent = file.name;
    document.getElementById('intelFileSize').textContent = fmt(file.size);
    document.getElementById('intelFileBar').style.display = 'flex';
    btn.disabled = false;
  });

  zone.addEventListener('dragover',  e => { e.preventDefault(); zone.classList.add('drag-over'); });
  zone.addEventListener('dragleave', ()=> zone.classList.remove('drag-over'));
  zone.addEventListener('drop', e => {
    e.preventDefault(); zone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) { input.files = e.dataTransfer.files; input.dispatchEvent(new Event('change')); }
  });
}

async function runIntelAnalysis() {
  const file = document.getElementById('intelFileInput').files[0];
  if (!file) return;

  document.getElementById('noDataState').style.display = 'none';
  document.getElementById('intelLoading').style.display = 'flex';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/api/predict/csv', { method: 'POST', body: formData });
    if (!res.ok) throw new Error('Analysis failed');
    const data = await res.json();
    buildIntelligence(data, file.name);
  } catch (err) {
    document.getElementById('intelLoading').style.display = 'none';
    document.getElementById('noDataState').style.display = 'block';
    alert('Error: ' + err.message);
  }
}

// ── Core Intelligence Engine ──────────────────────────────────
function buildIntelligence(rawData, datasetName) {
  document.getElementById('intelLoading').style.display   = 'none';
  document.getElementById('noDataState').style.display    = 'none';

  // Enrich every prediction
  rawData.predictions = rawData.predictions.map(p => ({
    ...p,
    airline:   detectAirline(p.text),
    issues:    detectIssues(p.text),
    sarcasm:   computeSarcasmScore(p.text, p.probabilities),
    severity:  p.prediction === 'negative' ? p.confidence : p.prediction === 'positive' ? 0 : p.confidence * 0.3,
  }));

  intelData = rawData;

  // Header
  document.getElementById('intelRecordCount').textContent = rawData.total_texts.toLocaleString();
  document.getElementById('intelDatasetName').textContent = datasetName;
  document.getElementById('intelTimestamp').textContent   = new Date().toLocaleTimeString();
  document.getElementById('liveStatus').textContent       = '⬤ LIVE';
  document.getElementById('liveStatus').style.color       = 'var(--positive)';

  // Render all sections
  renderKPIs();
  renderAirlineChart(airlineView);
  renderIssueChart(issueView);
  renderKeywordChart(keywordView);
  renderSeverityMatrix();
  renderPulseChart();
  renderHealthGauge();
  renderAlertFeed();
  renderCriticalList();
  renderHeatmap();

  document.getElementById('intelDashboard').style.display = 'block';
  document.getElementById('intelDashboard').scrollIntoView({ behavior:'smooth', block:'start' });
}

// ── Airline Detection ─────────────────────────────────────────
function detectAirline(text) {
  const lower = text.toLowerCase();
  for (const [airline, keywords] of Object.entries(AIRLINES)) {
    if (keywords.some(k => lower.includes(k))) return airline;
  }
  return 'Other / Unknown';
}

// ── Issue Detection ───────────────────────────────────────────
function detectIssues(text) {
  const lower   = text.toLowerCase();
  const matched = [];
  for (const [cat, kws] of Object.entries(ISSUES)) {
    if (kws.some(k => lower.includes(k))) matched.push(cat);
  }
  return matched.length ? matched : ['Uncategorised'];
}

// ── Sarcasm Score (same heuristic) ───────────────────────────
function computeSarcasmScore(text, probs) {
  const lower    = text.toLowerCase();
  const posWords = ['amazing','great','excellent','wonderful','fantastic','awesome','love','best','perfect','brilliant','superb','outstanding','delightful','incredible','terrific'];
  const rhetoric = ['as always','of course','naturally','obviously','just what i needed','thanks for nothing','what a surprise','so helpful','so efficient'];
  const politeN  = ['thanks','thank you','appreciate','lovely'];
  const isNeg    = probs.negative > 0.45;
  const isPos    = probs.positive > 0.45;
  const conf     = Math.max(probs.positive, probs.neutral, probs.negative);
  let s = 0;
  if (posWords.some(w => lower.includes(w)) && isNeg)   s += 0.55;
  if (rhetoric.some(p => lower.includes(p)))            s += 0.35;
  if (politeN.some(w => lower.includes(w)) && isNeg)    s += 0.25;
  if ((text.match(/!/g)||[]).length >= 2 && isNeg)      s += 0.15;
  s += (1 - conf) * 0.25;
  return Math.min(Math.max(s, 0), 0.95);
}

// ── KPI Computation ───────────────────────────────────────────
function renderKPIs() {
  const preds   = intelData.predictions;
  const total   = preds.length;

  // Crisis: high-confidence negative (>0.85) + high sarcasm (>0.6)
  const crisisPreds = preds.filter(p => p.prediction === 'negative' && p.confidence > 0.80);
  const highSar     = preds.filter(p => p.sarcasm > 0.60);
  document.getElementById('kpiCrisis').textContent    = crisisPreds.length;
  document.getElementById('kpiCrisisSub').textContent = `${highSar.length} high-sarcasm flagged`;

  // Worst airline by negative %
  const airlineStats = computeAirlineStats();
  const sorted       = Object.entries(airlineStats)
    .filter(([k]) => k !== 'Other / Unknown')
    .sort((a, b) => (b[1].neg / b[1].total) - (a[1].neg / a[1].total));
  if (sorted.length > 0) {
    const [wName, wStat] = sorted[0];
    document.getElementById('kpiWorstAirline').textContent = wName;
    document.getElementById('kpiWorstSub').textContent     = `${((wStat.neg / wStat.total)*100).toFixed(0)}% negative from ${wStat.total} mentions`;
  }

  // Top issue
  const issueCounts = computeIssueCounts();
  const topIssue    = Object.entries(issueCounts).sort((a,b)=>b[1].count-a[1].count)[0];
  if (topIssue) {
    document.getElementById('kpiTopIssue').textContent   = topIssue[0];
    document.getElementById('kpiTopIssueSub').textContent = `${topIssue[1].count} complaints · avg severity ${(topIssue[1].totalSev / topIssue[1].count * 100).toFixed(0)}%`;
  }

  // Sentiment health score: (pos - neg) / total × 100
  const posCount   = preds.filter(p => p.prediction === 'positive').length;
  const negCount   = preds.filter(p => p.prediction === 'negative').length;
  const healthScore = ((posCount - negCount) / total * 100).toFixed(1);
  const healthEl    = document.getElementById('kpiHealth');
  healthEl.textContent = (healthScore > 0 ? '+' : '') + healthScore + '%';
  healthEl.style.color = healthScore > 0 ? 'var(--positive)' : healthScore < -20 ? 'var(--negative)' : '#fbbf24';
  document.getElementById('kpiHealthSub').textContent = healthScore < -20 ? 'Critical — urgent action needed' : healthScore < 0 ? 'Below average — needs attention' : 'Healthy brand perception';

  // Sarcasm rate
  const sarRate = (preds.filter(p => p.sarcasm > 0.4).length / total * 100).toFixed(1);
  document.getElementById('kpiSarcasmRate').textContent   = sarRate + '%';
  document.getElementById('kpiSarcasmRateSub').textContent = `${preds.filter(p => p.sarcasm > 0.6).length} high-risk records`;

  // Show alert strip if crisis count > 5
  if (crisisPreds.length > 5) {
    document.getElementById('alertStrip').style.display  = 'flex';
    document.getElementById('alertStripText').textContent =
      `⚠ ${crisisPreds.length} high-confidence crisis complaints detected — immediate review recommended`;
  }
}

// ── Airline Stats Helper ──────────────────────────────────────
function computeAirlineStats() {
  const stats = {};
  intelData.predictions.forEach(p => {
    const a = p.airline;
    if (!stats[a]) stats[a] = { total:0, pos:0, neu:0, neg:0, sevSum:0, sarSum:0 };
    stats[a].total++;
    stats[a][p.prediction[0] === 'p' ? 'pos' : p.prediction[0] === 'n' && p.prediction[1] === 'e' ? 'neg' : 'neu']++;
    stats[a].sevSum += p.severity;
    stats[a].sarSum += p.sarcasm;
  });
  return stats;
}

// ── Issue Counts Helper ───────────────────────────────────────
function computeIssueCounts() {
  const counts = {};
  intelData.predictions.forEach(p => {
    p.issues.forEach(issue => {
      if (!counts[issue]) counts[issue] = { count:0, totalSev:0, totalSar:0 };
      counts[issue].count++;
      counts[issue].totalSev += p.severity;
      counts[issue].totalSar += p.sarcasm;
    });
  });
  return counts;
}

// ── Chart 1: Airline Scoreboard ───────────────────────────────
function renderAirlineChart(mode) {
  if (airlineChartI) airlineChartI.destroy();
  airlineView = mode;

  const stats = computeAirlineStats();
  const rows  = Object.entries(stats)
    .filter(([k]) => k !== 'Other / Unknown')
    .map(([name, s]) => ({
      name,
      value: mode === 'negative'
        ? (s.neg / s.total * 100)
        : ((s.pos - s.neg) / s.total * 100),
      total: s.total,
    }))
    .sort((a, b) => mode === 'negative' ? b.value - a.value : a.value - b.value);

  if (rows.length === 0) { document.getElementById('airlineChart').parentElement.innerHTML = '<div class="no-data-msg">No airline names detected in this dataset</div>'; return; }

  const colors = rows.map(r => {
    if (mode === 'negative') return r.value > 60 ? 'rgba(244,63,94,0.8)' : r.value > 40 ? 'rgba(251,146,60,0.75)' : 'rgba(251,191,36,0.75)';
    return r.value >= 0 ? 'rgba(16,185,129,0.75)' : 'rgba(244,63,94,0.75)';
  });

  const ctx = document.getElementById('airlineChart').getContext('2d');
  airlineChartI = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: rows.map(r => r.name),
      datasets: [{
        data:            rows.map(r => r.value.toFixed(1)),
        backgroundColor: colors,
        borderRadius: 5, borderSkipped: false,
      }],
    },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      scales: {
        x: { ticks: { color:'#4b6080', callback: v => v + (mode==='negative'?'%':'%') },
             grid: { color:'rgba(255,255,255,0.04)' } },
        y: { ticks: { color:'#94a3b8', font:{ size:11 } }, grid:{ display:false } },
      },
      plugins: {
        tooltip: { callbacks: {
          label: ctx => ` ${parseFloat(ctx.raw).toFixed(1)}% ${mode === 'negative' ? 'negative' : 'net score'} (${rows[ctx.dataIndex].total} records)`
        }}
      },
    },
  });
}

function setAirlineView(mode) {
  document.getElementById('airlineByNeg').classList.toggle('active',   mode === 'negative');
  document.getElementById('airlineByScore').classList.toggle('active', mode === 'score');
  renderAirlineChart(mode);
}

// ── Chart 2: Issue Breakdown ──────────────────────────────────
function renderIssueChart(mode) {
  if (issueChartI) issueChartI.destroy();
  issueView = mode;

  const counts = computeIssueCounts();
  delete counts['Uncategorised'];

  const rows = Object.entries(counts)
    .map(([name, s]) => ({
      name,
      value: mode === 'count' ? s.count : (s.totalSev / s.count * 100),
    }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 8);

  const ISSUE_COLORS = [
    'rgba(239,68,68,0.75)','rgba(251,146,60,0.75)','rgba(251,191,36,0.75)',
    'rgba(34,211,238,0.65)','rgba(167,139,250,0.7)','rgba(16,185,129,0.7)',
    'rgba(59,130,246,0.7)','rgba(244,114,182,0.7)',
  ];

  const ctx = document.getElementById('issueChart').getContext('2d');
  issueChartI = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: rows.map(r => r.name),
      datasets: [{
        data:            rows.map(r => r.value.toFixed(mode==='count'?0:1)),
        backgroundColor: ISSUE_COLORS.slice(0, rows.length),
        borderRadius: 5, borderSkipped: false,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      scales: {
        x: { ticks: { color:'#4b6080', maxRotation:35, font:{ size:9 } }, grid:{ display:false } },
        y: { ticks: { color:'#4b6080' }, grid:{ color:'rgba(255,255,255,0.04)' } },
      },
      plugins: {
        tooltip: { callbacks: {
          label: ctx => mode === 'count'
            ? ` ${ctx.raw} complaints`
            : ` ${parseFloat(ctx.raw).toFixed(1)}% avg severity`
        }}
      },
    },
  });
}

function setIssueView(mode) {
  document.getElementById('issueByCount').classList.toggle('active',  mode === 'count');
  document.getElementById('issueBySev').classList.toggle('active',    mode === 'severity');
  renderIssueChart(mode);
}

// ── Chart 3: Trending Keywords ────────────────────────────────
function renderKeywordChart(mode) {
  if (keywordChartI) keywordChartI.destroy();
  keywordView = mode;

  const subset = mode === 'all'      ? intelData.predictions
               : mode === 'negative' ? intelData.predictions.filter(p => p.prediction === 'negative')
               :                       intelData.predictions.filter(p => p.sarcasm > 0.4);

  const freq = {};
  subset.forEach(p => {
    p.text.toLowerCase()
      .replace(/[^a-z0-9\s@]/g, ' ')
      .split(/\s+/)
      .filter(w => w.length > 3 && !STOPWORDS.has(w))
      .forEach(w => { freq[w] = (freq[w] || 0) + 1; });
  });

  const top = Object.entries(freq)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 15);

  const maxF = top[0]?.[1] || 1;
  const colors = top.map(([,f]) => {
    const t = f / maxF;
    return `rgba(${Math.round(244*t + 59*(1-t))}, ${Math.round(63*(1-t) + 130*t)}, ${Math.round(94*(1-t) + 246*t)}, 0.75)`;
  });

  const ctx = document.getElementById('keywordChart').getContext('2d');
  keywordChartI = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: top.map(([w]) => w),
      datasets: [{ data: top.map(([,f]) => f), backgroundColor: colors, borderRadius:4, borderSkipped:false }],
    },
    options: {
      indexAxis: 'y', responsive: true, maintainAspectRatio: false,
      scales: {
        x: { ticks:{ color:'#4b6080' }, grid:{ color:'rgba(255,255,255,0.04)' } },
        y: { ticks:{ color:'#94a3b8', font:{ size:11 } }, grid:{ display:false } },
      },
      plugins: { tooltip: { callbacks: { label: ctx => ` ${ctx.raw} mentions` }}},
    },
  });
}

function setKeywordView(mode) {
  ['kwAll','kwNeg','kwSar'].forEach(id => document.getElementById(id).classList.remove('active'));
  document.getElementById({ all:'kwAll', negative:'kwNeg', sarcasm:'kwSar' }[mode]).classList.add('active');
  renderKeywordChart(mode);
}

// ── Chart 4: Severity Bubble Matrix ──────────────────────────
function renderSeverityMatrix() {
  if (severityChartI) severityChartI.destroy();

  const counts = computeIssueCounts();
  delete counts['Uncategorised'];

  const points = Object.entries(counts)
    .filter(([,s]) => s.count > 0)
    .map(([name, s]) => ({
      label: name,
      x: s.count,
      y: (s.totalSev / s.count * 100),
      r: Math.max(5, Math.min(20, (s.totalSar / s.count) * 40)),
    }));

  const COLS = ['rgba(239,68,68,0.7)','rgba(251,146,60,0.7)','rgba(251,191,36,0.7)',
                'rgba(34,211,238,0.6)','rgba(167,139,250,0.65)','rgba(16,185,129,0.65)',
                'rgba(59,130,246,0.65)','rgba(244,114,182,0.65)'];

  const ctx = document.getElementById('severityChart').getContext('2d');
  severityChartI = new Chart(ctx, {
    type: 'bubble',
    data: {
      datasets: points.map((p, i) => ({
        label: p.label,
        data: [{ x: p.x, y: p.y, r: p.r }],
        backgroundColor: COLS[i % COLS.length],
        borderColor:     COLS[i % COLS.length].replace('0.7','1').replace('0.6','1').replace('0.65','1'),
        borderWidth: 1,
      })),
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: 'bottom', labels: { color:'#64748b', font:{ size:9 }, boxWidth:8, padding:8 }},
        tooltip: { callbacks: {
          label: ctx => `${ctx.dataset.label}: ${ctx.raw.x} records, ${ctx.raw.y.toFixed(0)}% severity`,
        }},
      },
      scales: {
        x: { title:{ display:true, text:'Volume (# records)', color:'#4b6080', font:{ size:10 }},
             ticks:{ color:'#4b6080' }, grid:{ color:'rgba(255,255,255,0.04)' }},
        y: { title:{ display:true, text:'Avg Severity %',     color:'#4b6080', font:{ size:10 }},
             ticks:{ color:'#4b6080', callback: v => v + '%' }, grid:{ color:'rgba(255,255,255,0.04)' }},
      },
    },
  });
}

// ── Chart 5: Complaint Pulse ──────────────────────────────────
function renderPulseChart() {
  if (pulseChartI) pulseChartI.destroy();

  const preds   = intelData.predictions;
  const win     = Math.max(5, Math.floor(preds.length * 0.08));
  const negRate = preds.map(p => p.prediction === 'negative' ? 1 : 0);
  const sarRate = preds.map(p => p.sarcasm > 0.4 ? 1 : 0);

  const rollingNeg = rollingAvg(negRate, win).map(v => (v * 100).toFixed(1));
  const rollingSar = rollingAvg(sarRate, win).map(v => (v * 100).toFixed(1));
  const labels     = preds.map((_, i) => i + 1);

  const ctx = document.getElementById('pulseChart').getContext('2d');
  pulseChartI = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Negative Rate %',
          data: rollingNeg,
          borderColor: 'rgba(244,63,94,0.9)',
          backgroundColor: 'rgba(244,63,94,0.08)',
          fill: true, tension: 0.4,
          pointRadius: 0, borderWidth: 2,
        },
        {
          label: 'Sarcasm Rate %',
          data: rollingSar,
          borderColor: 'rgba(251,146,60,0.85)',
          backgroundColor: 'rgba(251,146,60,0.05)',
          fill: true, tension: 0.4,
          pointRadius: 0, borderWidth: 2,
          borderDash: [4, 3],
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display:true, position:'bottom',
                  labels:{ color:'#64748b', font:{ size:9 }, boxWidth:10, padding:8 }},
        tooltip: { callbacks: { label: ctx => ` ${ctx.dataset.label}: ${parseFloat(ctx.raw).toFixed(1)}%` }},
      },
      scales: {
        x: { ticks:{ maxTicksLimit:8, color:'#4b6080', font:{ size:9 }},
             grid:{ color:'rgba(255,255,255,0.04)' }},
        y: { ticks:{ color:'#4b6080', callback: v => v + '%' },
             grid:{ color:'rgba(255,255,255,0.04)' }},
      },
    },
  });
}

// ── Brand Health Gauge ────────────────────────────────────────
function renderHealthGauge() {
  const preds = intelData.predictions;
  const pos   = preds.filter(p => p.prediction === 'positive').length;
  const neg   = preds.filter(p => p.prediction === 'negative').length;
  const score = (pos - neg) / preds.length;  // -1 to +1
  const norm  = (score + 1) / 2;            //  0 to 1

  const canvas = document.getElementById('healthGauge');
  const ctx    = canvas.getContext('2d');
  const cx = 80, cy = 80, r = 60;

  ctx.clearRect(0, 0, 160, 160);

  // Outer ring track
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, 0, false);
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = 14; ctx.lineCap = 'round';
  ctx.stroke();

  // Colored fill
  const colorStop = norm > 0.6 ? '#10b981' : norm > 0.35 ? '#fbbf24' : '#f43f5e';
  const endAngle  = Math.PI + (Math.PI * norm);
  const grad      = ctx.createLinearGradient(20, cy, 140, cy);
  grad.addColorStop(0, '#f43f5e');
  grad.addColorStop(0.5, '#fbbf24');
  grad.addColorStop(1, '#10b981');

  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI, endAngle, false);
  ctx.strokeStyle = grad;
  ctx.lineWidth = 14; ctx.lineCap = 'round';
  ctx.stroke();

  // Center text
  ctx.fillStyle = '#f1f5f9';
  ctx.font = 'bold 22px Sora, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText((score * 100).toFixed(0) + '%', cx, cy + 8);

  const label = norm > 0.6 ? 'Healthy' : norm > 0.35 ? 'At Risk' : 'Critical';
  document.getElementById('healthGaugeLabel').textContent = label;
  document.getElementById('healthGaugeLabel').style.color = colorStop;
  document.getElementById('healthGaugeSub').textContent   =
    `${pos} positive · ${neg} negative · ${preds.length - pos - neg} neutral`;
}

// ── Priority Alert Feed ───────────────────────────────────────
function renderAlertFeed() {
  const preds  = intelData.predictions;
  const alerts = preds
    .filter(p => p.prediction === 'negative' && p.confidence > 0.78)
    .sort((a, b) => (b.confidence + b.sarcasm) - (a.confidence + a.sarcasm))
    .slice(0, 6);

  simAlerts = alerts;

  const feed = document.getElementById('alertFeed');
  feed.innerHTML = '';

  if (alerts.length === 0) {
    feed.innerHTML = '<div class="no-data-msg">No critical alerts detected 🎉</div>';
    return;
  }

  alerts.forEach((p, i) => {
    const el    = buildAlertCard(p, i + 1);
    el.style.animationDelay = (i * 0.07) + 's';
    feed.appendChild(el);
  });
}

function buildAlertCard(p, rank) {
  const sarLevel = p.sarcasm > 0.6 ? 'HIGH' : p.sarcasm > 0.3 ? 'MED' : 'LOW';
  const sarColor = p.sarcasm > 0.6 ? 'var(--sarcasm)' : p.sarcasm > 0.3 ? '#fbbf24' : 'var(--text-3)';
  const div = document.createElement('div');
  div.className = 'alert-card fade-in';
  div.innerHTML = `
    <div class="alert-card-top">
      <span class="alert-rank">#${rank}</span>
      <span class="alert-airline">${escHtml(p.airline)}</span>
      <span class="alert-issue">${escHtml(p.issues[0])}</span>
      <span class="alert-conf">${(p.confidence*100).toFixed(0)}% conf</span>
      <span style="font-size:0.7rem;color:${sarColor};font-family:var(--font-mono);">🎭 ${sarLevel}</span>
    </div>
    <div class="alert-card-text">${escHtml(p.text.substring(0, 120))}${p.text.length > 120 ? '…' : ''}</div>
  `;
  return div;
}

// ── Critical Complaints List ──────────────────────────────────
function renderCriticalList() {
  const preds = intelData.predictions;
  const top8  = preds
    .filter(p => p.prediction === 'negative')
    .sort((a, b) => b.severity - a.severity)
    .slice(0, 8);

  const list = document.getElementById('criticalList');
  list.innerHTML = '';

  top8.forEach((p, i) => {
    const div = document.createElement('div');
    div.className = 'critical-row fade-in';
    div.style.animationDelay = (i * 0.06) + 's';
    div.innerHTML = `
      <div class="critical-num">${i + 1}</div>
      <div class="critical-body">
        <div class="critical-text">${escHtml(p.text.substring(0, 100))}${p.text.length > 100 ? '…' : ''}</div>
        <div class="critical-meta">
          <span class="badge negative" style="font-size:0.7rem;padding:2px 7px;">😞 ${(p.confidence*100).toFixed(0)}%</span>
          <span class="critical-tag">${escHtml(p.airline)}</span>
          <span class="critical-tag">${escHtml(p.issues[0])}</span>
          ${p.sarcasm > 0.5 ? '<span class="critical-tag sar">🎭 Sarcasm</span>' : ''}
        </div>
      </div>
      <div class="critical-sev" style="color:${p.severity > 0.7 ? 'var(--negative)' : '#fbbf24'};">${(p.severity*100).toFixed(0)}%</div>
    `;
    list.appendChild(div);
  });
}

// ── Airline × Issue Heatmap ───────────────────────────────────
function renderHeatmap() {
  const airlineKeys = Object.keys(AIRLINES).concat(['Other / Unknown']);
  const issueKeys   = Object.keys(ISSUES).filter(k => k !== 'Uncategorised');

  // Build count matrix
  const matrix = {};
  airlineKeys.forEach(a => {
    matrix[a] = {};
    issueKeys.forEach(i => { matrix[a][i] = 0; });
  });

  intelData.predictions.forEach(p => {
    if (p.prediction !== 'negative') return;
    const a = p.airline;
    if (!matrix[a]) return;
    p.issues.forEach(issue => {
      if (matrix[a][issue] !== undefined) matrix[a][issue]++;
    });
  });

  // Max value for color scaling
  let maxVal = 0;
  airlineKeys.forEach(a => issueKeys.forEach(i => { maxVal = Math.max(maxVal, matrix[a][i]); }));
  if (maxVal === 0) maxVal = 1;

  // Filter out airlines with zero data
  const activeAirlines = airlineKeys.filter(a => issueKeys.some(i => matrix[a][i] > 0));

  const container = document.getElementById('heatmapContainer');
  container.innerHTML = '';

  // Column headers
  const headerRow = document.createElement('div');
  headerRow.className = 'hm-row';
  headerRow.innerHTML = `<div class="hm-label-cell"></div>` +
    issueKeys.map(ik => `<div class="hm-col-header">${ik.replace(' / ',' /\n').replace(' & ',' &\n')}</div>`).join('');
  container.appendChild(headerRow);

  activeAirlines.forEach(airline => {
    const row = document.createElement('div');
    row.className = 'hm-row';
    row.innerHTML = `<div class="hm-label-cell">${airline}</div>` +
      issueKeys.map(issue => {
        const val = matrix[airline][issue];
        const t   = val / maxVal;
        const alpha = 0.08 + t * 0.75;
        const bg  = val === 0 ? 'rgba(255,255,255,0.02)'
                              : `rgba(244, ${Math.round(63 + (1-t)*80), 94}, ${alpha})`;
        const textColor = t > 0.5 ? '#fff' : t > 0.2 ? '#f87171' : 'var(--text-3)';
        return `<div class="hm-cell" title="${airline} · ${issue}: ${val} complaints"
                     style="background:${bg};color:${textColor};">${val > 0 ? val : '·'}</div>`;
      }).join('');
    container.appendChild(row);
  });
}

// ── Live Simulation ───────────────────────────────────────────
let simRunning = false;

function toggleSimulation() {
  if (simRunning) {
    clearInterval(simInterval);
    simRunning = false;
    document.getElementById('simBtn').textContent  = '▶ Start Live Feed';
    document.getElementById('livePill').style.opacity = '0.4';
    document.getElementById('alertLiveDot').style.display = 'none';
  } else {
    if (!intelData) return;
    simRunning = true;
    simIndex   = 0;
    document.getElementById('simBtn').textContent  = '⏸ Pause';
    document.getElementById('livePill').style.opacity = '1';
    document.getElementById('alertLiveDot').style.display = 'block';

    const criticals = intelData.predictions
      .filter(p => p.prediction === 'negative' && p.confidence > 0.75)
      .sort(() => Math.random() - 0.5);

    const feed = document.getElementById('alertFeed');
    feed.innerHTML = '<div class="sim-header">⬤ LIVE — Incoming complaints</div>';

    simInterval = setInterval(() => {
      if (simIndex >= criticals.length) {
        simIndex = 0;
        feed.innerHTML = '<div class="sim-header">⬤ LIVE — Incoming complaints</div>';
      }
      const p  = criticals[simIndex++];
      const el = buildAlertCard(p, simIndex);
      el.classList.add('sim-new');
      feed.insertBefore(el, feed.children[1]);

      // Cap to 5 visible
      while (feed.children.length > 6) feed.removeChild(feed.lastChild);
    }, 1800);
  }
}

// ── Reset ─────────────────────────────────────────────────────
function resetIntel() {
  if (simRunning) toggleSimulation();
  intelData = null;
  sessionStorage.removeItem('sentimentResults');
  document.getElementById('intelDashboard').style.display = 'none';
  document.getElementById('noDataState').style.display    = 'block';
  document.getElementById('intelFileInput').value         = '';
  document.getElementById('intelFileBar').style.display   = 'none';
  document.getElementById('intelAnalyseBtn').disabled     = true;
  document.getElementById('liveStatus').textContent       = '⬤ AWAITING DATA';
  document.getElementById('liveStatus').style.color       = '';
  window.scrollTo({ top:0, behavior:'smooth' });
}

// ── Utilities ─────────────────────────────────────────────────
function rollingAvg(arr, win) {
  return arr.map((_, i) => {
    const slice = arr.slice(Math.max(0, i - win + 1), i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

function fmt(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes/1024).toFixed(1) + ' KB';
  return (bytes/1048576).toFixed(1) + ' MB';
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}
