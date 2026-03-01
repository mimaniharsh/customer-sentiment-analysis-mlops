/* ============================================================
   SentimentIQ · Single Analysis Page
   ============================================================ */

'use strict';

const MAX_CHARS = 512;
let history = [];
let sarcasmCtx = null;

// ── Textarea char count ────────────────────────────────────────
const textarea = document.getElementById('textInput');
const charCount = document.getElementById('charCount');

textarea.addEventListener('input', () => {
  const len = textarea.value.length;
  charCount.textContent = `${len} / ${MAX_CHARS}`;
  charCount.style.color = len > MAX_CHARS * 0.9 ? 'var(--negative)' : 'var(--text-3)';
});

// ── Core Analysis ──────────────────────────────────────────────
async function runAnalysis() {
  const text = textarea.value.trim();
  if (!text) { flashTextarea(); return; }
  if (text.length > MAX_CHARS) { alert(`Please keep text under ${MAX_CHARS} characters.`); return; }

  showLoading();

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    if (!res.ok) throw new Error(`Server returned ${res.status}`);

    const data = await res.json();
    const sarcasm = computeSarcasmScore(text, data.probabilities);
    renderResult(data, sarcasm);
    pushHistory(text, data, sarcasm);
  } catch (err) {
    console.error(err);
    showError(err.message);
  } finally {
    hideLoading();
  }
}

// ── Sarcasm Heuristic ──────────────────────────────────────────
// Computes a 0–1 sarcasm likelihood from linguistic patterns.
// Signals: (1) positive words + negative prediction,
//          (2) rhetorical markers ("as always", "of course"),
//          (3) excessive positivity in negative context,
//          (4) prediction ambiguity (low confidence = possible irony).
function computeSarcasmScore(text, probs) {
  const lower = text.toLowerCase();

  const posWords   = ['amazing', 'great', 'excellent', 'wonderful', 'fantastic',
                      'awesome', 'love', 'best', 'perfect', 'brilliant', 'superb',
                      'outstanding', 'delightful', 'incredible', 'terrific', 'splendid'];
  const negWords   = ['terrible', 'awful', 'worst', 'hate', 'horrible', 'dreadful',
                      'disgusting', 'pathetic', 'useless', 'atrocious'];
  const rhetorical = ['as always', 'of course', 'naturally', 'obviously',
                      'clearly', 'just what i needed', 'thanks for nothing',
                      'what a surprise', 'so helpful', 'so efficient'];
  const politeNeg  = ['thanks', 'thank you', 'appreciate', 'lovely'];

  const hasPosWord   = posWords.some(w => lower.includes(w));
  const hasNegWord   = negWords.some(w => lower.includes(w));
  const hasRhetoric  = rhetorical.some(p => lower.includes(p));
  const hasPoliteNeg = politeNeg.some(w => lower.includes(w));
  const hasExclaim   = (text.match(/!/g) || []).length >= 2;

  const isNegPred = probs.negative > 0.45;
  const isPosPred = probs.positive > 0.45;
  const confidence = Math.max(probs.positive, probs.neutral, probs.negative);
  const ambiguity  = 1 - confidence;

  let score = 0;

  // Strong signals
  if (hasPosWord && isNegPred)   score += 0.55;  // "great service" → model says negative
  if (hasNegWord && isPosPred)   score += 0.30;
  if (hasRhetoric)               score += 0.35;  // "as always", "of course"
  if (hasPoliteNeg && isNegPred) score += 0.25;  // "thanks" in a negative prediction
  if (hasExclaim && isNegPred)   score += 0.15;  // multiple exclamations in negative context

  // Weak signals
  score += ambiguity * 0.25;     // low model confidence = possible irony

  // Never go above 0.95 or below 0
  return Math.min(Math.max(score, 0), 0.95);
}

// ── Render Result ──────────────────────────────────────────────
function renderResult(data, sarcasmScore) {
  const panel = document.getElementById('resultPanel');
  const pred  = data.prediction;

  // Verdict banner
  const banner = document.getElementById('verdictBanner');
  banner.className = `result-verdict ${pred}`;
  document.getElementById('verdictEmoji').textContent         = data.sentiment_emoji;
  document.getElementById('verdictBadge').textContent         = pred.toUpperCase();
  document.getElementById('verdictBadge').className           = `verdict-badge ${pred}`;
  document.getElementById('verdictLabel').textContent         = capitalize(pred) + ' Sentiment';
  document.getElementById('verdictInterpretation').textContent = data.interpretation;

  // Confidence metric
  const confPct = (data.confidence * 100).toFixed(1);
  const confEl  = document.getElementById('metricConfidence');
  confEl.textContent  = confPct + '%';
  confEl.className    = `metric-box-value ${pred === 'positive' ? 'pos' : pred === 'negative' ? 'neg' : 'neu'}`;

  // Sentiment score (+1 to -1)
  const sentimentScore = (
    data.probabilities.positive * 1 +
    data.probabilities.neutral  * 0 +
    data.probabilities.negative * -1
  ).toFixed(2);
  const scoreEl = document.getElementById('metricScore');
  scoreEl.textContent = sentimentScore > 0 ? `+${sentimentScore}` : sentimentScore;
  scoreEl.className   = `metric-box-value ${parseFloat(sentimentScore) > 0.1 ? 'pos' : parseFloat(sentimentScore) < -0.1 ? 'neg' : 'neu'}`;

  // Probability bars (animate after tiny delay)
  setTimeout(() => {
    setBar('barPos', 'pctPos', data.probabilities.positive);
    setBar('barNeu', 'pctNeu', data.probabilities.neutral);
    setBar('barNeg', 'pctNeg', data.probabilities.negative);
  }, 80);

  // Sarcasm meter
  renderSarcasmGauge(sarcasmScore);

  // Show panel
  document.getElementById('placeholder').style.display     = 'none';
  document.getElementById('loadingOverlay').style.display  = 'none';
  panel.classList.remove('hidden');
}

function setBar(barId, pctId, prob) {
  const pct = (prob * 100).toFixed(1);
  document.getElementById(barId).style.width  = pct + '%';
  document.getElementById(pctId).textContent  = pct + '%';
}

// ── Sarcasm Gauge (canvas arc) ────────────────────────────────
function renderSarcasmGauge(score) {
  const pct   = Math.round(score * 100);
  const level = score < 0.30 ? 'low' : score < 0.60 ? 'medium' : 'high';

  const descriptions = {
    low:    'Likely genuine — the language and prediction align naturally.',
    medium: 'Possible irony detected. Positive wording with a negative or ambiguous prediction.',
    high:   'Strong sarcasm indicators: positive language, rhetorical markers, or polite framing with a negative prediction.'
  };

  const levelLabels = { low: 'Low Risk', medium: 'Moderate Risk', high: 'High Risk' };
  const colors       = { low: 'var(--positive)', medium: '#fbbf24', high: 'var(--sarcasm)' };

  document.getElementById('sarcasmPct').textContent   = pct + '%';
  const lvlEl = document.getElementById('sarcasmLevel');
  lvlEl.textContent = levelLabels[level];
  lvlEl.className   = `sarcasm-level ${level}`;
  document.getElementById('sarcasmDesc').textContent  = descriptions[level];

  // Draw arc
  const canvas = document.getElementById('sarcasmCanvas');
  const ctx    = canvas.getContext('2d');
  const cx = 36, cy = 36, r = 28, lw = 6;

  ctx.clearRect(0, 0, 72, 72);

  // Track
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  ctx.lineWidth = lw;
  ctx.stroke();

  // Fill arc
  const startAngle = -Math.PI / 2;
  const endAngle   = startAngle + (Math.PI * 2 * score);

  const gradient = ctx.createLinearGradient(0, 0, 72, 72);
  gradient.addColorStop(0, level === 'low' ? '#10b981' : level === 'medium' ? '#fbbf24' : '#fb923c');
  gradient.addColorStop(1, level === 'low' ? '#34d399' : level === 'medium' ? '#f59e0b' : '#ef4444');

  ctx.beginPath();
  ctx.arc(cx, cy, r, startAngle, endAngle);
  ctx.strokeStyle = gradient;
  ctx.lineWidth   = lw;
  ctx.lineCap     = 'round';
  ctx.stroke();
}

// ── History ────────────────────────────────────────────────────
function pushHistory(text, data, sarcasmScore) {
  history.unshift({ text, data, sarcasmScore, ts: Date.now() });
  if (history.length > 5) history.pop();
  renderHistory();
}

function renderHistory() {
  const section = document.getElementById('historySection');
  const list    = document.getElementById('historyList');

  if (history.length === 0) { section.style.display = 'none'; return; }
  section.style.display = 'block';

  list.innerHTML = history.map((item, i) => `
    <div class="history-item" onclick="reloadHistory(${i})">
      <div class="history-dot ${item.data.prediction}"></div>
      <div class="history-text">${escapeHtml(item.text.substring(0, 80))}${item.text.length > 80 ? '…' : ''}</div>
      <div class="history-pct">${(item.data.confidence * 100).toFixed(0)}%</div>
    </div>
  `).join('');
}

function reloadHistory(idx) {
  const item = history[idx];
  if (!item) return;
  textarea.value = item.text;
  charCount.textContent = `${item.text.length} / ${MAX_CHARS}`;
  const sarcasm = computeSarcasmScore(item.text, item.data.probabilities);
  renderResult(item.data, sarcasm);
}

// ── UI Helpers ─────────────────────────────────────────────────
function showLoading() {
  document.getElementById('placeholder').style.display    = 'none';
  document.getElementById('resultPanel').classList.add('hidden');
  document.getElementById('loadingOverlay').style.display = 'flex';
}

function hideLoading() {
  document.getElementById('loadingOverlay').style.display = 'none';
}

function showError(msg) {
  document.getElementById('loadingOverlay').style.display = 'none';
  document.getElementById('placeholder').style.display    = 'flex';
  document.getElementById('placeholder').innerHTML = `
    <div class="results-placeholder-icon">⚠️</div>
    <div class="results-placeholder-text" style="color:var(--negative);">Error: ${escapeHtml(msg)}</div>
    <div style="font-size:0.8rem;color:var(--text-3);margin-top:4px;">Check that the SageMaker endpoint is running.</div>
  `;
}

function flashTextarea() {
  textarea.style.borderColor = 'var(--negative)';
  textarea.style.boxShadow   = '0 0 0 3px var(--negative-dim)';
  setTimeout(() => {
    textarea.style.borderColor = '';
    textarea.style.boxShadow   = '';
  }, 700);
  textarea.focus();
}

function clearAll() {
  textarea.value = '';
  charCount.textContent = '0 / 512';
  document.getElementById('resultPanel').classList.add('hidden');
  document.getElementById('placeholder').style.display = 'flex';
  document.getElementById('placeholder').innerHTML = `
    <div class="results-placeholder-icon">📊</div>
    <div class="results-placeholder-text">Enter text on the left to see analysis</div>
  `;
  textarea.focus();
}

function setExample(text) {
  textarea.value = text;
  charCount.textContent = `${text.length} / ${MAX_CHARS}`;
  textarea.focus();
}

function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// Enter key submits
textarea.addEventListener('keydown', e => {
  if (e.key === 'Enter' && e.ctrlKey) runAnalysis();
});
