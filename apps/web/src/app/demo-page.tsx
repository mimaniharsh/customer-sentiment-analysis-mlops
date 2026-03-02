"use client";

import { useState, useRef, useCallback } from "react";
import axios from "axios";
import {
  Zap, Upload, Send, RotateCcw, ChevronDown, ChevronUp,
  AlertTriangle, CheckCircle, Minus, FileText, X,
  TrendingUp, TrendingDown, Activity, Target, Clock
} from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}/api/v1`
  : "http://localhost:8000/api/v1";

// ─── Types ────────────────────────────────────────────────────────────────────

interface Prediction {
  id: number;
  text: string;
  sentiment: "positive" | "neutral" | "negative";
  sentiment_score: number;
  confidence: number;
  probabilities: { positive: number; neutral: number; negative: number };
  urgency: "critical" | "high" | "medium" | "low";
  urgency_score: number;
  severity_score: number;
  business_impact_score: number;
  attributes: string[];
  attribute_scores: Record<string, number>;
  explanation?: {
    top_tokens?: { token: string; importance: number }[];
  };
  latency_ms: number;
}

interface BatchSummary {
  total: number;
  positive: number;
  neutral: number;
  negative: number;
  positive_rate: number;
  negative_rate: number;
  avg_confidence: number;
  avg_business_impact: number;
  critical_count: number;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const SENTIMENT_STYLES = {
  positive: { text: "text-emerald-400", bg: "bg-emerald-500/15", border: "border-emerald-500/30", icon: CheckCircle },
  neutral:  { text: "text-amber-400",   bg: "bg-amber-500/15",   border: "border-amber-500/30",   icon: Minus },
  negative: { text: "text-red-400",     bg: "bg-red-500/15",     border: "border-red-500/30",     icon: AlertTriangle },
};

const URGENCY_STYLES = {
  critical: "bg-red-500/20 text-red-300 border-red-500/40",
  high:     "bg-orange-500/20 text-orange-300 border-orange-500/40",
  medium:   "bg-amber-500/20 text-amber-300 border-amber-500/40",
  low:      "bg-slate-700/50 text-slate-400 border-slate-600/40",
};

function ScoreBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="h-1.5 w-full bg-slate-700/60 rounded-full overflow-hidden">
      <div className={`h-full rounded-full transition-all duration-700 ${color}`} style={{ width: `${Math.round(value * 100)}%` }} />
    </div>
  );
}

function SentimentBadge({ sentiment }: { sentiment: "positive" | "neutral" | "negative" }) {
  const s = SENTIMENT_STYLES[sentiment];
  const Icon = s.icon;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${s.text} ${s.bg} ${s.border}`}>
      <Icon className="w-3 h-3" />
      {sentiment.toUpperCase()}
    </span>
  );
}

function ProbBar({ pos, neu, neg }: { pos: number; neu: number; neg: number }) {
  return (
    <div className="flex h-2 rounded-full overflow-hidden w-full gap-px">
      <div className="bg-emerald-500 transition-all duration-700" style={{ width: `${pos * 100}%` }} />
      <div className="bg-amber-400 transition-all duration-700"  style={{ width: `${neu * 100}%` }} />
      <div className="bg-red-500 transition-all duration-700"   style={{ width: `${neg * 100}%` }} />
    </div>
  );
}

// ─── Result Card ─────────────────────────────────────────────────────────────

function ResultCard({ result, index }: { result: Prediction; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const s = SENTIMENT_STYLES[result.sentiment] ?? SENTIMENT_STYLES.neutral;
  const tokens = result.explanation?.top_tokens ?? [];

  return (
    <div className={`border rounded-xl overflow-hidden transition-all ${s.border} ${s.bg}`}>
      {/* Header row */}
      <div className="flex items-start gap-3 p-4">
        <span className="text-xs text-slate-500 font-mono mt-0.5 shrink-0">#{String(index + 1).padStart(3, "0")}</span>
        <div className="flex-1 min-w-0">
          <p className="text-sm text-slate-200 leading-snug line-clamp-2">{result.text}</p>
          <div className="flex flex-wrap items-center gap-2 mt-2">
            <SentimentBadge sentiment={result.sentiment} />
            <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs border ${URGENCY_STYLES[result.urgency]}`}>
              {result.urgency}
            </span>
            {result.attributes.slice(0, 3).map((a) => (
              <span key={a} className="px-2 py-0.5 rounded-full text-xs bg-slate-700/50 text-slate-400 border border-slate-600/30">{a}</span>
            ))}
          </div>
        </div>
        <div className="text-right shrink-0">
          <p className={`text-lg font-bold ${s.text}`}>{(result.confidence * 100).toFixed(1)}%</p>
          <p className="text-xs text-slate-500">{result.latency_ms}ms</p>
        </div>
      </div>

      {/* Score bars */}
      <div className="px-4 pb-3 grid grid-cols-3 gap-3 text-xs text-slate-400">
        <div>
          <div className="flex justify-between mb-1"><span>Severity</span><span>{result.severity_score.toFixed(2)}</span></div>
          <ScoreBar value={result.severity_score} color="bg-orange-500" />
        </div>
        <div>
          <div className="flex justify-between mb-1"><span>Impact</span><span>{result.business_impact_score.toFixed(2)}</span></div>
          <ScoreBar value={result.business_impact_score} color="bg-violet-500" />
        </div>
        <div>
          <div className="flex justify-between mb-1"><span>Urgency</span><span>{result.urgency_score.toFixed(2)}</span></div>
          <ScoreBar value={result.urgency_score} color="bg-red-500" />
        </div>
      </div>

      {/* Probability bar */}
      <div className="px-4 pb-3">
        <ProbBar pos={result.probabilities.positive} neu={result.probabilities.neutral} neg={result.probabilities.negative} />
        <div className="flex justify-between text-xs text-slate-500 mt-1">
          <span className="text-emerald-500">pos {(result.probabilities.positive * 100).toFixed(0)}%</span>
          <span className="text-amber-400">neu {(result.probabilities.neutral * 100).toFixed(0)}%</span>
          <span className="text-red-400">neg {(result.probabilities.negative * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Expand tokens */}
      {tokens.length > 0 && (
        <>
          <button
            onClick={() => setExpanded(!expanded)}
            className="w-full flex items-center justify-center gap-1 py-2 text-xs text-slate-500 hover:text-slate-300 border-t border-slate-700/40 transition-colors"
          >
            {expanded ? <><ChevronUp className="w-3 h-3" /> Hide tokens</> : <><ChevronDown className="w-3 h-3" /> Show token saliency</>}
          </button>
          {expanded && (
            <div className="px-4 pb-4 flex flex-wrap gap-2">
              {tokens.map((t, i) => (
                <span key={i} className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-slate-800 border border-slate-700/50 text-xs text-slate-300">
                  <span className="font-mono">{t.token}</span>
                  <span className="text-slate-500">{(t.importance * 100).toFixed(1)}%</span>
                </span>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ─── Summary Panel ────────────────────────────────────────────────────────────

function SummaryPanel({ summary, filename }: { summary: BatchSummary; filename?: string }) {
  const total = summary.total;
  return (
    <div className="bg-slate-800/60 border border-slate-700/60 rounded-xl p-5 mb-4">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-semibold text-white">
          {filename ? `📄 ${filename}` : "Batch"} — {total} messages analysed
        </h3>
        {summary.critical_count > 0 && (
          <span className="flex items-center gap-1.5 px-2.5 py-1 bg-red-500/20 text-red-300 border border-red-500/30 rounded-full text-xs font-semibold">
            <AlertTriangle className="w-3 h-3" />
            {summary.critical_count} critical
          </span>
        )}
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: "Positive", value: `${summary.positive} (${(summary.positive_rate * 100).toFixed(1)}%)`, color: "text-emerald-400" },
          { label: "Negative", value: `${summary.negative} (${(summary.negative_rate * 100).toFixed(1)}%)`, color: "text-red-400" },
          { label: "Avg Confidence", value: `${(summary.avg_confidence * 100).toFixed(1)}%`, color: "text-indigo-400" },
          { label: "Avg Impact", value: summary.avg_business_impact.toFixed(3), color: "text-violet-400" },
        ].map(({ label, value, color }) => (
          <div key={label} className="bg-slate-900/60 rounded-lg p-3">
            <p className="text-xs text-slate-500 mb-1">{label}</p>
            <p className={`text-base font-bold ${color}`}>{value}</p>
          </div>
        ))}
      </div>
      {/* Distribution bar */}
      <div className="mt-4">
        <ProbBar pos={summary.positive_rate} neu={1 - summary.positive_rate - summary.negative_rate} neg={summary.negative_rate} />
      </div>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function DemoPage() {
  // Single predict
  const [text, setText]           = useState("");
  const [loading, setLoading]     = useState(false);
  const [singleResult, setSingle] = useState<Prediction | null>(null);
  const [singleError, setSingleError] = useState("");

  // Batch / CSV
  const [batchText, setBatchText]     = useState("");
  const [batchResults, setBatchResults] = useState<Prediction[]>([]);
  const [batchSummary, setBatchSummary] = useState<BatchSummary | null>(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchError, setBatchError]   = useState("");

  // CSV upload
  const [csvFile, setCsvFile]         = useState<File | null>(null);
  const [csvLoading, setCsvLoading]   = useState(false);
  const [csvError, setCsvError]       = useState("");
  const [csvFilename, setCsvFilename] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  // ── Single predict ──────────────────────────────────────────────────────────
  const handleSingle = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true);
    setSingleError("");
    setSingle(null);
    try {
      const { data } = await axios.post(`${API}/predict`, { text: text.trim(), source: "demo" });
      setSingle(data);
    } catch (e: any) {
      setSingleError(e.response?.data?.detail ?? "API unavailable — is docker compose up?");
    } finally {
      setLoading(false);
    }
  }, [text]);

  // ── Batch predict ──────────────────────────────────────────────────────────
  const handleBatch = useCallback(async () => {
    const lines = batchText.split("\n").map(l => l.trim()).filter(Boolean);
    if (!lines.length) return;
    setBatchLoading(true);
    setBatchError("");
    setBatchResults([]);
    setBatchSummary(null);
    try {
      const { data } = await axios.post(`${API}/predict/batch`, { texts: lines, source: "demo" });
      setBatchResults(data.predictions ?? []);
      setBatchSummary(data.summary ?? null);
    } catch (e: any) {
      setBatchError(e.response?.data?.detail ?? "API unavailable — is docker compose up?");
    } finally {
      setBatchLoading(false);
    }
  }, [batchText]);

  // ── CSV upload ──────────────────────────────────────────────────────────────
  const handleCsvUpload = useCallback(async () => {
    if (!csvFile) return;
    setCsvLoading(true);
    setCsvError("");
    setBatchResults([]);
    setBatchSummary(null);
    setCsvFilename(csvFile.name);
    const form = new FormData();
    form.append("file", csvFile);
    try {
      const { data } = await axios.post(`${API}/predict/csv`, form, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 300_000,
      });
      setBatchResults(data.predictions ?? []);
      setBatchSummary(data.summary ?? null);
    } catch (e: any) {
      setCsvError(e.response?.data?.detail ?? "Upload failed — check file format and API status.");
    } finally {
      setCsvLoading(false);
    }
  }, [csvFile]);

  const clearAll = () => {
    setText(""); setSingle(null); setSingleError("");
    setBatchText(""); setBatchResults([]); setBatchSummary(null); setBatchError("");
    setCsvFile(null); setCsvError(""); setCsvFilename("");
  };

  return (
    <div className="space-y-8 pb-10">

      {/* Page header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Live Demo</h1>
          <p className="text-slate-400 text-sm mt-1">
            Type text, paste a batch, or upload a CSV — see the full intelligence pipeline in action
          </p>
        </div>
        <button
          onClick={clearAll}
          className="flex items-center gap-2 px-3 py-1.5 text-xs text-slate-400 hover:text-white border border-slate-700 hover:border-slate-500 rounded-lg transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Clear all
        </button>
      </div>

      {/* ── Section 1: Single predict ─────────────────────────────────────── */}
      <section>
        <div className="flex items-center gap-2 mb-3">
          <div className="w-6 h-6 rounded bg-indigo-600/30 flex items-center justify-center">
            <Zap className="w-3.5 h-3.5 text-indigo-400" />
          </div>
          <h2 className="text-sm font-semibold text-white">Single Message</h2>
          <span className="text-xs text-slate-500">Type any customer feedback</span>
        </div>

        <div className="bg-slate-800/50 border border-slate-700/60 rounded-xl p-4">
          <textarea
            value={text}
            onChange={e => setText(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleSingle(); }}
            placeholder="e.g. The flight was delayed 3 hours and the staff were completely unhelpful..."
            rows={3}
            className="w-full bg-slate-900/60 border border-slate-700/60 rounded-lg px-4 py-3 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-indigo-500/50 resize-none"
          />
          <div className="flex items-center justify-between mt-3">
            <span className="text-xs text-slate-600">{text.length} chars · Ctrl+Enter to submit</span>
            <button
              onClick={handleSingle}
              disabled={!text.trim() || loading}
              className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
            >
              {loading ? <Activity className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
              {loading ? "Analysing…" : "Analyse"}
            </button>
          </div>
        </div>

        {singleError && (
          <div className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-xs text-red-400 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" /> {singleError}
          </div>
        )}

        {singleResult && (
          <div className="mt-4">
            <ResultCard result={singleResult} index={0} />
          </div>
        )}

        {/* Quick-fire examples */}
        <div className="mt-3">
          <p className="text-xs text-slate-600 mb-2">Try these:</p>
          <div className="flex flex-wrap gap-2">
            {[
              "Absolutely love the new features — this app has transformed how we work!",
              "Third outage this month. We're losing customers because of your unreliable service.",
              "The onboarding was confusing but the product itself is decent once you figure it out.",
              "Your support team left me on hold for 90 minutes and then hung up.",
              "Smooth checkout, fast delivery. Exactly as advertised.",
            ].map((ex) => (
              <button
                key={ex}
                onClick={() => setText(ex)}
                className="px-2.5 py-1 text-xs text-slate-400 hover:text-slate-200 bg-slate-800/40 hover:bg-slate-700/60 border border-slate-700/40 rounded-lg transition-colors text-left truncate max-w-xs"
              >
                {ex.slice(0, 50)}…
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* ── Section 2: Batch text input ───────────────────────────────────── */}
      <section>
        <div className="flex items-center gap-2 mb-3">
          <div className="w-6 h-6 rounded bg-violet-600/30 flex items-center justify-center">
            <Target className="w-3.5 h-3.5 text-violet-400" />
          </div>
          <h2 className="text-sm font-semibold text-white">Batch Input</h2>
          <span className="text-xs text-slate-500">One message per line — up to 500</span>
        </div>

        <div className="bg-slate-800/50 border border-slate-700/60 rounded-xl p-4">
          <textarea
            value={batchText}
            onChange={e => setBatchText(e.target.value)}
            placeholder={"Great flight, very friendly crew!\nTotal disaster — lost my luggage again.\nService was fine, nothing special.\nThe app crashed during checkout. Unacceptable!"}
            rows={6}
            className="w-full bg-slate-900/60 border border-slate-700/60 rounded-lg px-4 py-3 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-violet-500/50 resize-none font-mono"
          />
          <div className="flex items-center justify-between mt-3">
            <span className="text-xs text-slate-600">
              {batchText.split("\n").filter(l => l.trim()).length} messages
            </span>
            <button
              onClick={handleBatch}
              disabled={!batchText.trim() || batchLoading}
              className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
            >
              {batchLoading ? <Activity className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
              {batchLoading ? "Analysing…" : "Analyse Batch"}
            </button>
          </div>
        </div>

        {batchError && (
          <div className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-xs text-red-400 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" /> {batchError}
          </div>
        )}
      </section>

      {/* ── Section 3: CSV Upload ─────────────────────────────────────────── */}
      <section>
        <div className="flex items-center gap-2 mb-3">
          <div className="w-6 h-6 rounded bg-emerald-600/30 flex items-center justify-center">
            <Upload className="w-3.5 h-3.5 text-emerald-400" />
          </div>
          <h2 className="text-sm font-semibold text-white">CSV Upload</h2>
          <span className="text-xs text-slate-500">Upload Tweets.csv or any feedback CSV · up to 10,000 rows</span>
        </div>

        <div
          className="bg-slate-800/50 border-2 border-dashed border-slate-700/60 hover:border-emerald-500/40 rounded-xl p-8 text-center cursor-pointer transition-colors"
          onClick={() => fileRef.current?.click()}
          onDragOver={e => e.preventDefault()}
          onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f?.name.endsWith(".csv")) setCsvFile(f); }}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".csv"
            className="hidden"
            onChange={e => setCsvFile(e.target.files?.[0] ?? null)}
          />
          {csvFile ? (
            <div className="flex items-center justify-center gap-3">
              <FileText className="w-5 h-5 text-emerald-400" />
              <span className="text-sm text-emerald-300 font-medium">{csvFile.name}</span>
              <span className="text-xs text-slate-500">({(csvFile.size / 1024).toFixed(1)} KB)</span>
              <button
                onClick={e => { e.stopPropagation(); setCsvFile(null); }}
                className="text-slate-600 hover:text-slate-300"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <>
              <Upload className="w-8 h-8 text-slate-600 mx-auto mb-2" />
              <p className="text-sm text-slate-400">Drop your CSV here or click to browse</p>
              <p className="text-xs text-slate-600 mt-1">
                Auto-detects columns: text, tweet, message, review, feedback
              </p>
            </>
          )}
        </div>

        {csvFile && (
          <div className="mt-3 flex justify-end">
            <button
              onClick={handleCsvUpload}
              disabled={csvLoading}
              className="flex items-center gap-2 px-5 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
            >
              {csvLoading ? <Activity className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
              {csvLoading ? "Processing…" : "Upload & Analyse"}
            </button>
          </div>
        )}

        {csvError && (
          <div className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-xs text-red-400 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0" /> {csvError}
          </div>
        )}
      </section>

      {/* ── Results ──────────────────────────────────────────────────────── */}
      {(batchResults.length > 0 || batchSummary) && (
        <section>
          <div className="flex items-center gap-2 mb-4">
            <div className="w-6 h-6 rounded bg-slate-700/60 flex items-center justify-center">
              <Clock className="w-3.5 h-3.5 text-slate-400" />
            </div>
            <h2 className="text-sm font-semibold text-white">Results</h2>
          </div>

          {batchSummary && <SummaryPanel summary={batchSummary} filename={csvFilename || undefined} />}

          <div className="space-y-3 max-h-[70vh] overflow-y-auto pr-1">
            {batchResults.map((r, i) => (
              <ResultCard key={i} result={r} index={i} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
