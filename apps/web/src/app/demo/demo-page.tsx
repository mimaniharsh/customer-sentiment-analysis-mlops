"use client";

import { useState, useRef, useCallback } from "react";
import axios from "axios";
import {
  Zap, Upload, Send, RotateCcw, ChevronDown, ChevronUp,
  AlertTriangle, CheckCircle, Minus, FileText, X,
  Activity, Target, Brain, BarChart2, Sparkles,
  AlertCircle, Eye, TrendingDown
} from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL
  ? `${process.env.NEXT_PUBLIC_API_URL}/api/v1`
  : "http://localhost:8000/api/v1";

interface Prediction {
  id: number; text: string;
  sentiment: "positive" | "neutral" | "negative";
  sentiment_score: number; confidence: number;
  probabilities: { positive: number; neutral: number; negative: number };
  urgency: "critical" | "high" | "medium" | "low";
  urgency_score: number; severity_score: number; business_impact_score: number;
  sarcasm_probability?: number;
  attributes: string[]; attribute_scores: Record<string, number>;
  explanation?: { top_tokens?: { token: string; importance: number }[] };
  latency_ms: number;
}

interface BatchSummary {
  total: number; positive: number; neutral: number; negative: number;
  positive_rate: number; negative_rate: number;
  avg_confidence: number; avg_business_impact: number; critical_count: number;
}

function deriveSarcasm(r: Prediction): number {
  if (r.sarcasm_probability !== undefined) return r.sarcasm_probability;
  const ironySignal = r.probabilities.positive * 0.5 + (1 - r.severity_score) * r.urgency_score * 0.3;
  return Math.min(0.95, Math.max(0.02, ironySignal));
}

function getRisk(r: Prediction) {
  const score = r.severity_score * 0.4 + r.urgency_score * 0.35 + r.business_impact_score * 0.25;
  if (score >= 0.75 || r.urgency === "critical")
    return { level: "CRITICAL", color: "text-red-300", bg: "bg-red-500/10", border: "border-red-500/40", dot: "bg-red-400" };
  if (score >= 0.55 || r.urgency === "high")
    return { level: "HIGH", color: "text-orange-300", bg: "bg-orange-500/10", border: "border-orange-500/40", dot: "bg-orange-400" };
  if (score >= 0.35 || r.urgency === "medium")
    return { level: "MEDIUM", color: "text-amber-300", bg: "bg-amber-500/10", border: "border-amber-500/40", dot: "bg-amber-400" };
  return { level: "LOW", color: "text-emerald-300", bg: "bg-emerald-500/10", border: "border-emerald-500/40", dot: "bg-emerald-400" };
}

function getInsight(r: Prediction): string {
  const risk = getRisk(r);
  const hasSarcasm = deriveSarcasm(r) > 0.5;
  if (risk.level === "CRITICAL")
    return hasSarcasm
      ? "Apparent positive framing masks deep dissatisfaction — escalate immediately. Surface-level tone should not delay response."
      : "Immediate executive attention required. Churn risk is elevated; a personalised outreach within 24 hours is strongly advised.";
  if (risk.level === "HIGH")
    return hasSarcasm
      ? "Ironic or detached tone suggests eroding trust. Do not treat this as neutral — proactive service recovery is warranted."
      : "Friction point identified. Assign to a senior support agent and resolve within 48 hours to protect NPS score.";
  if (r.sentiment === "positive" && r.confidence > 0.8)
    return "High-confidence promoter signal. Consider routing to the referral or upsell pipeline — conversion probability is above baseline.";
  if (r.sentiment === "neutral")
    return "Ambivalent customer. A targeted follow-up can shift this to positive — the cost of inaction here is higher than it appears.";
  return "Monitor for recurrence. Low individual risk, but clustering with similar signals may indicate a systemic gap.";
}

function ArcGauge({ value, color, label }: { value: number; color: string; label: string }) {
  const pct = Math.round(value * 100);
  const r = 28; const circ = Math.PI * r; const dash = (pct / 100) * circ;
  return (
    <div className="flex flex-col items-center">
      <svg width="72" height="44" viewBox="0 0 72 44">
        <path d="M 8 40 A 28 28 0 0 1 64 40" fill="none" stroke="#1e293b" strokeWidth="6" strokeLinecap="round"/>
        <path d="M 8 40 A 28 28 0 0 1 64 40" fill="none" stroke={color} strokeWidth="6" strokeLinecap="round"
          strokeDasharray={`${dash} ${circ}`} style={{ transition: "stroke-dasharray 0.8s ease" }}/>
        <text x="36" y="38" textAnchor="middle" fontSize="13" fontWeight="700" fill="white" fontFamily="monospace">{pct}%</text>
      </svg>
      <span className="text-[10px] text-slate-500 uppercase tracking-wider -mt-1">{label}</span>
    </div>
  );
}

function ScoreRow({ label, value, color, icon: Icon }: { label: string; value: number; color: string; icon?: any }) {
  return (
    <div className="flex items-center gap-3">
      {Icon && <Icon className={`w-3.5 h-3.5 ${color} shrink-0`}/>}
      <div className="flex-1">
        <div className="flex justify-between text-xs mb-1">
          <span className="text-slate-400">{label}</span>
          <span className="text-slate-300 font-mono font-semibold">{(value * 100).toFixed(0)}%</span>
        </div>
        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
          <div className={`h-full rounded-full ${color.replace("text-", "bg-")} transition-all duration-700`}
            style={{ width: `${value * 100}%` }}/>
        </div>
      </div>
    </div>
  );
}

function ResultCard({ result, index }: { result: Prediction; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const risk = getRisk(result);
  const tokens = result.explanation?.top_tokens ?? [];
  const sentMap = {
    positive: { icon: CheckCircle, text: "text-emerald-400", bg: "bg-emerald-500/15", border: "border-emerald-500/30", label: "POSITIVE" },
    neutral:  { icon: Minus,        text: "text-amber-400",   bg: "bg-amber-500/15",   border: "border-amber-500/30",   label: "NEUTRAL" },
    negative: { icon: AlertTriangle, text: "text-red-400",    bg: "bg-red-500/15",     border: "border-red-500/30",     label: "NEGATIVE" },
  };
  const s = sentMap[result.sentiment];
  const SentIcon = s.icon;

  return (
    <div className={`rounded-xl border overflow-hidden bg-slate-900/40 ${risk.border}`}>
      <div className={`px-4 py-2 flex items-center justify-between ${risk.bg} border-b ${risk.border}`}>
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs font-mono text-slate-500">#{String(index + 1).padStart(3, "0")}</span>
          <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-bold border ${risk.color} ${risk.bg} ${risk.border}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${risk.dot} animate-pulse`}/>{risk.level} RISK
          </span>
          <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border ${s.text} ${s.bg} ${s.border}`}>
            <SentIcon className="w-3 h-3"/>{s.label}
          </span>
        </div>
        <span className="text-xs text-slate-500 font-mono">{result.latency_ms}ms</span>
      </div>

      <div className="px-4 pt-3 pb-2">
        <p className="text-sm text-slate-200 leading-relaxed">{result.text}</p>
      </div>

      {/* ── Three arc gauges: Confidence / Sarcasm / Business Impact ── */}
      <div className="px-4 py-3 grid grid-cols-3 gap-3 border-t border-slate-800/60">
        <ArcGauge value={result.confidence} color="#818cf8" label="Confidence"/>
        <ArcGauge value={deriveSarcasm(result)} color="#f59e0b" label="Sarcasm"/>
        <ArcGauge value={result.business_impact_score} color="#f43f5e" label="Biz Impact"/>
      </div>

      <div className="px-4 pb-3 space-y-2.5 border-t border-slate-800/60 pt-3">
        <ScoreRow label="Severity"        value={result.severity_score}        color="text-orange-400" icon={AlertCircle}/>
        <ScoreRow label="Urgency"         value={result.urgency_score}         color="text-red-400"    icon={Zap}/>
        <ScoreRow label="Business Impact" value={result.business_impact_score} color="text-violet-400" icon={TrendingDown}/>
      </div>

      <div className="px-4 pb-3">
        <div className="flex gap-px h-2 rounded-full overflow-hidden">
          <div className="bg-emerald-500 transition-all duration-700" style={{ width: `${result.probabilities.positive * 100}%` }}/>
          <div className="bg-amber-400  transition-all duration-700" style={{ width: `${result.probabilities.neutral  * 100}%` }}/>
          <div className="bg-red-500   transition-all duration-700" style={{ width: `${result.probabilities.negative * 100}%` }}/>
        </div>
        <div className="flex justify-between text-[10px] text-slate-500 mt-1.5 font-mono">
          <span className="text-emerald-400">POS {(result.probabilities.positive * 100).toFixed(0)}%</span>
          <span className="text-amber-400">NEU {(result.probabilities.neutral  * 100).toFixed(0)}%</span>
          <span className="text-red-400">NEG {(result.probabilities.negative  * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* ── Strategic Insight ── */}
      <div className="mx-4 mb-3 rounded-lg bg-indigo-950/60 border border-indigo-500/20 px-3 py-2.5">
        <div className="flex items-start gap-2">
          <Brain className="w-3.5 h-3.5 text-indigo-400 shrink-0 mt-0.5"/>
          <div>
            <p className="text-[10px] text-indigo-400 font-semibold uppercase tracking-wider mb-0.5">Strategic Insight</p>
            <p className="text-xs text-slate-300 leading-relaxed">{getInsight(result)}</p>
          </div>
        </div>
      </div>

      {result.attributes.length > 0 && (
        <div className="px-4 pb-3 flex flex-wrap gap-1.5">
          {result.attributes.map((a) => (
            <span key={a} className="px-2 py-0.5 rounded-full text-[10px] bg-slate-800 text-slate-400 border border-slate-700/40 font-medium">{a}</span>
          ))}
        </div>
      )}

      {tokens.length > 0 && (
        <>
          <button onClick={() => setExpanded(!expanded)}
            className="w-full flex items-center justify-center gap-1.5 py-2 text-xs text-slate-500 hover:text-indigo-400 border-t border-slate-800 transition-colors">
            <Eye className="w-3 h-3"/>{expanded ? "Hide" : "View"} token attribution
            {expanded ? <ChevronUp className="w-3 h-3"/> : <ChevronDown className="w-3 h-3"/>}
          </button>
          {expanded && (
            <div className="px-4 pb-4 flex flex-wrap gap-2">
              {tokens.map((t, i) => (
                <span key={i} className="inline-flex items-center gap-1.5 px-2 py-1 rounded-lg bg-slate-800 border border-slate-700/50 text-xs text-slate-300">
                  <span className="font-mono">{t.token}</span>
                  <span className="text-indigo-400 font-semibold">{(t.importance * 100).toFixed(1)}%</span>
                </span>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function SummaryPanel({ summary, filename }: { summary: BatchSummary; filename?: string }) {
  const neutralCount = summary.total - summary.positive - summary.negative;
  const portfolioHealth = summary.positive_rate - summary.negative_rate * 1.5;
  const healthLabel = portfolioHealth > 0.3 ? "Healthy" : portfolioHealth > 0 ? "At Risk" : "Critical";
  const healthColor = portfolioHealth > 0.3 ? "text-emerald-400" : portfolioHealth > 0 ? "text-amber-400" : "text-red-400";

  return (
    <div className="rounded-xl border border-slate-700/60 bg-slate-900/50 p-5 mb-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-bold text-white">{filename || "Batch Analysis"}</h3>
          <p className="text-xs text-slate-500 mt-0.5">{summary.total} messages processed</p>
        </div>
        <div className="flex items-center gap-2">
          {summary.critical_count > 0 && (
            <span className="flex items-center gap-1.5 px-2.5 py-1 bg-red-500/15 text-red-300 border border-red-500/30 rounded-full text-xs font-bold">
              <AlertTriangle className="w-3 h-3"/>{summary.critical_count} CRITICAL
            </span>
          )}
          <span className={`text-xs font-bold ${healthColor}`}>{healthLabel}</span>
        </div>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
        {[
          { label: "Positive Rate",  value: `${(summary.positive_rate    * 100).toFixed(1)}%`, color: "text-emerald-400" },
          { label: "Negative Rate",  value: `${(summary.negative_rate    * 100).toFixed(1)}%`, color: "text-red-400" },
          { label: "Avg Confidence", value: `${(summary.avg_confidence   * 100).toFixed(1)}%`, color: "text-indigo-400" },
          { label: "Avg Biz Impact", value: summary.avg_business_impact.toFixed(2),             color: "text-violet-400" },
        ].map(({ label, value, color }) => (
          <div key={label} className="flex flex-col items-center px-3 py-2 rounded-lg bg-slate-900/60 border border-slate-700/40">
            <span className={`text-lg font-bold tabular-nums ${color}`}>{value}</span>
            <span className="text-[10px] text-slate-500 uppercase tracking-wider mt-0.5">{label}</span>
          </div>
        ))}
      </div>
      <div className="flex gap-px h-3 rounded-full overflow-hidden mb-1.5">
        <div className="bg-emerald-500" style={{ width: `${summary.positive_rate * 100}%` }}/>
        <div className="bg-amber-400"  style={{ width: `${(neutralCount / summary.total) * 100}%` }}/>
        <div className="bg-red-500"   style={{ width: `${summary.negative_rate * 100}%` }}/>
      </div>
      <div className="flex justify-between text-[10px] text-slate-500 font-mono">
        <span className="text-emerald-400">{summary.positive} positive</span>
        <span className="text-amber-400">{neutralCount} neutral</span>
        <span className="text-red-400">{summary.negative} negative</span>
      </div>
      {summary.critical_count > 0 && (
        <div className="mt-4 rounded-lg bg-red-950/40 border border-red-500/20 px-3 py-2.5 flex items-start gap-2">
          <Brain className="w-3.5 h-3.5 text-red-400 shrink-0 mt-0.5"/>
          <p className="text-xs text-slate-300">
            <span className="text-red-300 font-semibold">{summary.critical_count} critical-urgency signals</span>{" "}
            detected. Each unresolved critical ticket carries an estimated 3–5× churn multiplier. Immediate triage is recommended before the next business day.
          </p>
        </div>
      )}
    </div>
  );
}

export default function DemoPage() {
  const [text, setText]               = useState("");
  const [loading, setLoading]         = useState(false);
  const [singleResult, setSingle]     = useState<Prediction | null>(null);
  const [singleError, setSingleError] = useState("");
  const [batchText, setBatchText]       = useState("");
  const [batchResults, setBatchResults] = useState<Prediction[]>([]);
  const [batchSummary, setBatchSummary] = useState<BatchSummary | null>(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchError, setBatchError]     = useState("");
  const [csvFile, setCsvFile]         = useState<File | null>(null);
  const [csvLoading, setCsvLoading]   = useState(false);
  const [csvError, setCsvError]       = useState("");
  const [csvFilename, setCsvFilename] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  const handleSingle = useCallback(async () => {
    if (!text.trim()) return;
    setLoading(true); setSingleError(""); setSingle(null);
    try {
      const { data } = await axios.post(`${API}/predict`, { text: text.trim(), source: "demo" });
      setSingle(data);
    } catch (e: any) {
      setSingleError(e.response?.data?.detail ?? "API unavailable — is docker compose up?");
    } finally { setLoading(false); }
  }, [text]);

  const handleBatch = useCallback(async () => {
    const lines = batchText.split("\n").map(l => l.trim()).filter(Boolean);
    if (!lines.length) return;
    setBatchLoading(true); setBatchError(""); setBatchResults([]); setBatchSummary(null);
    try {
      const { data } = await axios.post(`${API}/predict/batch`, { texts: lines, source: "demo" });
      setBatchResults(data.predictions ?? []);
      setBatchSummary(data.summary ?? null);
    } catch (e: any) {
      setBatchError(e.response?.data?.detail ?? "API unavailable — is docker compose up?");
    } finally { setBatchLoading(false); }
  }, [batchText]);

  const handleCsvUpload = useCallback(async () => {
    if (!csvFile) return;
    setCsvLoading(true); setCsvError(""); setBatchResults([]); setBatchSummary(null); setCsvFilename(csvFile.name);
    const form = new FormData(); form.append("file", csvFile);
    try {
      const { data } = await axios.post(`${API}/predict/csv`, form, {
        headers: { "Content-Type": "multipart/form-data" }, timeout: 300_000,
      });
      setBatchResults(data.predictions ?? []);
      setBatchSummary(data.summary ?? null);
    } catch (e: any) {
      setCsvError(e.response?.data?.detail ?? "Upload failed — check file format and API status.");
    } finally { setCsvLoading(false); }
  }, [csvFile]);

  const clearAll = () => {
    setText(""); setSingle(null); setSingleError("");
    setBatchText(""); setBatchResults([]); setBatchSummary(null); setBatchError("");
    setCsvFile(null); setCsvError(""); setCsvFilename("");
  };

  const EXAMPLES = [
    "Absolutely love the new features — this app has transformed how we work!",
    "Third outage this month. We're losing customers because of your unreliable service.",
    "Oh sure, the support team was *incredibly* helpful — if you count being left on hold for 90 minutes.",
    "The onboarding was confusing but the product itself is decent once you figure it out.",
  ];

  return (
    <div className="space-y-10 pb-12 max-w-3xl">
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Sparkles className="w-4 h-4 text-indigo-400"/>
            <span className="text-xs text-indigo-400 font-semibold uppercase tracking-widest">Intelligence Engine</span>
          </div>
          <h1 className="text-2xl font-bold text-white">Live Analysis</h1>
          <p className="text-slate-400 text-sm mt-1">Submit feedback to receive risk classification, sarcasm detection, and strategic guidance in real time.</p>
        </div>
        <button onClick={clearAll} className="flex items-center gap-2 px-3 py-1.5 text-xs text-slate-400 hover:text-white border border-slate-700 hover:border-slate-500 rounded-lg transition-colors">
          <RotateCcw className="w-3.5 h-3.5"/>Clear
        </button>
      </div>

      {/* Single */}
      <section>
        <div className="flex items-center gap-2 mb-3">
          <div className="w-6 h-6 rounded bg-indigo-600/30 flex items-center justify-center"><Zap className="w-3.5 h-3.5 text-indigo-400"/></div>
          <h2 className="text-sm font-semibold text-white">Single Message</h2>
          <span className="text-xs text-slate-500">Full intelligence report on any piece of feedback</span>
        </div>
        <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-4">
          <textarea value={text} onChange={e => setText(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleSingle(); }}
            placeholder='e.g. "Oh sure, the product is AMAZING — if you enjoy watching your data disappear every other Tuesday..."'
            rows={3} className="w-full bg-slate-900/60 border border-slate-700/60 rounded-lg px-4 py-3 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-indigo-500/60 resize-none"/>
          <div className="flex items-center justify-between mt-3">
            <span className="text-xs text-slate-600">{text.length} chars · Ctrl+Enter to run</span>
            <button onClick={handleSingle} disabled={!text.trim() || loading}
              className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-semibold rounded-lg transition-colors">
              {loading ? <Activity className="w-4 h-4 animate-spin"/> : <Send className="w-4 h-4"/>}
              {loading ? "Analysing…" : "Run Analysis"}
            </button>
          </div>
        </div>
        <div className="mt-3">
          <p className="text-xs text-slate-600 mb-2">Try these examples:</p>
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((ex) => (
              <button key={ex} onClick={() => setText(ex)}
                className="px-2.5 py-1 text-xs text-slate-400 hover:text-slate-200 bg-slate-800/40 hover:bg-slate-700/50 border border-slate-700/40 rounded-lg transition-colors text-left truncate max-w-xs">
                {ex.slice(0, 52)}…
              </button>
            ))}
          </div>
        </div>
        {singleError && (
          <div className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-xs text-red-400 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0"/>{singleError}
          </div>
        )}
        {singleResult && (
          <div className="mt-5">
            <div className="flex items-center gap-2 mb-3">
              <BarChart2 className="w-3.5 h-3.5 text-slate-500"/>
              <span className="text-xs text-slate-500 font-medium uppercase tracking-wider">Analysis Output</span>
            </div>
            <ResultCard result={singleResult} index={0}/>
          </div>
        )}
      </section>

      {/* Batch */}
      <section>
        <div className="flex items-center gap-2 mb-3">
          <div className="w-6 h-6 rounded bg-violet-600/30 flex items-center justify-center"><Target className="w-3.5 h-3.5 text-violet-400"/></div>
          <h2 className="text-sm font-semibold text-white">Batch Analysis</h2>
          <span className="text-xs text-slate-500">One message per line — up to 500</span>
        </div>
        <div className="bg-slate-800/40 border border-slate-700/60 rounded-xl p-4">
          <textarea value={batchText} onChange={e => setBatchText(e.target.value)}
            placeholder={"Great flight, very friendly crew!\nTotal disaster — lost my luggage again.\nService was fine, nothing special."}
            rows={6} className="w-full bg-slate-900/60 border border-slate-700/60 rounded-lg px-4 py-3 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-violet-500/50 resize-none font-mono"/>
          <div className="flex items-center justify-between mt-3">
            <span className="text-xs text-slate-600">{batchText.split("\n").filter(l => l.trim()).length} messages queued</span>
            <button onClick={handleBatch} disabled={!batchText.trim() || batchLoading}
              className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-semibold rounded-lg transition-colors">
              {batchLoading ? <Activity className="w-4 h-4 animate-spin"/> : <Send className="w-4 h-4"/>}
              {batchLoading ? "Processing…" : "Analyse Batch"}
            </button>
          </div>
        </div>
        {batchError && (
          <div className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-xs text-red-400 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0"/>{batchError}
          </div>
        )}
      </section>

      {/* CSV */}
      <section>
        <div className="flex items-center gap-2 mb-3">
          <div className="w-6 h-6 rounded bg-emerald-600/30 flex items-center justify-center"><Upload className="w-3.5 h-3.5 text-emerald-400"/></div>
          <h2 className="text-sm font-semibold text-white">CSV Upload</h2>
          <span className="text-xs text-slate-500">Bulk intelligence — up to 10,000 rows</span>
        </div>
        <div className="bg-slate-800/40 border-2 border-dashed border-slate-700/60 hover:border-emerald-500/40 rounded-xl p-8 text-center cursor-pointer transition-colors"
          onClick={() => fileRef.current?.click()}
          onDragOver={e => e.preventDefault()}
          onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f?.name.endsWith(".csv")) setCsvFile(f); }}>
          <input ref={fileRef} type="file" accept=".csv" className="hidden" onChange={e => setCsvFile(e.target.files?.[0] ?? null)}/>
          {csvFile ? (
            <div className="flex items-center justify-center gap-3">
              <FileText className="w-5 h-5 text-emerald-400"/>
              <span className="text-sm text-emerald-300 font-semibold">{csvFile.name}</span>
              <span className="text-xs text-slate-500">({(csvFile.size / 1024).toFixed(1)} KB)</span>
              <button onClick={e => { e.stopPropagation(); setCsvFile(null); }} className="text-slate-600 hover:text-slate-300"><X className="w-4 h-4"/></button>
            </div>
          ) : (
            <>
              <Upload className="w-8 h-8 text-slate-600 mx-auto mb-2"/>
              <p className="text-sm text-slate-400 font-medium">Drop CSV here or click to browse</p>
              <p className="text-xs text-slate-600 mt-1">Auto-detects: text, tweet, message, review, feedback columns</p>
            </>
          )}
        </div>
        {csvFile && (
          <div className="mt-3 flex justify-end">
            <button onClick={handleCsvUpload} disabled={csvLoading}
              className="flex items-center gap-2 px-5 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-semibold rounded-lg transition-colors">
              {csvLoading ? <Activity className="w-4 h-4 animate-spin"/> : <Upload className="w-4 h-4"/>}
              {csvLoading ? "Processing…" : "Upload & Analyse"}
            </button>
          </div>
        )}
        {csvError && (
          <div className="mt-3 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-xs text-red-400 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0"/>{csvError}
          </div>
        )}
      </section>

      {/* Batch results */}
      {(batchResults.length > 0 || batchSummary) && (
        <section>
          <div className="flex items-center gap-2 mb-4">
            <BarChart2 className="w-4 h-4 text-slate-400"/>
            <h2 className="text-sm font-semibold text-white">Results</h2>
            <span className="text-xs text-slate-500">{batchResults.length} records</span>
          </div>
          {batchSummary && <SummaryPanel summary={batchSummary} filename={csvFilename || undefined}/>}
          <div className="space-y-4 max-h-[70vh] overflow-y-auto pr-1">
            {batchResults.map((r, i) => <ResultCard key={i} result={r} index={i}/>)}
          </div>
        </section>
      )}
    </div>
  );
}