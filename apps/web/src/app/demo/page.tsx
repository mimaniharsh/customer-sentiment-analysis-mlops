"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import axios from "axios";
import {
  Zap, Upload, Send, Square, Play, RotateCcw,
  AlertTriangle, CheckCircle, Minus, FileText, X,
  Activity, Radio, MessageSquare, Layers,
  TrendingUp, TrendingDown, ChevronDown, ChevronUp,
  Wifi, WifiOff, Brain, ShieldAlert, Eye,
} from "lucide-react";

const API = "/api/v1";

const TWEETS = [
  "Just landed — crew were incredible, smoothest flight I've had all year. Genuinely impressed.",
  "Third delay this week. Zero explanation, zero apology. I am done with this airline for good.",
  "The app crashed mid-checkout and I nearly missed my flight. Completely unacceptable.",
  "Upgraded to business class as a surprise! Best travel experience of my life. Thank you!",
  "Lost my luggage AGAIN. Second time this month. Where on earth is it??",
  "Customer support hung up on me after 45 mins on hold. Issue still unresolved.",
  "Flight was fine. On time, seat comfortable, nothing special either way.",
  "Seat wouldn't recline, entertainment broken, meal was cold. Complete disaster.",
  "Overcharged for bags with zero warning. Feeling completely ripped off right now.",
  "Gate changed three times with no announcement. Ran half a mile in the wrong direction.",
  "The flight attendant remembered my name from a previous flight. That's real service.",
  "Cancelled flight, no hotel, no meal voucher. Left stranded for 14 hours. Disgraceful.",
  "Oh wow, another delay. Just what I needed on a Monday morning. Truly phenomenal service.",
  "Brilliant. Lost my bag twice in one trip. You've really outdone yourselves.",
  "Check-in was seamless, security fast, boarded on time. Stress-free travel for once.",
  "Premium seats were worth every penny. Arrived refreshed and productive.",
  "Double charged on my card and nobody will pick up the phone. Furious.",
  "Three emails promising a refund. Still nothing after 6 weeks. Absolutely shocking.",
  "The new mobile boarding pass feature is a game changer. So smooth.",
  "Baggage carousel took 70 minutes. Why even have priority tags at this point?",
];

const SENTIMENT_CONFIG: Record<string, any> = {
  positive: { label: "Positive", textColor: "text-emerald-400", bg: "bg-emerald-500/8", border: "border-emerald-500/20", barColor: "bg-emerald-500", Icon: CheckCircle },
  neutral:  { label: "Neutral",  textColor: "text-slate-400",   bg: "bg-slate-500/8",   border: "border-slate-600/30",  barColor: "bg-slate-500",   Icon: Minus },
  negative: { label: "Negative", textColor: "text-red-400",     bg: "bg-red-500/8",     border: "border-red-500/20",    barColor: "bg-red-500",     Icon: AlertTriangle },
};

const RISK_CONFIG: Record<string, any> = {
  CRITICAL: { text: "text-red-300",     bg: "bg-red-950/60",     border: "border-red-500/40",     dot: "bg-red-500" },
  HIGH:     { text: "text-orange-300",  bg: "bg-orange-950/60",  border: "border-orange-500/40",  dot: "bg-orange-500" },
  MEDIUM:   { text: "text-amber-300",   bg: "bg-amber-950/60",   border: "border-amber-500/40",   dot: "bg-amber-500" },
  LOW:      { text: "text-emerald-400", bg: "bg-emerald-950/40", border: "border-emerald-500/30", dot: "bg-emerald-500" },
};

function GaugeBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
      <div className={`h-full rounded-full transition-all duration-700 ${color}`} style={{ width: `${Math.min(value * 100, 100).toFixed(1)}%` }} />
    </div>
  );
}

function SentimentDistBar({ pos, neu, neg }: { pos: number; neu: number; neg: number }) {
  return (
    <div className="flex h-2 w-full rounded-full overflow-hidden gap-px">
      <div className="bg-emerald-500 transition-all duration-700" style={{ width: `${pos * 100}%` }} />
      <div className="bg-slate-500 transition-all duration-700"   style={{ width: `${neu * 100}%` }} />
      <div className="bg-red-500 transition-all duration-700"     style={{ width: `${neg * 100}%` }} />
    </div>
  );
}

function MetricPill({ label, value, sub = null }: { label: string; value: string | number; sub?: string | null }) {
  return (
    <div className="flex flex-col items-center justify-center bg-slate-900/70 border border-slate-700/40 rounded-lg px-3 py-2.5 text-center min-w-[80px]">
      <p className="text-xs text-slate-500 mb-1 whitespace-nowrap">{label}</p>
      <p className="text-sm font-bold text-white tabular-nums">{value}</p>
      {sub && <p className="text-xs text-slate-600 mt-0.5">{sub}</p>}
    </div>
  );
}

function SarcasmMeter({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color    = value >= 0.6 ? "text-violet-400" : value >= 0.35 ? "text-amber-400" : "text-slate-500";
  const barColor = value >= 0.6 ? "bg-violet-500"   : value >= 0.35 ? "bg-amber-400"   : "bg-slate-600";
  const label    = value >= 0.6 ? "Likely sarcastic" : value >= 0.35 ? "Possible irony" : "Literal tone";
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5">
          <Brain className="w-3.5 h-3.5 text-slate-500" />
          <span className="text-xs text-slate-500">Sarcasm Indicator</span>
          <span className="text-xs text-slate-700">(signal mismatch)</span>
        </div>
        <span className={`text-sm font-bold tabular-nums ${color}`}>{pct}%</span>
      </div>
      <GaugeBar value={value} color={barColor} />
      <p className={`text-xs ${color}`}>{label}</p>
    </div>
  );
}

function RiskBadge({ risk }: { risk: any }) {
  const cfg = RISK_CONFIG[risk.tier] || RISK_CONFIG.LOW;
  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border ${cfg.bg} ${cfg.border}`}>
      <div className={`w-2 h-2 rounded-full ${cfg.dot} animate-pulse`} />
      <span className={`text-xs font-bold tracking-widest ${cfg.text}`}>{risk.tier}</span>
      <span className="text-xs text-slate-500 hidden lg:inline">— {risk.label}</span>
    </div>
  );
}

function IntelligenceCard({ r, index, isLatest = false }: { r: any; index: number; isLatest?: boolean }) {
  const [expanded, setExpanded] = useState(isLatest);
  const cfg = SENTIMENT_CONFIG[r.sentiment] || SENTIMENT_CONFIG.neutral;
  const Icon = cfg.Icon;
  const tokens = r.explanation?.top_tokens || [];
  const riskCfg = RISK_CONFIG[r.risk?.tier] || RISK_CONFIG.LOW;

  return (
    <div className={`border rounded-xl overflow-hidden transition-all duration-300 ${cfg.border} ${isLatest ? "ring-1 ring-indigo-500/30" : ""}`} style={{ background: "rgba(10,15,25,0.97)" }}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800/60">
        <div className="flex items-center gap-3">
          <Icon className={`w-4 h-4 shrink-0 ${cfg.textColor}`} />
          <span className={`text-sm font-bold uppercase tracking-wide ${cfg.textColor}`}>{cfg.label}</span>
          {isLatest && (
            <span className="flex items-center gap-1 px-2 py-0.5 bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 rounded-full text-xs">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse" />Live
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {r.risk && <RiskBadge risk={r.risk} />}
          <span className="text-xs text-slate-700">{r.latency_ms}ms</span>
        </div>
      </div>

      {/* Text */}
      <div className="px-4 py-3 border-b border-slate-800/40">
        <p className="text-sm text-slate-200 leading-relaxed">
          <span className="text-slate-700 font-mono text-xs mr-2">#{String(index + 1).padStart(3, "0")}</span>
          {r.text}
        </p>
      </div>

      {/* Metrics */}
      <div className="flex flex-wrap gap-2 px-4 py-3 border-b border-slate-800/40">
        <MetricPill label="Confidence"      value={`${(r.confidence * 100).toFixed(1)}%`} />
        <MetricPill label="Severity"        value={r.severity_score?.toFixed(2)} sub="/1.00" />
        <MetricPill label="Business Impact" value={r.business_impact_score?.toFixed(2)} sub="/1.00" />
        <MetricPill label="Urgency Score"   value={r.urgency_score?.toFixed(2)} sub="/1.00" />
        {r.attributes?.length > 0 && (
          <div className="flex items-center gap-1.5 flex-wrap">
            {r.attributes.slice(0, 4).map((a: any) => (
              <span key={a} className="px-2 py-1.5 text-xs text-indigo-300 bg-indigo-500/10 border border-indigo-500/20 rounded-lg">{a}</span>
            ))}
          </div>
        )}
      </div>

      {/* Prob bar */}
      <div className="px-4 py-3 border-b border-slate-800/40 space-y-2">
        <SentimentDistBar pos={r.probabilities.positive} neu={r.probabilities.neutral} neg={r.probabilities.negative} />
        <div className="flex justify-between text-xs text-slate-700">
          <span className="text-emerald-600">+{(r.probabilities.positive * 100).toFixed(1)}% pos</span>
          <span>~{(r.probabilities.neutral * 100).toFixed(1)}% neu</span>
          <span className="text-red-600">−{(r.probabilities.negative * 100).toFixed(1)}% neg</span>
        </div>
      </div>

      {/* Expand toggle */}
      <button onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-2.5 text-xs text-slate-600 hover:text-slate-300 hover:bg-slate-800/20 transition-all">
        <span className="flex items-center gap-1.5"><Eye className="w-3.5 h-3.5" />{expanded ? "Hide" : "Full"} intelligence</span>
        {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-4 border-t border-slate-800/40">
          <div className="pt-4">
            {r.sarcasm_probability !== undefined && <SarcasmMeter value={r.sarcasm_probability} />}
          </div>
          {r.strategic_insight && (
            <div className="bg-indigo-950/40 border border-indigo-500/20 rounded-lg p-3.5">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="w-3.5 h-3.5 text-indigo-400" />
                <span className="text-xs font-semibold text-indigo-400 uppercase tracking-wider">Strategic Insight</span>
              </div>
              <p className="text-sm text-slate-300 leading-relaxed">{r.strategic_insight}</p>
            </div>
          )}
          {r.risk?.action && (
            <div className={`border rounded-lg p-3.5 ${riskCfg.bg} ${riskCfg.border}`}>
              <div className="flex items-center gap-2 mb-1.5">
                <ShieldAlert className={`w-3.5 h-3.5 ${riskCfg.text}`} />
                <span className={`text-xs font-semibold uppercase tracking-wider ${riskCfg.text}`}>Recommended Action</span>
              </div>
              <p className="text-sm text-slate-300">{r.risk.action}</p>
            </div>
          )}
          {tokens.length > 0 && (
            <div>
              <p className="text-xs text-slate-700 mb-2 uppercase tracking-wider">Key Tokens</p>
              <div className="flex flex-wrap gap-1.5">
                {tokens.map((t: any, i: number) => (
                  <span key={i} className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-slate-800 border border-slate-700/50 text-xs font-mono text-slate-400">
                    {t.token}<span className="text-slate-600">{(t.importance * 100).toFixed(0)}%</span>
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function TabBtn({ active, onClick, icon: Icon, label, sub }: { active: boolean; onClick: () => void; icon: any; label: string; sub: string }) {
  return (
    <button onClick={onClick}
      className={`flex items-center gap-3 px-4 py-3.5 rounded-xl border text-left transition-all w-full ${active
        ? "bg-indigo-600/15 border-indigo-500/35 text-white"
        : "bg-slate-800/30 border-slate-700/40 text-slate-400 hover:border-slate-600 hover:text-slate-200"}`}>
      <Icon className={`w-4 h-4 shrink-0 ${active ? "text-indigo-400" : "text-slate-600"}`} />
      <div>
        <p className={`text-sm font-semibold leading-tight ${active ? "text-white" : ""}`}>{label}</p>
        <p className="text-xs text-slate-600 mt-0.5">{sub}</p>
      </div>
    </button>
  );
}

export default function DemoPage() {
  const [tab, setTab] = useState("stream");
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);
  const [streaming, setStreaming]   = useState(false);
  const [feed, setFeed]             = useState<any[]>([]);
  const [stats, setStats]           = useState({ pos: 0, neu: 0, neg: 0, crit: 0 });
  const [speed, setSpeed]           = useState(2200);
  const streamRef                   = useRef<NodeJS.Timeout | null>(null);
  const tweetIdx                    = useRef(0);
  const keyRef                      = useRef(0);
  const [text, setText]             = useState("");
  const [singleLoading, setSL]      = useState(false);
  const [singleResult, setSR]       = useState<any>(null);
  const [singleError, setSE]        = useState("");
  const [batchText, setBT]          = useState("");
  const [batchLoading, setBL]       = useState(false);
  const [batchResults, setBR]       = useState<any[]>([]);
  const [batchSummary, setBS]       = useState<any>(null);
  const [batchError, setBE]         = useState("");
  const [csvFile, setCsvFile]       = useState<File | null>(null);
  const [csvLoading, setCL]         = useState(false);
  const [csvError, setCE]           = useState("");
  const fileRef                     = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const check = async () => {
      try { await axios.get("/health", { timeout: 4000 }); setApiOnline(true); }
      catch { setApiOnline(false); }
    };
    check(); const id = setInterval(check, 15000); return () => clearInterval(id);
  }, []);

  const sendTweet = useCallback(async () => {
    const tweet = TWEETS[tweetIdx.current % TWEETS.length]; tweetIdx.current++;
    try {
      const { data } = await axios.post(`${API}/predict`, { text: tweet, source: "live-stream" }, { timeout: 15000 });
      setFeed(prev => [{ ...data, _key: keyRef.current++ }, ...prev].slice(0, 50));
      setStats(prev => ({
        pos:  prev.pos  + (data.sentiment === "positive" ? 1 : 0),
        neu:  prev.neu  + (data.sentiment === "neutral"  ? 1 : 0),
        neg:  prev.neg  + (data.sentiment === "negative" ? 1 : 0),
        crit: prev.crit + (data.urgency   === "critical" ? 1 : 0),
      }));
    } catch {}
  }, []);

  const startStream = useCallback(() => {
    setStreaming(true);
    const tick = async () => { await sendTweet(); streamRef.current = setTimeout(tick, speed); };
    tick();
  }, [sendTweet, speed]);

  const stopStream = useCallback(() => { setStreaming(false); if (streamRef.current) clearTimeout(streamRef.current); }, []);
  const resetStream = () => { stopStream(); setFeed([]); setStats({ pos: 0, neu: 0, neg: 0, crit: 0 }); tweetIdx.current = 0; };
  useEffect(() => () => { if (streamRef.current) clearTimeout(streamRef.current); }, []);

  const handleSingle = async () => {
    if (!text.trim() || singleLoading) return;
    setSL(true); setSE(""); setSR(null);
    try { const { data } = await axios.post(`${API}/predict`, { text: text.trim(), source: "demo" }); setSR(data); }
    catch (e: any) { setSE(e.response?.data?.detail || "API unavailable — is the platform running?"); }
    finally { setSL(false); }
  };

  const handleBatch = async () => {
    const lines = batchText.split("\n").map(l => l.trim()).filter(Boolean);
    if (!lines.length || batchLoading) return;
    setBL(true); setBE(""); setBR([]); setBS(null);
    try { const { data } = await axios.post(`${API}/predict/batch`, { texts: lines, source: "demo" }); setBR(data.predictions || []); setBS(data.summary || null); }
    catch (e: any) { setBE(e.response?.data?.detail || "Batch failed."); }
    finally { setBL(false); }
  };

  const handleCsv = async () => {
    if (!csvFile || csvLoading) return;
    setCL(true); setCE(""); setBR([]); setBS(null);
    const form = new FormData(); form.append("file", csvFile);
    try { const { data } = await axios.post(`${API}/predict/csv`, form, { headers: { "Content-Type": "multipart/form-data" }, timeout: 300000 }); setBR(data.predictions || []); setBS(data.summary || null); }
    catch (e: any) { setCE(e.response?.data?.detail || "Upload failed."); }
    finally { setCL(false); }
  };

  const total = (stats.pos + stats.neu + stats.neg) || 1;

  return (
    <div className="space-y-6 pb-12">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">Intelligence Demo</h1>
          <p className="text-slate-500 text-sm mt-1">Live NLP pipeline · Confidence scoring · Risk classification · Sarcasm detection</p>
        </div>
        <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs border ${apiOnline === null ? "border-slate-700 text-slate-500" : apiOnline ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400" : "border-red-500/30 bg-red-500/10 text-red-400"}`}>
          {apiOnline ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
          <span>{apiOnline === null ? "Checking…" : apiOnline ? "API Online" : "API Offline"}</span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2">
        <TabBtn active={tab === "stream"} onClick={() => setTab("stream")} icon={Radio}        label="Live Stream"     sub="Auto-stream tweets" />
        <TabBtn active={tab === "single"} onClick={() => setTab("single")} icon={MessageSquare} label="Single Analysis" sub="Any text, instant" />
        <TabBtn active={tab === "batch"}  onClick={() => setTab("batch")}  icon={Layers}        label="Batch / CSV"    sub="Up to 500 messages" />
      </div>

      {tab === "stream" && (
        <div className="space-y-4">
          <div className="flex items-center gap-3 flex-wrap">
            <button onClick={streaming ? stopStream : startStream} disabled={apiOnline === false}
              className={`flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all ${streaming ? "bg-red-600/80 hover:bg-red-600 text-white border border-red-500/40" : "bg-indigo-600 hover:bg-indigo-500 text-white border border-indigo-500/40 disabled:opacity-40"}`}>
              {streaming ? <><Square className="w-4 h-4" />Stop</> : <><Play className="w-4 h-4" />Start Stream</>}
            </button>
            <button onClick={resetStream} className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm text-slate-400 hover:text-white border border-slate-700 transition-all">
              <RotateCcw className="w-4 h-4" />Reset
            </button>
            <div className="flex items-center gap-2 ml-auto">
              <span className="text-xs text-slate-600">Speed</span>
              {[{ l: "Slow", v: 3500 }, { l: "Normal", v: 2200 }, { l: "Fast", v: 900 }].map(({ l, v }) => (
                <button key={v} onClick={() => setSpeed(v)} className={`px-3 py-1 rounded-lg text-xs border transition-all ${speed === v ? "bg-indigo-600/20 text-indigo-300 border-indigo-500/30" : "text-slate-500 hover:text-slate-300 border-slate-700/30"}`}>{l}</button>
              ))}
            </div>
          </div>

          {(stats.pos + stats.neu + stats.neg) > 0 && (
            <div className="grid grid-cols-4 gap-3">
              {[
                { label: "Positive", val: stats.pos, pct: stats.pos / total, Icon: TrendingUp, color: "text-emerald-400", bar: "bg-emerald-500" },
                { label: "Neutral",  val: stats.neu, pct: stats.neu / total, Icon: Minus, color: "text-slate-400", bar: "bg-slate-500" },
                { label: "Negative", val: stats.neg, pct: stats.neg / total, Icon: TrendingDown, color: "text-red-400", bar: "bg-red-500" },
                { label: "Critical", val: stats.crit,pct: stats.crit/total,  Icon: AlertTriangle, color: "text-orange-400", bar: "bg-orange-500" },
              ].map(({ label, val, pct, Icon: Ic, color, bar }) => (
                <div key={label} className="bg-slate-900 border border-slate-800 rounded-xl p-4">
                  <div className="flex items-center justify-between mb-2"><span className="text-xs text-slate-600">{label}</span><Ic className={`w-3.5 h-3.5 ${color}`} /></div>
                  <p className={`text-2xl font-bold tabular-nums ${color}`}>{val}</p>
                  <div className="mt-2 space-y-1"><GaugeBar value={pct} color={bar} /><p className="text-xs text-slate-600">{(pct * 100).toFixed(1)}%</p></div>
                </div>
              ))}
            </div>
          )}

          {feed.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 border border-dashed border-slate-800 rounded-xl text-slate-700">
              <Radio className="w-10 h-10 mb-3 opacity-30" />
              <p className="text-sm">Press <span className="text-indigo-400 font-semibold">Start Stream</span> to begin</p>
              <p className="text-xs mt-1 opacity-50">Real tweets · full intelligence pipeline</p>
            </div>
          ) : (
            <div className="space-y-3 max-h-[65vh] overflow-y-auto pr-1">
              {feed.map((r, i) => <IntelligenceCard key={r._key} r={r} index={i} isLatest={i === 0} />)}
            </div>
          )}
        </div>
      )}

      {tab === "single" && (
        <div className="space-y-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <textarea value={text} onChange={e => setText(e.target.value)} onKeyDown={e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) handleSingle(); }}
              placeholder="Enter any customer feedback, tweet, or review…" rows={4}
              className="w-full bg-slate-950/80 border border-slate-700/50 rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-700 focus:outline-none focus:border-indigo-500/40 resize-none" />
            <div className="flex items-center justify-between mt-3">
              <span className="text-xs text-slate-700">{text.length} chars · Ctrl+Enter</span>
              <button onClick={handleSingle} disabled={!text.trim() || singleLoading}
                className="flex items-center gap-2 px-5 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white text-sm font-semibold rounded-xl transition-all">
                {singleLoading ? <Activity className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
                {singleLoading ? "Analysing…" : "Analyse"}
              </button>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            {[
              "The flight was delayed 3 hours and nobody at the gate would help us.",
              "Oh great, another lost bag. You've really outdone yourselves this time.",
              "Best travel experience of my life. Crew were outstanding.",
              "App crashed during check-in. Nearly missed my flight.",
              "Decent flight. Nothing special, arrived on time.",
            ].map(ex => (
              <button key={ex} onClick={() => { setText(ex); setSR(null); }}
                className="px-3 py-1.5 text-xs text-slate-500 hover:text-slate-200 bg-slate-800/40 border border-slate-700/30 rounded-lg transition-all max-w-xs truncate">
                {ex.slice(0, 55)}…
              </button>
            ))}
          </div>
          {singleError && <div className="p-3 bg-red-950/40 border border-red-500/25 rounded-xl text-xs text-red-400 flex items-center gap-2"><AlertTriangle className="w-4 h-4 shrink-0" />{singleError}</div>}
          {singleResult && <IntelligenceCard r={singleResult} index={0} isLatest />}
        </div>
      )}

      {tab === "batch" && (
        <div className="space-y-4">
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4">
            <p className="text-xs text-slate-600 mb-2 uppercase tracking-wider">One message per line · up to 500</p>
            <textarea value={batchText} onChange={e => setBT(e.target.value)}
              placeholder={"Great flight, friendly crew!\nTotal disaster — lost my luggage.\nOh brilliant, another delay.\nApp crashed during checkout."}
              rows={7} className="w-full bg-slate-950/80 border border-slate-700/50 rounded-xl px-4 py-3 text-sm text-slate-200 placeholder-slate-700 focus:outline-none focus:border-violet-500/40 resize-none font-mono" />
            <div className="flex items-center justify-between mt-3">
              <span className="text-xs text-slate-700">{batchText.split("\n").filter(l => l.trim()).length} messages</span>
              <button onClick={handleBatch} disabled={!batchText.trim() || batchLoading}
                className="flex items-center gap-2 px-5 py-2 bg-violet-600 hover:bg-violet-500 disabled:opacity-40 text-white text-sm font-semibold rounded-xl transition-all">
                {batchLoading ? <Activity className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                {batchLoading ? "Analysing…" : "Analyse Batch"}
              </button>
            </div>
          </div>
          <div className="border-2 border-dashed border-slate-800 hover:border-emerald-500/30 rounded-xl p-8 text-center cursor-pointer transition-all"
            onClick={() => fileRef.current?.click()}
            onDragOver={e => e.preventDefault()}
            onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f?.name.endsWith(".csv")) setCsvFile(f); }}>
            <input ref={fileRef} type="file" accept=".csv" className="hidden" onChange={e => setCsvFile(e.target.files?.[0] || null)} />
            {csvFile ? (
              <div className="flex items-center justify-center gap-3">
                <FileText className="w-5 h-5 text-emerald-400" />
                <span className="text-sm text-emerald-300 font-medium">{csvFile.name}</span>
                <span className="text-xs text-slate-600">({(csvFile.size / 1024).toFixed(0)} KB)</span>
                <button onClick={e => { e.stopPropagation(); setCsvFile(null); }} className="text-slate-700 hover:text-slate-400"><X className="w-4 h-4" /></button>
              </div>
            ) : (
              <><Upload className="w-7 h-7 text-slate-700 mx-auto mb-2" /><p className="text-sm text-slate-500">Drop CSV or click to browse</p><p className="text-xs text-slate-700 mt-1">Auto-detects: text, tweet, message, review, feedback</p></>
            )}
          </div>
          {csvFile && (
            <div className="flex justify-end">
              <button onClick={handleCsv} disabled={csvLoading} className="flex items-center gap-2 px-5 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 text-white text-sm font-semibold rounded-xl transition-all">
                {csvLoading ? <Activity className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
                {csvLoading ? "Processing…" : "Upload & Analyse"}
              </button>
            </div>
          )}
          {(batchError || csvError) && <div className="p-3 bg-red-950/40 border border-red-500/25 rounded-xl text-xs text-red-400 flex items-center gap-2"><AlertTriangle className="w-4 h-4 shrink-0" />{batchError || csvError}</div>}
          {batchSummary && (
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-white">{batchSummary.total} messages analysed</h3>
                {batchSummary.critical_count > 0 && (
                  <span className="flex items-center gap-1.5 px-2.5 py-1 bg-red-950/60 text-red-300 border border-red-500/30 rounded-full text-xs font-semibold">
                    <AlertTriangle className="w-3 h-3" />{batchSummary.critical_count} critical
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
                {[
                  { l: "Positive", v: `${batchSummary.positive} (${(batchSummary.positive_rate * 100).toFixed(1)}%)`, c: "text-emerald-400" },
                  { l: "Negative", v: `${batchSummary.negative} (${(batchSummary.negative_rate * 100).toFixed(1)}%)`, c: "text-red-400" },
                  { l: "Avg Confidence", v: `${(batchSummary.avg_confidence * 100).toFixed(1)}%`, c: "text-indigo-400" },
                  { l: "Avg Impact", v: batchSummary.avg_business_impact?.toFixed(3) || "—", c: "text-violet-400" },
                ].map(({ l, v, c }) => (
                  <div key={l} className="bg-slate-950/60 border border-slate-800 rounded-lg p-3">
                    <p className="text-xs text-slate-600 mb-1">{l}</p>
                    <p className={`text-base font-bold ${c}`}>{v}</p>
                  </div>
                ))}
              </div>
              <SentimentDistBar pos={batchSummary.positive / batchSummary.total} neu={batchSummary.neutral / batchSummary.total} neg={batchSummary.negative / batchSummary.total} />
            </div>
          )}
          {batchResults.length > 0 && (
            <div className="space-y-3 max-h-[65vh] overflow-y-auto pr-1">
              {batchResults.map((r, i) => <IntelligenceCard key={i} r={r} index={i} />)}
            </div>
          )}
        </div>
      )}
    </div>
  );
}