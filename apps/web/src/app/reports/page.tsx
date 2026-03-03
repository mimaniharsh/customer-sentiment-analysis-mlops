"use client";

import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import {
  FileText, Download, Plus, Clock, Trash2, ChevronDown,
  ChevronUp, AlertTriangle, CheckCircle, BarChart2,
  Shield, Network, TrendingUp, Loader2,
} from "lucide-react";

const REPORT_TYPES = [
  {
    type: "executive_summary",
    title: "Executive Summary",
    desc: "Sentiment health, risk exposure, and top issue areas. Board-ready in one click.",
    icon: BarChart2,
    color: "text-indigo-400",
    bg: "bg-indigo-500/10",
    border: "border-indigo-500/20",
  },
  {
    type: "risk_breakdown",
    title: "Risk Breakdown",
    desc: "Tiered risk classification with escalation actions per urgency level.",
    icon: Shield,
    color: "text-red-400",
    bg: "bg-red-500/10",
    border: "border-red-500/20",
  },
  {
    type: "cluster_intelligence",
    title: "Cluster Intelligence",
    desc: "Root-cause analysis grouped by topic with negative rate and impact score.",
    icon: Network,
    color: "text-violet-400",
    bg: "bg-violet-500/10",
    border: "border-violet-500/20",
  },
  {
    type: "drift_analysis",
    title: "Drift & Stability",
    desc: "PSI-based distribution shift analysis between session halves.",
    icon: TrendingUp,
    color: "text-amber-400",
    bg: "bg-amber-500/10",
    border: "border-amber-500/20",
  },
];

const STATUS_STYLE = {
  good:    { text: "text-emerald-400", bg: "bg-emerald-500/10", border: "border-emerald-500/20" },
  neutral: { text: "text-slate-400",   bg: "bg-slate-700/30",   border: "border-slate-600/20" },
  risk:    { text: "text-red-400",     bg: "bg-red-500/10",     border: "border-red-500/20" },
  warn:    { text: "text-amber-400",   bg: "bg-amber-500/10",   border: "border-amber-500/20" },
};

function KPIChip({ label, value, unit = "", status = "neutral" }) {
  const s = STATUS_STYLE[status] || STATUS_STYLE.neutral;
  return (
    <div className={`flex items-center justify-between px-3 py-2 rounded-lg border ${s.bg} ${s.border}`}>
      <span className="text-xs text-slate-500">{label}</span>
      <span className={`text-sm font-bold tabular-nums ${s.text}`}>{value}{unit && <span className="text-xs font-normal ml-0.5">{unit}</span>}</span>
    </div>
  );
}

function ReportSection({ section }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="border border-slate-800 rounded-xl overflow-hidden">
      <button onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 bg-slate-900/60 hover:bg-slate-800/40 transition-colors">
        <span className="text-sm font-semibold text-white">{section.title}</span>
        {open ? <ChevronUp className="w-4 h-4 text-slate-500" /> : <ChevronDown className="w-4 h-4 text-slate-500" />}
      </button>
      {open && (
        <div className="px-4 pb-4 space-y-3 pt-3">
          {section.kpis && (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
              {section.kpis.map(kpi => (
                <KPIChip key={kpi.label} label={kpi.label} value={kpi.value} unit={kpi.unit} status={kpi.status} />
              ))}
            </div>
          )}
          {section.attributes && (
            <div className="space-y-2">
              {section.attributes.map(a => (
                <div key={a.attribute} className="flex items-center justify-between px-3 py-2 bg-slate-800/30 rounded-lg">
                  <span className="text-sm text-slate-300 capitalize">{a.attribute}</span>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-slate-500">{a.count} signals</span>
                    <span className="text-sm font-semibold text-indigo-400">{a.share}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
          {section.insight && (
            <div className="bg-indigo-950/30 border border-indigo-500/15 rounded-lg p-3">
              <p className="text-sm text-slate-300 leading-relaxed">{section.insight}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function RiskReport({ data }) {
  if (!data?.tiers) return <p className="text-sm text-slate-500">{data?.note || "No data."}</p>;
  const TIER_COLOR = { CRITICAL: "text-red-400", HIGH: "text-orange-400", MEDIUM: "text-amber-400", LOW: "text-emerald-400" };
  return (
    <div className="space-y-3">
      {data.insight && (
        <div className="bg-slate-800/40 border border-slate-700/30 rounded-lg p-3">
          <p className="text-sm text-slate-300">{data.insight}</p>
        </div>
      )}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-2">
        {[
          { l: "Avg Severity", v: data.avg_severity?.toFixed(3), c: "text-orange-400" },
          { l: "Avg Impact",   v: data.avg_business_impact?.toFixed(3), c: "text-violet-400" },
        ].map(({ l, v, c }) => (
          <div key={l} className="bg-slate-900/60 border border-slate-800 rounded-lg p-3">
            <p className="text-xs text-slate-600 mb-1">{l}</p>
            <p className={`text-base font-bold ${c}`}>{v ?? "—"}</p>
          </div>
        ))}
      </div>
      {data.tiers.map(t => (
        <div key={t.tier} className="border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-3">
              <span className={`text-sm font-bold tracking-wider ${TIER_COLOR[t.tier] || "text-slate-400"}`}>{t.tier}</span>
              <span className="text-lg font-bold text-white tabular-nums">{t.count}</span>
              <span className="text-xs text-slate-500">({t.share})</span>
            </div>
          </div>
          <p className="text-xs text-slate-400">{t.action}</p>
        </div>
      ))}
    </div>
  );
}

function ClusterReport({ data }) {
  if (!data?.clusters?.length) return <p className="text-sm text-slate-500">{data?.note || "No data."}</p>;
  const SENT_COLOR = { negative: "text-red-400", positive: "text-emerald-400", neutral: "text-slate-400" };
  return (
    <div className="space-y-3">
      {data.insight && <div className="bg-slate-800/40 border border-slate-700/30 rounded-lg p-3"><p className="text-sm text-slate-300">{data.insight}</p></div>}
      {data.clusters.map(c => (
        <div key={c.theme} className="border border-slate-800 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-semibold text-white">{c.theme}</span>
            <span className={`text-xs font-semibold ${SENT_COLOR[c.dominant_sentiment] || "text-slate-400"}`}>{c.dominant_sentiment}</span>
          </div>
          <div className="grid grid-cols-4 gap-2 text-center text-xs">
            {[
              { l: "Signals", v: c.count },
              { l: "Share", v: c.share },
              { l: "Neg Rate", v: c.negative_rate },
              { l: "Avg Impact", v: c.avg_impact?.toFixed(2) },
            ].map(({ l, v }) => (
              <div key={l} className="bg-slate-900/60 rounded-lg py-1.5">
                <p className="text-slate-600 mb-0.5">{l}</p>
                <p className="text-slate-300 font-semibold">{v}</p>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function DriftReport({ data }) {
  if (!data?.total_analysed) return <p className="text-sm text-slate-500">{data?.note || "No data."}</p>;
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        {[
          { l: "PSI Score", v: data.psi_score?.toFixed(4), status: data.psi_score > 0.2 ? "risk" : "good" },
          { l: "Drift Detected", v: data.drift_detected ? "Yes" : "No", status: data.drift_detected ? "risk" : "good" },
          { l: "Total Analysed", v: data.total_analysed, status: "neutral" },
        ].map(({ l, v, status }) => (
          <KPIChip key={l} label={l} value={v} status={status} />
        ))}
      </div>
      {data.insight && <div className="bg-slate-800/40 border border-slate-700/30 rounded-lg p-3"><p className="text-sm text-slate-300">{data.insight}</p></div>}
      <div className="grid grid-cols-2 gap-3">
        {[
          { label: "First Half",  d: data.first_half },
          { label: "Second Half", d: data.second_half },
        ].map(({ label, d }) => d && (
          <div key={label} className="border border-slate-800 rounded-xl p-4">
            <p className="text-xs text-slate-500 mb-3 uppercase tracking-wider">{label} ({d.n} msgs)</p>
            <div className="space-y-2">
              <KPIChip label="Negative Rate" value={`${(d.negative_rate * 100).toFixed(1)}%`} status="neutral" />
              <KPIChip label="Avg Confidence" value={`${(d.avg_confidence * 100).toFixed(1)}%`} status="neutral" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ReportViewer({ report, onDelete }) {
  const [open, setOpen] = useState(false);
  const d = report.data;
  const hasNote = !!d?.note;

  return (
    <div className="border border-slate-800 rounded-xl overflow-hidden">
      <div className="flex items-center gap-4 px-4 py-3.5 bg-slate-900/50">
        <FileText className="w-4 h-4 text-slate-500 shrink-0" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-slate-100 truncate">{report.title}</p>
          <p className="text-xs text-slate-500 flex items-center gap-2 mt-0.5">
            <Clock className="w-3 h-3" />
            {report.created_at_display}
            {report.record_count > 0 && <span>· {report.record_count} predictions</span>}
          </p>
        </div>
        <span className="flex items-center gap-1 px-2 py-0.5 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 rounded-full text-xs">
          <CheckCircle className="w-3 h-3" />ready
        </span>
        <button onClick={() => setOpen(!open)} className="p-1.5 text-slate-500 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition-colors">
          {open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
        <button onClick={() => onDelete(report.id)} className="p-1.5 text-slate-600 hover:text-red-400 hover:bg-red-950/30 rounded-lg transition-colors">
          <Trash2 className="w-4 h-4" />
        </button>
      </div>

      {open && (
        <div className="px-4 pb-5 pt-4 border-t border-slate-800 space-y-4">
          {hasNote ? (
            <div className="bg-amber-950/30 border border-amber-500/20 rounded-lg p-3 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
              <p className="text-sm text-amber-300">{d.note}</p>
            </div>
          ) : (
            <>
              {report.type === "executive_summary"  && d.sections?.map((s, i) => <ReportSection key={i} section={s} />)}
              {report.type === "risk_breakdown"      && <RiskReport data={d} />}
              {report.type === "cluster_intelligence" && <ClusterReport data={d} />}
              {report.type === "drift_analysis"      && <DriftReport data={d} />}
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default function ReportsPage() {
  const qc = useQueryClient();

  const { data, isLoading } = useQuery({
    queryKey: ["reports"],
    queryFn: () => api.get("/reports").then(r => r.data),
    refetchInterval: 30_000,
  });

  const generate = useMutation({
    mutationFn: (type) => api.post("/reports", { type }).then(r => r.data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["reports"] }),
  });

  const remove = useMutation({
    mutationFn: (id) => api.delete(`/reports/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["reports"] }),
  });

  const reports = data?.reports ?? [];

  return (
    <div className="space-y-6 pb-12">
      <div>
        <h1 className="text-2xl font-bold text-white tracking-tight">Intelligence Reports</h1>
        <p className="text-slate-500 text-sm mt-1">Generate structured reports from live prediction data — board-ready, instantly</p>
      </div>

      {/* Generate cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        {REPORT_TYPES.map(rt => {
          const Icon = rt.icon;
          const isGenerating = generate.isPending && generate.variables === rt.type;
          return (
            <div key={rt.type} className={`border rounded-xl p-4 flex flex-col gap-3 ${rt.bg} ${rt.border} hover:border-opacity-60 transition-all`}>
              <div className={`w-8 h-8 rounded-lg bg-slate-900/60 flex items-center justify-center`}>
                <Icon className={`w-4 h-4 ${rt.color}`} />
              </div>
              <div className="flex-1">
                <h3 className="text-sm font-semibold text-white">{rt.title}</h3>
                <p className="text-xs text-slate-500 mt-1 leading-relaxed">{rt.desc}</p>
              </div>
              <button
                onClick={() => generate.mutate(rt.type)}
                disabled={isGenerating}
                className="flex items-center justify-center gap-1.5 w-full py-2 text-xs font-semibold text-white bg-slate-800 hover:bg-slate-700 disabled:opacity-50 rounded-lg transition-all border border-slate-700/50"
              >
                {isGenerating ? <><Loader2 className="w-3 h-3 animate-spin" />Generating…</> : <><Plus className="w-3 h-3" />Generate</>}
              </button>
            </div>
          );
        })}
      </div>

      {generate.isError && (
        <div className="p-3 bg-red-950/40 border border-red-500/25 rounded-xl text-xs text-red-400 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          {generate.error?.response?.data?.detail ?? "Failed to generate report — is the API running?"}
        </div>
      )}

      {/* Reports list */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-300">Generated Reports</h2>
          {reports.length > 0 && <span className="text-xs text-slate-600">{reports.length} report{reports.length !== 1 ? "s" : ""}</span>}
        </div>

        {isLoading ? (
          <div className="flex items-center justify-center py-16 border border-slate-800 rounded-xl">
            <Loader2 className="w-5 h-5 text-slate-600 animate-spin" />
          </div>
        ) : reports.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-16 border border-dashed border-slate-800 rounded-xl text-slate-700">
            <FileText className="w-10 h-10 mb-3 opacity-30" />
            <p className="text-sm">No reports generated yet</p>
            <p className="text-xs mt-1 opacity-60">Click Generate above to create your first report</p>
            <p className="text-xs mt-1 opacity-40">Tip: run the Live Stream demo first to populate data</p>
          </div>
        ) : (
          reports.map(r => (
            <ReportViewer key={r.id} report={r} onDelete={(id) => remove.mutate(id)} />
          ))
        )}
      </div>
    </div>
  );
}
