"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { AlertTriangle, CheckCircle, Activity } from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
  BarChart, Bar, Cell,
} from "recharts";
import { useState } from "react";

// ─── DriftIndicator ──────────────────────────────────────────────────────────
export function DriftIndicator() {
  const { data } = useQuery({
    queryKey: ["drift"],
    queryFn: () => api.get("/analytics/drift").then((r) => r.data),
  });

  const history = data?.drift_history ?? [];
  const currentPSI = data?.current_psi ?? 0;
  const driftDetected = currentPSI > 0.2;

  return (
    <div className="card h-full">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-100">Model Drift Monitor</h3>
          <p className="text-xs text-slate-500 mt-0.5">PSI · KS Statistic · 12-week window</p>
        </div>
        <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${
          driftDetected
            ? "bg-red-500/15 text-red-400 border border-red-500/30"
            : "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
        }`}>
          {driftDetected ? <AlertTriangle className="w-3 h-3" /> : <CheckCircle className="w-3 h-3" />}
          {driftDetected ? "Drift Detected" : "Stable"}
        </div>
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={history}>
          <XAxis dataKey="week_start" tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} />
          <YAxis tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={false} domain={[0, 0.4]} />
          <Tooltip
            contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }}
            labelStyle={{ color: "#94a3b8" }}
          />
          <ReferenceLine y={0.2} stroke="#ef4444" strokeDasharray="3 3" label={{ value: "Threshold", fill: "#ef4444", fontSize: 10 }} />
          <Line type="monotone" dataKey="psi_score" stroke="#6366f1" strokeWidth={2} dot={false} name="PSI" />
          <Line type="monotone" dataKey="ks_statistic" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="KS" strokeDasharray="4 2" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─── ClusterInsights ─────────────────────────────────────────────────────────
export function ClusterInsights() {
  const { data } = useQuery({
    queryKey: ["clusters"],
    queryFn: () => api.get("/analytics/clusters").then((r) => r.data),
  });

  const clusters = (data?.clusters ?? []).slice(0, 5);
  const COLOR: Record<string, string> = {
    positive: "#10b981",
    neutral: "#f59e0b",
    negative: "#ef4444",
  };

  return (
    <div className="card h-full">
      <div className="flex items-center gap-2 mb-4">
        <Activity className="w-4 h-4 text-indigo-400" />
        <h3 className="text-sm font-semibold text-slate-100">Root-Cause Clusters</h3>
      </div>
      <div className="space-y-2.5">
        {clusters.map((c: any) => (
          <div key={c.cluster_id} className="flex items-center gap-3 p-2.5 rounded-lg bg-slate-800/50 hover:bg-slate-800 transition-colors">
            <div
              className="w-2 h-8 rounded-full shrink-0"
              style={{ background: COLOR[c.dominant_sentiment] ?? "#6366f1" }}
            />
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-slate-100 truncate">{c.theme}</p>
              <p className="text-xs text-slate-500">{c.message_count} msgs · impact {(c.avg_business_impact * 100).toFixed(0)}</p>
            </div>
            <div className="text-right shrink-0">
              <p className="text-xs text-slate-400">{(c.avg_severity * 100).toFixed(0)} sev</p>
              <div className="flex gap-1 mt-1">
                {(c.top_terms ?? []).slice(0, 2).map((t: string) => (
                  <span key={t} className="text-xs bg-slate-700 text-slate-400 px-1.5 py-0.5 rounded">{t}</span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── BusinessImpactSimulator ─────────────────────────────────────────────────
export function BusinessImpactSimulator() {
  const [criticalPct, setCriticalPct] = useState(0.5);
  const [highPct, setHighPct] = useState(0.3);

  const { data } = useQuery({
    queryKey: ["impact", criticalPct, highPct],
    queryFn: () =>
      api.get(`/analytics/business-impact?resolve_critical_pct=${criticalPct}&resolve_high_pct=${highPct}`)
         .then((r) => r.data),
  });

  const baseline = data?.baseline_health_score ?? 62.5;
  const simulated = data?.simulated_health_score ?? baseline;
  const improvement = data?.improvement ?? 0;

  return (
    <div className="card h-full">
      <h3 className="text-sm font-semibold text-slate-100 mb-1">Scenario Simulator</h3>
      <p className="text-xs text-slate-500 mb-4">Simulate impact of resolving urgent issues</p>

      <div className="space-y-4">
        <div>
          <div className="flex justify-between mb-1">
            <label className="text-xs text-slate-400">Resolve Critical Urgency</label>
            <span className="text-xs text-indigo-300 font-medium">{Math.round(criticalPct * 100)}%</span>
          </div>
          <input
            type="range" min={0} max={1} step={0.05}
            value={criticalPct}
            onChange={(e) => setCriticalPct(parseFloat(e.target.value))}
            className="w-full accent-indigo-500"
          />
        </div>
        <div>
          <div className="flex justify-between mb-1">
            <label className="text-xs text-slate-400">Resolve High Urgency</label>
            <span className="text-xs text-indigo-300 font-medium">{Math.round(highPct * 100)}%</span>
          </div>
          <input
            type="range" min={0} max={1} step={0.05}
            value={highPct}
            onChange={(e) => setHighPct(parseFloat(e.target.value))}
            className="w-full accent-indigo-500"
          />
        </div>

        <div className="grid grid-cols-3 gap-3 pt-2">
          <div className="bg-slate-800 rounded-lg p-3 text-center">
            <p className="text-xs text-slate-500">Baseline</p>
            <p className="text-xl font-bold text-slate-300">{baseline.toFixed(1)}</p>
          </div>
          <div className="bg-indigo-600/20 border border-indigo-500/30 rounded-lg p-3 text-center">
            <p className="text-xs text-indigo-400">Simulated</p>
            <p className="text-xl font-bold text-indigo-300">{simulated.toFixed(1)}</p>
          </div>
          <div className={`rounded-lg p-3 text-center ${improvement >= 0 ? "bg-emerald-500/10 border border-emerald-500/20" : "bg-red-500/10 border border-red-500/20"}`}>
            <p className="text-xs text-slate-400">Δ Score</p>
            <p className={`text-xl font-bold ${improvement >= 0 ? "text-emerald-400" : "text-red-400"}`}>
              {improvement >= 0 ? "+" : ""}{improvement.toFixed(1)}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
