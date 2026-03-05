"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { useState } from "react";
import { TrendingUp } from "lucide-react";

interface ImpactData {
  baseline_health_score: number;
  simulated_health_score: number;
  improvement: number;
}

export function BusinessImpactSimulator() {
  const [criticalPct, setCriticalPct] = useState(0.5);
  const [highPct, setHighPct] = useState(0.3);

  const { data, isFetching } = useQuery<ImpactData>({
    queryKey: ["impact", criticalPct, highPct],
    queryFn: () =>
      api
        .get(`/analytics/business-impact?resolve_critical_pct=${criticalPct}&resolve_high_pct=${highPct}`)
        .then((r) => r.data),
    placeholderData: (prev) => prev,
  });

  const baseline   = data?.baseline_health_score   ?? 62.5;
  const simulated  = data?.simulated_health_score   ?? baseline;
  const improvement = data?.improvement              ?? 0;

  return (
    <div className="card h-full">
      <div className="flex items-center gap-2 mb-1">
        <TrendingUp className="w-4 h-4 text-violet-400" />
        <h3 className="text-sm font-semibold text-slate-100">Scenario Simulator</h3>
        {isFetching && <span className="ml-auto w-2 h-2 bg-indigo-500 rounded-full animate-pulse" />}
      </div>
      <p className="text-xs text-slate-500 mb-5">
        Simulate the impact of resolving urgent issues on your health score
      </p>

      <div className="space-y-5">
        <div>
          <div className="flex justify-between mb-2">
            <label className="text-xs font-medium text-slate-400">Resolve Critical Issues</label>
            <span className="text-xs font-semibold text-indigo-300">{Math.round(criticalPct * 100)}%</span>
          </div>
          <input
            type="range" min={0} max={1} step={0.05}
            value={criticalPct}
            onChange={(e) => setCriticalPct(parseFloat(e.target.value))}
            className="w-full accent-indigo-500"
          />
        </div>

        <div>
          <div className="flex justify-between mb-2">
            <label className="text-xs font-medium text-slate-400">Resolve High Urgency Issues</label>
            <span className="text-xs font-semibold text-indigo-300">{Math.round(highPct * 100)}%</span>
          </div>
          <input
            type="range" min={0} max={1} step={0.05}
            value={highPct}
            onChange={(e) => setHighPct(parseFloat(e.target.value))}
            className="w-full accent-indigo-500"
          />
        </div>

        <div className="grid grid-cols-3 gap-3 pt-2">
          <div className="bg-slate-800 rounded-xl p-3 text-center border border-slate-700">
            <p className="text-xs text-slate-500 mb-1">Baseline</p>
            <p className="text-2xl font-bold text-slate-300">{baseline.toFixed(1)}</p>
          </div>
          <div className="bg-indigo-600/20 rounded-xl p-3 text-center border border-indigo-500/40">
            <p className="text-xs text-indigo-400 mb-1">Simulated</p>
            <p className="text-2xl font-bold text-indigo-200">{simulated.toFixed(1)}</p>
          </div>
          <div className={`rounded-xl p-3 text-center border ${
            improvement >= 0 ? "bg-emerald-500/10 border-emerald-500/30" : "bg-red-500/10 border-red-500/30"
          }`}>
            <p className="text-xs text-slate-400 mb-1">Δ Score</p>
            <p className={`text-2xl font-bold ${improvement >= 0 ? "text-emerald-400" : "text-red-400"}`}>
              {improvement >= 0 ? "+" : ""}{improvement.toFixed(1)}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
