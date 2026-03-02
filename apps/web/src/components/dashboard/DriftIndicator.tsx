"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { AlertTriangle, CheckCircle } from "lucide-react";
import {
  LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";

interface DriftRecord {
  week_start: string;
  psi_score: number;
  ks_statistic: number;
  drift_detected: boolean;
}

interface DriftResponse {
  drift_history: DriftRecord[];
  current_psi: number;
}

export function DriftIndicator() {
  const { data, isLoading } = useQuery<DriftResponse>({
    queryKey: ["drift"],
    queryFn: () => api.get("/analytics/drift").then((r) => r.data),
    staleTime: 300_000,
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
        <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
          driftDetected
            ? "bg-red-500/15 text-red-400 border-red-500/30"
            : "bg-emerald-500/15 text-emerald-400 border-emerald-500/30"
        }`}>
          {driftDetected ? <AlertTriangle className="w-3 h-3" /> : <CheckCircle className="w-3 h-3" />}
          {driftDetected ? "Drift Detected" : "Stable"}
        </div>
      </div>
      {isLoading ? (
        <div className="h-44 bg-slate-800 rounded-lg animate-pulse" />
      ) : (
        <ResponsiveContainer width="100%" height={180}>
          <LineChart data={history}>
            <XAxis
              dataKey="week_start"
              tick={{ fontSize: 10, fill: "#64748b" }}
              tickLine={false}
              tickFormatter={(v: string) => v.slice(5)}
            />
            <YAxis
              tick={{ fontSize: 10, fill: "#64748b" }}
              tickLine={false}
              axisLine={false}
              domain={[0, 0.4]}
            />
            <Tooltip
              contentStyle={{
                background: "#1e293b",
                border: "1px solid #334155",
                borderRadius: 8,
                fontSize: 12,
              }}
              labelStyle={{ color: "#94a3b8" }}
            />
            <ReferenceLine
              y={0.2}
              stroke="#ef4444"
              strokeDasharray="3 3"
              label={{ value: "Threshold 0.2", fill: "#ef4444", fontSize: 10 }}
            />
            <Line type="monotone" dataKey="psi_score" stroke="#6366f1" strokeWidth={2} dot={false} name="PSI" />
            <Line type="monotone" dataKey="ks_statistic" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="KS" strokeDasharray="4 2" />
          </LineChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
