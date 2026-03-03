"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, Cell,
} from "recharts";

// ─── SentimentTrendChart ─────────────────────────────────────────────────────
export function SentimentTrendChart() {
  const { data, isLoading } = useQuery({
    queryKey: ["timeseries", 30],
    queryFn: () => api.get("/analytics/timeseries?days=30").then((r) => r.data),
    refetchInterval: 120_000,
  });

  const series = data?.series ?? [];

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-sm font-semibold text-slate-100">Sentiment Trend</h3>
          <p className="text-xs text-slate-500 mt-0.5">30-day rolling · daily granularity</p>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-emerald-500 inline-block rounded" /> Positive</span>
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-amber-500 inline-block rounded" /> Neutral</span>
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-red-500 inline-block rounded" /> Negative</span>
        </div>
      </div>
      {isLoading ? (
        <div className="h-56 bg-slate-800 rounded-lg animate-pulse" />
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={series}>
            <defs>
              <linearGradient id="gPos" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="gNeg" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: "#64748b" }}
              tickLine={false}
              tickFormatter={(v) => v.slice(5)}
            />
            <YAxis tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={false} />
            <Tooltip
              contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }}
              labelStyle={{ color: "#94a3b8" }}
            />
            <Area type="monotone" dataKey="positive" stroke="#10b981" strokeWidth={2} fill="url(#gPos)" name="Positive" />
            <Area type="monotone" dataKey="neutral" stroke="#f59e0b" strokeWidth={1.5} fill="none" name="Neutral" strokeDasharray="4 2" />
            <Area type="monotone" dataKey="negative" stroke="#ef4444" strokeWidth={2} fill="url(#gNeg)" name="Negative" />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

// ─── UrgencyBreakdown ────────────────────────────────────────────────────────
export function UrgencyBreakdown() {
  const { data } = useQuery({
    queryKey: ["urgency"],
    queryFn: () => api.get("/analytics/urgency?days=7").then((r) => r.data),
  });

  // Aggregate last 7 days
  const series = data?.series ?? [];
  const totals = series.reduce(
    (acc: any, d: any) => ({
      critical: acc.critical + d.critical,
      high: acc.high + d.high,
      medium: acc.medium + d.medium,
      low: acc.low + d.low,
    }),
    { critical: 0, high: 0, medium: 0, low: 0 }
  );

  const chartData = [
    { name: "Critical", value: totals.critical, color: "#dc2626" },
    { name: "High", value: totals.high, color: "#ea580c" },
    { name: "Medium", value: totals.medium, color: "#f59e0b" },
    { name: "Low", value: totals.low, color: "#22c55e" },
  ];

  return (
    <div className="card h-full">
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-slate-100">Urgency Breakdown</h3>
        <p className="text-xs text-slate-500 mt-0.5">Last 7 days</p>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData} layout="vertical">
          <XAxis type="number" tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={false} />
          <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: "#94a3b8" }} tickLine={false} axisLine={false} width={55} />
          <Tooltip
            contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8, fontSize: 12 }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {chartData.map((entry, index) => (
              <Cell key={index} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
