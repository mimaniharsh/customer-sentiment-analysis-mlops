"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis,
  CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { api } from "@/lib/api";
import { AlertTriangle, TrendingUp, Activity } from "lucide-react";

// ── Types ──────────────────────────────────────────────────────────────────

interface IssueBreakdown {
  issue: string;
  count: number;
}

interface Bubble {
  attribute: string;
  message_count: number;
  avg_severity: number;
  avg_impact: number;
  negative_rate: number;
  risk_tier: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
  top_issues: string[];
  issue_breakdown: IssueBreakdown[];
  x?: number;
  y?: number;
  z?: number;
}

interface BubbleResponse {
  bubbles: Bubble[];
  total_messages: number;
  highest_risk_attribute: string;
  highest_volume_attribute: string;
}

// ── Colours ────────────────────────────────────────────────────────────────

const RISK_COLOUR: Record<string, string> = {
  CRITICAL: "#ef4444",
  HIGH:     "#f97316",
  MEDIUM:   "#f59e0b",
  LOW:      "#10b981",
};

const RISK_BADGE: Record<string, string> = {
  CRITICAL: "bg-red-950/60 border-red-500/30 text-red-300",
  HIGH:     "bg-orange-950/60 border-orange-500/30 text-orange-300",
  MEDIUM:   "bg-amber-950/60 border-amber-500/30 text-amber-300",
  LOW:      "bg-emerald-950/50 border-emerald-500/25 text-emerald-400",
};

// ── Tooltip ────────────────────────────────────────────────────────────────

function BubbleTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: Array<{ payload: Bubble }>;
}) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload;
  const colour = RISK_COLOUR[d.risk_tier] ?? "#6366f1";
  return (
    <div className="bg-slate-900 border border-slate-700 rounded-xl p-4 shadow-2xl max-w-xs">
      <div className="flex items-center justify-between mb-3">
        <p className="text-sm font-bold text-white capitalize">{d.attribute}</p>
        <span
          className="text-xs font-bold px-2 py-0.5 rounded-full border"
          style={{ color: colour, borderColor: colour + "50", background: colour + "18" }}
        >
          {d.risk_tier}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-2 mb-3">
        {[
          { label: "Volume",   value: String(d.message_count) },
          { label: "Severity", value: `${(d.avg_severity * 100).toFixed(0)}%` },
          { label: "Neg rate", value: `${(d.negative_rate * 100).toFixed(0)}%` },
        ].map(({ label, value }) => (
          <div key={label} className="text-center bg-slate-800/60 rounded-lg py-2">
            <p className="text-xs text-slate-500 mb-1">{label}</p>
            <p className="text-sm font-bold text-white">{value}</p>
          </div>
        ))}
      </div>
      {d.issue_breakdown.length > 0 && (
        <div>
          <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Root causes</p>
          <div className="space-y-1.5">
            {d.issue_breakdown.slice(0, 4).map((ib, i) => {
              const max = d.issue_breakdown[0]?.count || 1;
              return (
                <div key={i}>
                  <div className="flex justify-between mb-0.5">
                    <span className="text-xs text-slate-300 truncate">{ib.issue}</span>
                    <span className="text-xs text-slate-500 ml-2 shrink-0">{ib.count}</span>
                  </div>
                  <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${(ib.count / max) * 100}%`, background: colour }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Custom bubble shape ────────────────────────────────────────────────────
// Must use `any` — recharts passes its own internal props object at runtime
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function renderBubble(props: any) {
  const cx: number = props.cx ?? 0;
  const cy: number = props.cy ?? 0;
  const payload: Bubble | undefined = props.payload;
  if (!payload) return <g />;
  const colour = RISK_COLOUR[payload.risk_tier] ?? "#6366f1";
  const r = Math.sqrt(payload.message_count / Math.PI) * 2.8;
  return (
    <g>
      <circle cx={cx} cy={cy} r={r + 4} fill={colour} opacity={0.10} />
      <circle cx={cx} cy={cy} r={r} fill={colour} opacity={0.82} stroke={colour} strokeWidth={1.5} />
    </g>
  );
}

// ── Drill-down ─────────────────────────────────────────────────────────────

function DrillDown({ bubble }: { bubble: Bubble }) {
  const colour = RISK_COLOUR[bubble.risk_tier];
  const total = bubble.issue_breakdown.reduce((a, b) => a + b.count, 0) || 1;
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <div>
            <p className="text-sm font-bold text-white capitalize">{bubble.attribute}</p>
            <p className="text-xs text-slate-500 mt-0.5">
              {bubble.message_count} mentions · {(bubble.negative_rate * 100).toFixed(0)}% negative
            </p>
          </div>
          <span className={`text-xs font-bold px-2 py-1 rounded-lg border ${RISK_BADGE[bubble.risk_tier]}`}>
            {bubble.risk_tier}
          </span>
        </div>
        <div className="flex justify-between text-xs mb-1">
          <span className="text-slate-500">Avg severity</span>
          <span className="text-slate-300 font-semibold">{(bubble.avg_severity * 100).toFixed(0)}%</span>
        </div>
        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
          <div className="h-full rounded-full" style={{ width: `${bubble.avg_severity * 100}%`, background: colour }} />
        </div>
      </div>

      <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4">
        <p className="text-xs text-slate-500 uppercase tracking-wider mb-3">Issue breakdown</p>
        <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
          {bubble.issue_breakdown.map((ib, i) => (
            <div key={i}>
              <div className="flex justify-between mb-0.5">
                <span className="text-xs text-slate-200 truncate">{ib.issue}</span>
                <span className="text-xs text-slate-500 tabular-nums ml-2 shrink-0">
                  {ib.count} ({((ib.count / total) * 100).toFixed(0)}%)
                </span>
              </div>
              <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full"
                  style={{
                    width: `${(ib.count / (bubble.issue_breakdown[0]?.count || 1)) * 100}%`,
                    background: colour + "cc",
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── Main ───────────────────────────────────────────────────────────────────

export function BubbleChart() {
  const [selected, setSelected] = useState<Bubble | null>(null);

  const { data, isLoading } = useQuery<BubbleResponse>({
    queryKey: ["bubble"],
    queryFn: () => api.get("/analytics/bubble").then((r) => r.data),
    staleTime: 60_000,
    refetchInterval: 120_000,
  });

  const bubbles: Bubble[] = (data?.bubbles ?? []).map((b, i) => ({
    ...b,
    x: i,
    y: b.avg_severity,
    z: b.message_count,
  }));

  const xTick = (v: number) => {
    const b = bubbles[v];
    if (!b) return "";
    return b.attribute.charAt(0).toUpperCase() + b.attribute.slice(1);
  };

  if (isLoading) {
    return (
      <div className="card flex items-center justify-center h-64">
        <div className="flex items-center gap-2 text-slate-600">
          <Activity className="w-4 h-4 animate-spin" />
          <span className="text-sm">Loading chart…</span>
        </div>
      </div>
    );
  }

  return (
    <div className="card space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-sm font-semibold text-slate-100">Resource Prioritisation Map</h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Bubble size = volume · Y-axis = severity · Colour = risk tier · Click to drill down
          </p>
        </div>
        <div className="flex items-center gap-2">
          {data?.highest_risk_attribute && (
            <div className="flex items-center gap-1.5 px-2 py-1 bg-red-950/40 border border-red-500/25 rounded-lg">
              <AlertTriangle className="w-3 h-3 text-red-400" />
              <span className="text-xs text-red-300 capitalize">{data.highest_risk_attribute}</span>
            </div>
          )}
          {data?.highest_volume_attribute && (
            <div className="flex items-center gap-1.5 px-2 py-1 bg-indigo-950/40 border border-indigo-500/25 rounded-lg">
              <TrendingUp className="w-3 h-3 text-indigo-400" />
              <span className="text-xs text-indigo-300 capitalize">{data.highest_volume_attribute}</span>
            </div>
          )}
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-5 flex-wrap">
        {Object.entries(RISK_COLOUR).map(([tier, colour]) => (
          <div key={tier} className="flex items-center gap-1.5">
            <div className="w-3 h-3 rounded-full" style={{ background: colour }} />
            <span className="text-xs text-slate-500">{tier}</span>
          </div>
        ))}
      </div>

      {/* Chart */}
      <div style={{ width: "100%", height: 320 }} className="cursor-pointer">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 16, right: 24, bottom: 24, left: 8 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis
              type="number"
              dataKey="x"
              domain={bubbles.length > 0 ? [-0.5, bubbles.length - 0.5] : [0, 1]}
              tickCount={Math.max(bubbles.length, 1)}
              tickFormatter={xTick}
              tick={{ fill: "#64748b", fontSize: 11 }}
              axisLine={{ stroke: "#1e293b" }}
              tickLine={false}
            />
            <YAxis
              type="number"
              dataKey="y"
              domain={[0, 1]}
              tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
              tick={{ fill: "#64748b", fontSize: 11 }}
              axisLine={{ stroke: "#1e293b" }}
              tickLine={false}
              label={{
                value: "Severity",
                angle: -90,
                position: "insideLeft",
                offset: 12,
                style: { fill: "#475569", fontSize: 10 },
              }}
            />
            <ZAxis type="number" dataKey="z" range={[300, 2800]} />
            <Tooltip
              content={({ active, payload }) => (
                <BubbleTooltip
                  active={active}
                  payload={payload as Array<{ payload: Bubble }>}
                />
              )}
              cursor={false}
            />
            <Scatter
              data={bubbles}
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              shape={(props: any) => renderBubble(props)}
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              onClick={(d: any) =>
                setSelected(selected?.attribute === d.attribute ? null : (d as Bubble))
              }
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Drill-down panel */}
      {selected && <DrillDown bubble={selected} />}

      {/* Summary strip */}
      {bubbles.length > 0 && (
        <div className="grid grid-cols-4 gap-2 pt-1 border-t border-slate-800">
          {bubbles.slice(0, 4).map((b) => (
            <button
              key={b.attribute}
              onClick={() => setSelected(selected?.attribute === b.attribute ? null : b)}
              className={`text-left p-2.5 rounded-lg border transition-all ${
                selected?.attribute === b.attribute
                  ? "bg-slate-800 border-slate-600"
                  : "bg-slate-900/40 border-slate-800/60 hover:border-slate-700"
              }`}
            >
              <p className="text-xs font-medium text-slate-200 capitalize truncate">{b.attribute}</p>
              <p className="text-xs text-slate-500 mt-0.5">{b.message_count} msgs</p>
              <div
                className="mt-1.5 h-1 rounded-full"
                style={{ width: `${b.avg_severity * 100}%`, background: RISK_COLOUR[b.risk_tier] }}
              />
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
