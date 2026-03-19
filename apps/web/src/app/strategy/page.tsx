"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import {
  TrendingUp, TrendingDown, Minus,
  AlertTriangle, CheckCircle, Shield,
  DollarSign, Users, Target, Zap,
  Clock, ArrowRight, Award, Flag,
  Activity, Briefcase,
} from "lucide-react";

// ─── Types ───────────────────────────────────────────────────────────────────

interface ScorecardRow {
  attribute: string;
  score: number;
  status: "critical" | "warn" | "ok";
  trend: "improving" | "worsening" | "stable";
  trend_delta: number;
  negative_rate: number;
  churn_risk: number;
  revenue_at_risk: number;
  recommendation: string;
  colour: string;
}

interface MatrixPoint {
  issue: string;
  attribute: string;
  impact: number;
  effort: number;
  quadrant: "quick_win" | "strategic" | "fill_in" | "avoid";
  colour: string;
}

interface RevenueModel {
  total_customers: number;
  dissatisfied_count: number;
  churn_risk_count: number;
  revenue_at_risk: number;
  avg_clv: number;
  dissatisfied_pct: number;
  churn_risk_pct: number;
}

interface ActionItem {
  rank: number;
  action: string;
  attribute: string;
  owner: string;
  timeline: string;
  impact: string;
  priority: "CRITICAL" | "HIGH" | "MEDIUM";
}

interface StrategyData {
  brand_scorecard: ScorecardRow[];
  priority_matrix: MatrixPoint[];
  revenue_model: RevenueModel;
  action_plan: ActionItem[];
  overall_health: number;
  churn_risk_index: number;
  total_revenue_at_risk: number;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const STATUS_LABEL: Record<string, string> = { critical: "CRITICAL", warn: "AT RISK", ok: "HEALTHY" };
const STATUS_COLOUR: Record<string, string> = {
  critical: "text-red-400",
  warn:     "text-amber-400",
  ok:       "text-emerald-400",
};
const STATUS_BG: Record<string, string> = {
  critical: "bg-red-500/15 border-red-500/30",
  warn:     "bg-amber-500/15 border-amber-500/30",
  ok:       "bg-emerald-500/15 border-emerald-500/30",
};
const PRIORITY_COLOUR: Record<string, string> = {
  CRITICAL: "text-red-400",
  HIGH:     "text-orange-400",
  MEDIUM:   "text-amber-400",
};
const PRIORITY_BG: Record<string, string> = {
  CRITICAL: "bg-red-950/50 border-red-500/25",
  HIGH:     "bg-orange-950/50 border-orange-500/25",
  MEDIUM:   "bg-amber-950/50 border-amber-500/25",
};

function fmt$M(v: number) {
  if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`;
  if (v >= 1_000)     return `$${(v / 1_000).toFixed(0)}K`;
  return `$${v}`;
}

function TrendIcon({ trend }: { trend: string }) {
  if (trend === "improving")  return <TrendingUp  className="w-3 h-3 text-emerald-400" />;
  if (trend === "worsening")  return <TrendingDown className="w-3 h-3 text-red-400" />;
  return <Minus className="w-3 h-3 text-slate-500" />;
}

// ─── Section: Brand Health Scorecard ─────────────────────────────────────────

function BrandScorecard({ rows }: { rows: ScorecardRow[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-5">
        <Shield className="w-4 h-4 text-indigo-400" />
        <h3 className="text-sm font-semibold text-slate-100">Brand Health Scorecard</h3>
        <span className="ml-auto text-xs text-slate-600">7 dimensions · RAG status</span>
      </div>

      {/* Table header */}
      <div className="grid grid-cols-[1fr_64px_80px_64px_80px_80px] gap-2 px-3 pb-2 border-b border-slate-800 text-xs text-slate-600 uppercase tracking-wide font-medium">
        <span>Dimension</span>
        <span className="text-center">Score</span>
        <span className="text-center">Status</span>
        <span className="text-center">Trend</span>
        <span className="text-right">Churn Risk</span>
        <span className="text-right">Rev. at Risk</span>
      </div>

      <div className="space-y-0.5 mt-1">
        {rows.map((row) => {
          const isOpen = expanded === row.attribute;
          return (
            <div key={row.attribute}>
              <button
                onClick={() => setExpanded(isOpen ? null : row.attribute)}
                className="w-full grid grid-cols-[1fr_64px_80px_64px_80px_80px] gap-2 px-3 py-2.5 rounded-lg hover:bg-slate-800/50 transition-colors text-left items-center"
              >
                {/* Dimension */}
                <div className="flex items-center gap-2">
                  <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: row.colour }} />
                  <span className="text-sm font-medium text-slate-200 capitalize">{row.attribute}</span>
                </div>

                {/* Score bar */}
                <div className="flex flex-col items-center gap-1">
                  <span className={`text-sm font-bold tabular-nums ${STATUS_COLOUR[row.status]}`}>{row.score}</span>
                  <div className="w-10 h-1 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${row.score}%`, background: row.colour }}
                    />
                  </div>
                </div>

                {/* Status badge */}
                <div className="flex justify-center">
                  <span className={`inline-flex items-center px-1.5 py-0.5 rounded text-xs font-bold border ${STATUS_BG[row.status]} ${STATUS_COLOUR[row.status]}`}>
                    {STATUS_LABEL[row.status]}
                  </span>
                </div>

                {/* Trend */}
                <div className="flex items-center justify-center gap-1">
                  <TrendIcon trend={row.trend} />
                  <span className={`text-xs ${row.trend === "improving" ? "text-emerald-400" : row.trend === "worsening" ? "text-red-400" : "text-slate-500"}`}>
                    {row.trend_delta > 0 ? `${row.trend_delta}` : "—"}
                  </span>
                </div>

                {/* Churn risk */}
                <span className={`text-sm font-semibold tabular-nums text-right ${row.churn_risk > 0.5 ? "text-red-400" : row.churn_risk > 0.3 ? "text-amber-400" : "text-emerald-400"}`}>
                  {(row.churn_risk * 100).toFixed(0)}%
                </span>

                {/* Revenue at risk */}
                <span className="text-sm font-semibold tabular-nums text-right text-orange-300">
                  {fmt$M(row.revenue_at_risk)}
                </span>
              </button>

              {/* Expanded recommendation */}
              {isOpen && (
                <div className="mx-3 mb-2 px-4 py-3 bg-indigo-950/30 border border-indigo-500/15 rounded-lg">
                  <p className="text-xs text-slate-500 uppercase tracking-wide mb-1.5 font-medium">
                    Recommendation
                  </p>
                  <p className="text-sm text-slate-200 leading-relaxed">{row.recommendation}</p>
                  <div className="flex gap-4 mt-3 text-xs text-slate-600">
                    <span>Negative rate: <span className="text-slate-400 font-semibold">{(row.negative_rate * 100).toFixed(0)}%</span></span>
                    <span>Churn exposure: <span className="text-slate-400 font-semibold">{(row.churn_risk * 100).toFixed(0)}%</span></span>
                    <span>Revenue at risk: <span className="text-orange-400 font-semibold">{fmt$M(row.revenue_at_risk)}</span></span>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Section: Strategic Priority Matrix (SVG 2×2) ────────────────────────────

const QUADRANT_CFG = [
  { key: "quick_win",  label: "Quick Wins",       x: 0,   y: 0,   fill: "#10b981", desc: "High impact · Low effort" },
  { key: "strategic",  label: "Strategic Bets",   x: 50,  y: 0,   fill: "#6366f1", desc: "High impact · High effort" },
  { key: "fill_in",    label: "Fill-Ins",          x: 0,   y: 50,  fill: "#475569", desc: "Low impact · Low effort" },
  { key: "avoid",      label: "Defer & Reassess",      x: 50,  y: 50,  fill: "#7f1d1d", desc: "Low impact · High effort" },
];

function PriorityMatrix({ points }: { points: MatrixPoint[] }) {
  const [hovered, setHovered] = useState<MatrixPoint | null>(null);
  const W = 480; const H = 340;
  const PAD = { l: 44, r: 16, t: 16, b: 36 };
  const plotW = W - PAD.l - PAD.r;
  const plotH = H - PAD.t - PAD.b;

  // Map effort (0→1) to X, impact (1→0 flipped so high is top) to Y
  const toX = (effort: number) => PAD.l + effort * plotW;
  const toY = (impact: number) => PAD.t + (1 - impact) * plotH;

  return (
    <div className="card h-full">
      <div className="flex items-center gap-2 mb-4">
        <Target className="w-4 h-4 text-violet-400" />
        <h3 className="text-sm font-semibold text-slate-100">Strategic Priority Matrix</h3>
      </div>
      <div className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 340 }}>
          {/* Quadrant backgrounds */}
          <rect x={PAD.l} y={PAD.t} width={plotW / 2} height={plotH / 2} fill="#10b98108" stroke="#10b98120" strokeWidth={1} />
          <rect x={PAD.l + plotW / 2} y={PAD.t} width={plotW / 2} height={plotH / 2} fill="#6366f108" stroke="#6366f120" strokeWidth={1} />
          <rect x={PAD.l} y={PAD.t + plotH / 2} width={plotW / 2} height={plotH / 2} fill="#47556908" stroke="#47556920" strokeWidth={1} />
          <rect x={PAD.l + plotW / 2} y={PAD.t + plotH / 2} width={plotW / 2} height={plotH / 2} fill="#7f1d1d10" stroke="#7f1d1d30" strokeWidth={1} />

          {/* Quadrant labels */}
          <text x={PAD.l + 8} y={PAD.t + 16} fill="#10b981" fontSize={10} fontWeight={600}>Quick Wins ↗</text>
          <text x={PAD.l + plotW / 2 + 8} y={PAD.t + 16} fill="#818cf8" fontSize={10} fontWeight={600}>Strategic Bets</text>
          <text x={PAD.l + 8} y={PAD.t + plotH / 2 + 16} fill="#64748b" fontSize={10} fontWeight={600}>Fill-Ins</text>
          <text x={PAD.l + plotW / 2 + 8} y={PAD.t + plotH / 2 + 16} fill="#ef4444" fontSize={10} fontWeight={600}>Defer & Reassess</text>

          {/* Divider lines */}
          <line x1={PAD.l + plotW / 2} y1={PAD.t} x2={PAD.l + plotW / 2} y2={PAD.t + plotH} stroke="#334155" strokeDasharray="4 3" strokeWidth={1} />
          <line x1={PAD.l} y1={PAD.t + plotH / 2} x2={PAD.l + plotW} y2={PAD.t + plotH / 2} stroke="#334155" strokeDasharray="4 3" strokeWidth={1} />

          {/* Axis labels */}
          <text x={PAD.l + plotW / 2} y={H - 4} fill="#64748b" fontSize={9} textAnchor="middle">Effort to Fix →</text>
          <text x={12} y={PAD.t + plotH / 2} fill="#64748b" fontSize={9} textAnchor="middle" transform={`rotate(-90, 12, ${PAD.t + plotH / 2})`}>Business Impact ↑</text>

          {/* Issue dots */}
          {points.map((pt, i) => {
            const cx = toX(pt.effort);
            const cy = toY(pt.impact);
            const isHov = hovered?.issue === pt.issue;
            return (
              <g key={i}
                onMouseEnter={() => setHovered(pt)}
                onMouseLeave={() => setHovered(null)}
                style={{ cursor: "pointer" }}
              >
                <circle cx={cx} cy={cy} r={isHov ? 10 : 7} fill={pt.colour} opacity={isHov ? 0.95 : 0.75} />
                <circle cx={cx} cy={cy} r={isHov ? 10 : 7} fill="none" stroke={pt.colour} strokeWidth={1.5} opacity={0.6} />
              </g>
            );
          })}

          {/* Axis ticks */}
          {[0, 0.5, 1].map((v) => (
            <g key={v}>
              <text x={toX(v)} y={PAD.t + plotH + 14} fill="#475569" fontSize={8} textAnchor="middle">
                {v === 0 ? "Low" : v === 1 ? "High" : ""}
              </text>
              <text x={PAD.l - 6} y={toY(v) + 3} fill="#475569" fontSize={8} textAnchor="end">
                {v === 1 ? "High" : v === 0 ? "Low" : ""}
              </text>
            </g>
          ))}
        </svg>

        {/* Hover tooltip */}
        {hovered && (
          <div className="absolute bottom-0 left-0 right-0 mx-4 mb-1 px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-xs pointer-events-none">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full shrink-0" style={{ background: hovered.colour }} />
              <span className="font-semibold text-slate-100">{hovered.issue}</span>
              <span className="ml-auto capitalize text-slate-500">{hovered.attribute}</span>
            </div>
            <div className="flex gap-4 mt-1 text-slate-500">
              <span>Impact: <span className="text-slate-300">{(hovered.impact * 100).toFixed(0)}%</span></span>
              <span>Effort: <span className="text-slate-300">{(hovered.effort * 100).toFixed(0)}%</span></span>
              <span className={`ml-auto font-semibold ${
                hovered.quadrant === "quick_win"  ? "text-emerald-400" :
                hovered.quadrant === "strategic"  ? "text-indigo-400"  :
                hovered.quadrant === "fill_in"    ? "text-slate-400"   : "text-red-400"
              }`}>
                {hovered.quadrant === "quick_win" ? "Quick Win" : hovered.quadrant === "strategic" ? "Strategic Bet" : hovered.quadrant === "fill_in" ? "Fill-In" : "Defer & Reassess"}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3 pt-3 border-t border-slate-800">
        {[
          { colour: "#10b981", label: "onboarding"  },
          { colour: "#f97316", label: "reliability" },
          { colour: "#8b5cf6", label: "support"     },
          { colour: "#ef4444", label: "pricing"     },
          { colour: "#3b82f6", label: "performance" },
          { colour: "#f59e0b", label: "features"    },
          { colour: "#ec4899", label: "security"    },
        ].map(({ colour, label }) => (
          <div key={label} className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full" style={{ background: colour }} />
            <span className="text-xs text-slate-600 capitalize">{label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Section: Revenue at Risk Funnel ─────────────────────────────────────────

function RevenueAtRisk({ model }: { model: RevenueModel }) {
  const steps = [
    {
      label:   "Total Customer Base",
      value:   model.total_customers.toLocaleString(),
      sub:     "Active accounts",
      pct:     100,
      colour:  "#6366f1",
      icon:    Users,
    },
    {
      label:   "Dissatisfied Customers",
      value:   model.dissatisfied_count.toLocaleString(),
      sub:     `${model.dissatisfied_pct}% of base`,
      pct:     model.dissatisfied_pct,
      colour:  "#f59e0b",
      icon:    AlertTriangle,
    },
    {
      label:   "At-Risk of Churn",
      value:   model.churn_risk_count.toLocaleString(),
      sub:     `${model.churn_risk_pct}% of base`,
      pct:     model.churn_risk_pct,
      colour:  "#f97316",
      icon:    TrendingDown,
    },
    {
      label:   "Revenue at Risk",
      value:   fmt$M(model.revenue_at_risk),
      sub:     `Avg CLV ${fmt$M(model.avg_clv)} × churn count`,
      pct:     null,
      colour:  "#ef4444",
      icon:    DollarSign,
    },
  ];

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-5">
        <DollarSign className="w-4 h-4 text-orange-400" />
        <h3 className="text-sm font-semibold text-slate-100">Revenue at Risk Model</h3>
        <span className="ml-auto text-xs text-slate-600">Customer funnel → $ exposure</span>
      </div>

      <div className="flex items-stretch gap-0">
        {steps.map((step, i) => {
          const Icon = step.icon;
          const isLast = i === steps.length - 1;
          return (
            <div key={step.label} className="flex items-center flex-1 min-w-0">
              <div className={`flex-1 rounded-xl p-4 border ${isLast ? "border-red-500/30 bg-red-950/20" : "border-slate-800 bg-slate-900/60"}`}>
                <div className="flex items-center gap-1.5 mb-2">
                  <Icon className="w-3.5 h-3.5 shrink-0" style={{ color: step.colour }} />
                  <p className="text-xs text-slate-500 truncate">{step.label}</p>
                </div>
                <p className="text-xl font-bold tabular-nums" style={{ color: step.colour }}>
                  {step.value}
                </p>
                <p className="text-xs text-slate-600 mt-0.5">{step.sub}</p>
                {step.pct !== null && (
                  <div className="mt-2 h-1 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{ width: `${step.pct}%`, background: step.colour }}
                    />
                  </div>
                )}
              </div>
              {!isLast && (
                <ArrowRight className="w-4 h-4 text-slate-700 mx-1 shrink-0" />
              )}
            </div>
          );
        })}
      </div>

      <p className="text-xs text-slate-600 mt-3 pt-3 border-t border-slate-800">
        Model assumptions: Avg. CLV = {fmt$M(model.avg_clv)} · Churn risk derived from negative sentiment rate × attribute churn propensity score
      </p>
    </div>
  );
}

// ─── Section: Executive Action Plan ──────────────────────────────────────────

function ActionPlan({ actions }: { actions: ActionItem[] }) {
  return (
    <div className="card h-full">
      <div className="flex items-center gap-2 mb-5">
        <Flag className="w-4 h-4 text-amber-400" />
        <h3 className="text-sm font-semibold text-slate-100">Executive Action Plan</h3>
        <span className="ml-auto text-xs text-slate-600">{actions.length} initiatives</span>
      </div>

      <div className="space-y-3">
        {actions.map((a) => (
          <div
            key={a.rank}
            className={`p-3.5 rounded-xl border ${PRIORITY_BG[a.priority]}`}
          >
            {/* Header row */}
            <div className="flex items-start gap-3 mb-2">
              <div className="w-6 h-6 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-center shrink-0">
                <span className="text-xs font-bold text-slate-400">{a.rank}</span>
              </div>
              <p className="text-sm font-semibold text-slate-100 leading-snug flex-1">{a.action}</p>
              <span className={`text-xs font-bold shrink-0 ${PRIORITY_COLOUR[a.priority]}`}>
                {a.priority}
              </span>
            </div>

            {/* Meta row */}
            <div className="flex flex-wrap items-center gap-3 ml-9 text-xs text-slate-500">
              <div className="flex items-center gap-1">
                <Briefcase className="w-3 h-3" />
                <span>{a.owner}</span>
              </div>
              <div className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                <span>{a.timeline}</span>
              </div>
              <div className="flex items-center gap-1 ml-auto">
                <Award className="w-3 h-3 text-indigo-400" />
                <span className="text-indigo-300 font-semibold">{a.impact}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Health Score Ring ────────────────────────────────────────────────────────

function HealthRing({ score }: { score: number }) {
  const R = 36;
  const C = 2 * Math.PI * R;
  const dash = (score / 100) * C;
  const colour = score < 40 ? "#ef4444" : score < 60 ? "#f59e0b" : "#10b981";

  return (
    <div className="relative w-24 h-24 shrink-0">
      <svg viewBox="0 0 96 96" className="w-full h-full -rotate-90">
        <circle cx={48} cy={48} r={R} fill="none" stroke="#1e293b" strokeWidth={8} />
        <circle
          cx={48} cy={48} r={R} fill="none"
          stroke={colour} strokeWidth={8}
          strokeDasharray={`${dash} ${C}`}
          strokeLinecap="round"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold text-white tabular-nums">{score}</span>
        <span className="text-xs text-slate-600">/100</span>
      </div>
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────────────────────────────

export default function StrategyPage() {
  const { data, isLoading } = useQuery<StrategyData>({
    queryKey: ["strategy"],
    queryFn: () => api.get("/analytics/strategy").then((r) => r.data),
    staleTime: 120_000,
    refetchInterval: 180_000,
  });

  const overallHealth     = data?.overall_health     ?? 0;
  const churnRisk         = data?.churn_risk_index   ?? 0;
  const revenueAtRisk     = data?.total_revenue_at_risk ?? 0;
  const criticalCount     = (data?.brand_scorecard ?? []).filter((r) => r.status === "critical").length;
  const healthStatus      = overallHealth < 40 ? "critical" : overallHealth < 60 ? "warn" : "ok";

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center gap-2 text-slate-600">
          <Activity className="w-5 h-5 animate-spin" />
          <span className="text-sm">Building strategic intelligence…</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 pb-12">

      {/* ── Page Header ─────────────────────────────────────────────── */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Strategic Intelligence</h1>
          <p className="text-slate-400 text-sm mt-1">
            Board-ready brand health · Revenue exposure · Action roadmap
          </p>
        </div>
        <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold border ${STATUS_BG[healthStatus]} ${STATUS_COLOUR[healthStatus]}`}>
          {healthStatus === "ok" ? <CheckCircle className="w-3.5 h-3.5" /> : <AlertTriangle className="w-3.5 h-3.5" />}
          Brand {STATUS_LABEL[healthStatus]}
        </div>
      </div>

      {/* ── Top KPI Strip ───────────────────────────────────────────── */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Overall Health */}
        <div className="card lg:col-span-1 flex items-center gap-4">
          <HealthRing score={overallHealth} />
          <div>
            <p className="text-xs text-slate-500 uppercase tracking-wide font-medium">Brand Health</p>
            <p className="text-sm font-semibold text-slate-300 mt-0.5">Overall Score</p>
            <span className={`text-xs font-bold ${STATUS_COLOUR[healthStatus]}`}>
              {STATUS_LABEL[healthStatus]}
            </span>
          </div>
        </div>

        {/* Revenue at Risk */}
        <div className="kpi-card border-red-500/20 hover:border-red-500/40">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wide font-medium">Revenue at Risk</p>
              <p className="text-2xl font-bold text-red-300 mt-1 tabular-nums">{fmt$M(revenueAtRisk)}</p>
              <p className="text-xs text-slate-600 mt-0.5">From churn-risk customers</p>
            </div>
            <div className="p-2 rounded-lg bg-slate-800"><DollarSign className="w-5 h-5 text-red-400" /></div>
          </div>
        </div>

        {/* Churn Risk Index */}
        <div className="kpi-card border-orange-500/20 hover:border-orange-500/40">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wide font-medium">Churn Risk Index</p>
              <p className="text-2xl font-bold text-orange-300 mt-1 tabular-nums">{(churnRisk * 100).toFixed(0)}%</p>
              <p className="text-xs text-slate-600 mt-0.5">Avg across attributes</p>
            </div>
            <div className="p-2 rounded-lg bg-slate-800"><Zap className="w-5 h-5 text-orange-400" /></div>
          </div>
        </div>

        {/* Critical Issues */}
        <div className="kpi-card border-red-700/20 hover:border-red-700/40">
          <div className="flex items-start justify-between">
            <div>
              <p className="text-xs text-slate-500 uppercase tracking-wide font-medium">Critical Dimensions</p>
              <p className="text-2xl font-bold text-red-400 mt-1 tabular-nums">{criticalCount}</p>
              <p className="text-xs text-slate-600 mt-0.5">Require immediate action</p>
            </div>
            <div className="p-2 rounded-lg bg-slate-800"><AlertTriangle className="w-5 h-5 text-red-400" /></div>
          </div>
        </div>
      </div>

      {/* ── Brand Health Scorecard ───────────────────────────────────── */}
      <BrandScorecard rows={data?.brand_scorecard ?? []} />

      {/* ── Priority Matrix + Action Plan ───────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        <div className="lg:col-span-3">
          <PriorityMatrix points={data?.priority_matrix ?? []} />
        </div>
        <div className="lg:col-span-2">
          <ActionPlan actions={data?.action_plan ?? []} />
        </div>
      </div>

      {/* ── Revenue at Risk Funnel ───────────────────────────────────── */}
      {data?.revenue_model && (
        <RevenueAtRisk model={data.revenue_model} />
      )}

      {/* ── Methodology note ────────────────────────────────────────── */}
      <div className="flex items-start gap-3 px-4 py-3 bg-slate-900/50 border border-slate-800 rounded-xl">
        <Shield className="w-4 h-4 text-indigo-400 shrink-0 mt-0.5" />
        <p className="text-xs text-slate-500 leading-relaxed">
          <span className="text-slate-400 font-medium">Methodology: </span>
          Brand health scores are computed from ML-predicted negative sentiment rates per attribute. 
          Churn risk is modelled via attribute-specific propensity scores weighted by signal volume.
          Revenue at risk = churn-risk customer count × average customer lifetime value.
          Priority matrix positions are derived from business impact (severity × volume) and internal effort estimates.
          All figures update in real time as new predictions flow through the platform.
        </p>
      </div>
    </div>
  );
}
