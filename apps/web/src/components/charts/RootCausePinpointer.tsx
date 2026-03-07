"use client";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { AlertTriangle, CheckCircle, Activity } from "lucide-react";

// ── Types ──────────────────────────────────────────────────────────────────

interface Issue {
  issue: string;
  attribute: string;
  count: number;
  priority_score: number;
  avg_severity: number;
  negative_rate: number;
  risk_tier: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW";
  count_pct: number;
}

interface IssuesResponse {
  issues: Issue[];
  total_issues: number;
  top_issue: string | null;
  top_attribute: string | null;
}

// ── Colour maps ────────────────────────────────────────────────────────────

const RISK_COLOUR: Record<string, string> = {
  CRITICAL: "#ef4444",
  HIGH:     "#f97316",
  MEDIUM:   "#f59e0b",
  LOW:      "#10b981",
};

const RISK_TEXT: Record<string, string> = {
  CRITICAL: "text-red-400",
  HIGH:     "text-orange-400",
  MEDIUM:   "text-amber-400",
  LOW:      "text-emerald-400",
};

const RISK_BG: Record<string, string> = {
  CRITICAL: "bg-red-950/50 border-red-500/25",
  HIGH:     "bg-orange-950/50 border-orange-500/25",
  MEDIUM:   "bg-amber-950/50 border-amber-500/25",
  LOW:      "bg-emerald-950/40 border-emerald-500/20",
};

const ATTR_COLOUR: Record<string, string> = {
  reliability: "#f97316",
  support:     "#8b5cf6",
  pricing:     "#ef4444",
  performance: "#3b82f6",
  onboarding:  "#10b981",
  features:    "#f59e0b",
  security:    "#ec4899",
};

// ── Filter tabs ────────────────────────────────────────────────────────────

const FILTERS = ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"] as const;
type Filter = typeof FILTERS[number];

// ── Single issue row ───────────────────────────────────────────────────────

function IssueRow({ issue, rank, maxPriority }: { issue: Issue; rank: number; maxPriority: number }) {
  const attrColour = ATTR_COLOUR[issue.attribute] ?? "#6366f1";
  const barWidth   = maxPriority > 0 ? (issue.priority_score / maxPriority) * 100 : 0;
  const riskColour = RISK_COLOUR[issue.risk_tier] ?? "#6366f1";

  return (
    <div className="group flex items-center gap-3 px-3 py-2.5 rounded-xl hover:bg-slate-800/50 transition-colors">
      {/* Rank */}
      <span className="text-xs font-mono text-slate-700 w-5 shrink-0 text-right">{rank}</span>

      {/* Attribute colour stripe */}
      <div
        className="w-1 h-8 rounded-full shrink-0"
        style={{ background: attrColour }}
      />

      {/* Main content */}
      <div className="flex-1 min-w-0">
        {/* Row 1: issue name + attribute chip + risk tier */}
        <div className="flex items-center gap-2 mb-1.5 flex-wrap">
          <span className="text-sm font-semibold text-slate-100 truncate">{issue.issue}</span>
          <span
            className="text-xs px-1.5 py-0.5 rounded-md capitalize shrink-0 font-medium"
            style={{ background: attrColour + "20", color: attrColour }}
          >
            {issue.attribute}
          </span>
          <span className={`ml-auto text-xs font-bold shrink-0 ${RISK_TEXT[issue.risk_tier]}`}>
            {issue.risk_tier}
          </span>
        </div>

        {/* Row 2: priority bar + stats */}
        <div className="flex items-center gap-3">
          {/* Priority bar */}
          <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-700"
              style={{ width: `${barWidth}%`, background: riskColour }}
            />
          </div>

          {/* Stats */}
          <div className="flex items-center gap-3 shrink-0 text-xs text-slate-500">
            <span className="tabular-nums">{issue.count} signals</span>
            <span className="tabular-nums">{(issue.avg_severity * 100).toFixed(0)}% sev</span>
            <span className="tabular-nums">{(issue.negative_rate * 100).toFixed(0)}% neg</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────

export function RootCausePinpointer() {
  const [filter, setFilter] = useState<Filter>("All");
  const [showAll, setShowAll] = useState(false);

  const { data, isLoading } = useQuery<IssuesResponse>({
    queryKey: ["issues"],
    queryFn: () => api.get("/analytics/issues").then((r) => r.data),
    staleTime: 60_000,
    refetchInterval: 120_000,
  });

  const allIssues = data?.issues ?? [];

  const filtered =
    filter === "All"
      ? allIssues
      : allIssues.filter((i) => i.risk_tier === filter);

  const visible = showAll ? filtered : filtered.slice(0, 10);
  const maxPriority = filtered[0]?.priority_score ?? 1;

  // Summary counts per tier
  const tierCounts = FILTERS.slice(1).map((tier) => ({
    tier,
    count: allIssues.filter((i) => i.risk_tier === tier).length,
  }));

  if (isLoading) {
    return (
      <div className="card flex items-center justify-center h-48">
        <div className="flex items-center gap-2 text-slate-600">
          <Activity className="w-4 h-4 animate-spin" />
          <span className="text-sm">Loading issue data…</span>
        </div>
      </div>
    );
  }

  return (
    <div className="card space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-sm font-semibold text-slate-100">Root Cause Pinpointer</h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Exact issues ranked by priority score (volume × severity)
          </p>
        </div>
        {/* Top issue badge */}
        {data?.top_issue && (
          <div className="flex items-center gap-1.5 px-2 py-1 bg-red-950/40 border border-red-500/25 rounded-lg max-w-[180px]">
            <AlertTriangle className="w-3 h-3 text-red-400 shrink-0" />
            <span className="text-xs text-red-300 truncate">{data.top_issue}</span>
          </div>
        )}
      </div>

      {/* Tier summary pills */}
      <div className="flex items-center gap-2 flex-wrap">
        {tierCounts.map(({ tier, count }) => (
          <div
            key={tier}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-xs font-semibold ${RISK_BG[tier]}`}
          >
            <div
              className="w-1.5 h-1.5 rounded-full"
              style={{ background: RISK_COLOUR[tier] }}
            />
            <span className={RISK_TEXT[tier]}>{tier}</span>
            <span className="text-slate-500 font-normal">{count}</span>
          </div>
        ))}
        <span className="ml-auto text-xs text-slate-600">{allIssues.length} total issues</span>
      </div>

      {/* Filter tabs */}
      <div className="flex gap-1 p-1 bg-slate-900 rounded-lg border border-slate-800">
        {FILTERS.map((f) => (
          <button
            key={f}
            onClick={() => { setFilter(f); setShowAll(false); }}
            className={`flex-1 py-1.5 text-xs font-semibold rounded-md transition-all ${
              filter === f
                ? f === "All"
                  ? "bg-indigo-600/30 text-indigo-300 border border-indigo-500/30"
                  : `text-white border`
                : "text-slate-500 hover:text-slate-300"
            }`}
            style={
              filter === f && f !== "All"
                ? {
                    background: RISK_COLOUR[f] + "25",
                    borderColor: RISK_COLOUR[f] + "50",
                    color: RISK_COLOUR[f],
                  }
                : {}
            }
          >
            {f}
          </button>
        ))}
      </div>

      {/* Issue list */}
      {filtered.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-10 text-slate-700">
          <CheckCircle className="w-8 h-8 mb-2 opacity-30" />
          <p className="text-sm">No {filter} issues detected</p>
        </div>
      ) : (
        <div className="space-y-0.5">
          {visible.map((issue, i) => (
            <IssueRow
              key={`${issue.attribute}-${issue.issue}`}
              issue={issue}
              rank={i + 1}
              maxPriority={maxPriority}
            />
          ))}
        </div>
      )}

      {/* Show more / less toggle */}
      {filtered.length > 10 && (
        <button
          onClick={() => setShowAll(!showAll)}
          className="w-full py-2 text-xs text-slate-500 hover:text-slate-200 border border-slate-800 hover:border-slate-700 rounded-xl transition-all"
        >
          {showAll
            ? "Show less"
            : `Show ${filtered.length - 10} more issues`}
        </button>
      )}

      {/* Attribution key */}
      <div className="flex flex-wrap gap-x-4 gap-y-1.5 pt-1 border-t border-slate-800">
        {Object.entries(ATTR_COLOUR).map(([attr, colour]) => (
          <div key={attr} className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full" style={{ background: colour }} />
            <span className="text-xs text-slate-600 capitalize">{attr}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
