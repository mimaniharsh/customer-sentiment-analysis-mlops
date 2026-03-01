"use client";

import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { KPICard } from "@/components/dashboard/KPICard";
import { SentimentTrendChart } from "@/components/charts/SentimentTrendChart";
import { UrgencyBreakdown } from "@/components/charts/UrgencyBreakdown";
import { AttributeHeatmap } from "@/components/charts/AttributeHeatmap";
import { DriftIndicator } from "@/components/dashboard/DriftIndicator";
import { ClusterInsights } from "@/components/dashboard/ClusterInsights";
import { BusinessImpactSimulator } from "@/components/dashboard/BusinessImpactSimulator";
import {
  MessageSquare, TrendingUp, TrendingDown, AlertTriangle,
  Activity, Target, Zap, Shield,
} from "lucide-react";

export default function DashboardPage() {
  const { data: overview, isLoading } = useQuery({
    queryKey: ["overview"],
    queryFn: () => api.get("/analytics/overview?days=30").then((r) => r.data),
    refetchInterval: 60_000,
  });

  const kpis = overview?.kpis;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Intelligence Overview</h1>
          <p className="text-slate-400 text-sm mt-1">Last 30 days · Auto-refreshes every 60s</p>
        </div>
        <div className="flex items-center gap-2">
          <span className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-semibold ${
            overview?.trend === "improving"
              ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
              : overview?.trend === "worsening"
              ? "bg-red-500/15 text-red-400 border border-red-500/30"
              : "bg-amber-500/15 text-amber-400 border border-amber-500/30"
          }`}>
            {overview?.trend === "improving" ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
            Trend: {overview?.trend ?? "—"}
          </span>
          <span className="bg-slate-800 border border-slate-700 text-slate-300 px-3 py-1.5 rounded-full text-xs font-medium">
            Health Score: {overview?.sentiment_health_score ?? "—"}
          </span>
        </div>
      </div>

      {/* KPI Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard
          title="Total Messages"
          value={kpis?.total_messages?.toLocaleString() ?? "—"}
          icon={<MessageSquare className="w-5 h-5 text-indigo-400" />}
          loading={isLoading}
        />
        <KPICard
          title="Positive Rate"
          value={kpis ? `${(kpis.positive_rate * 100).toFixed(1)}%` : "—"}
          icon={<TrendingUp className="w-5 h-5 text-emerald-400" />}
          accent="emerald"
          loading={isLoading}
        />
        <KPICard
          title="Negative Rate"
          value={kpis ? `${(kpis.negative_rate * 100).toFixed(1)}%` : "—"}
          icon={<TrendingDown className="w-5 h-5 text-red-400" />}
          accent="red"
          loading={isLoading}
        />
        <KPICard
          title="Critical Alerts"
          value={kpis?.critical_urgency_count?.toLocaleString() ?? "—"}
          icon={<AlertTriangle className="w-5 h-5 text-amber-400" />}
          accent="amber"
          loading={isLoading}
        />
        <KPICard
          title="Avg Confidence"
          value={kpis ? `${(kpis.avg_confidence * 100).toFixed(1)}%` : "—"}
          icon={<Target className="w-5 h-5 text-blue-400" />}
          loading={isLoading}
        />
        <KPICard
          title="Business Impact"
          value={kpis ? `${(kpis.avg_business_impact * 100).toFixed(1)}` : "—"}
          subtitle="avg impact score"
          icon={<Zap className="w-5 h-5 text-violet-400" />}
          accent="violet"
          loading={isLoading}
        />
        <KPICard
          title="Model Drift"
          value="PSI: 0.14"
          subtitle="Within threshold"
          icon={<Activity className="w-5 h-5 text-teal-400" />}
          loading={isLoading}
        />
        <KPICard
          title="SLA Compliance"
          value="98.2%"
          subtitle="Endpoint uptime"
          icon={<Shield className="w-5 h-5 text-green-400" />}
          accent="green"
          loading={isLoading}
        />
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2">
          <SentimentTrendChart />
        </div>
        <div>
          <UrgencyBreakdown />
        </div>
      </div>

      {/* Charts Row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <AttributeHeatmap />
        <DriftIndicator />
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <ClusterInsights />
        <BusinessImpactSimulator />
      </div>
    </div>
  );
}
