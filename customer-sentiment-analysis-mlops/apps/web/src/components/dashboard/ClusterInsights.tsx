"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Activity } from "lucide-react";

const SENTIMENT_COLOR: Record<string, string> = {
  positive: "#10b981",
  neutral:  "#f59e0b",
  negative: "#ef4444",
};

interface Cluster {
  cluster_id: number;
  theme: string;
  dominant_sentiment: string;
  message_count: number;
  avg_severity: number;
  avg_business_impact: number;
  top_terms: string[];
}

interface ClustersResponse {
  clusters: Cluster[];
}

export function ClusterInsights() {
  const { data } = useQuery<ClustersResponse>({
    queryKey: ["clusters"],
    queryFn: () => api.get("/analytics/clusters").then((r) => r.data),
    staleTime: 300_000,
  });

  const clusters = (data?.clusters ?? []).slice(0, 5);

  return (
    <div className="card h-full">
      <div className="flex items-center gap-2 mb-4">
        <Activity className="w-4 h-4 text-indigo-400" />
        <h3 className="text-sm font-semibold text-slate-100">Root-Cause Clusters</h3>
        <span className="ml-auto text-xs text-slate-500">{clusters.length} clusters</span>
      </div>
      <div className="space-y-2.5">
        {clusters.map((c) => (
          <div
            key={c.cluster_id}
            className="flex items-center gap-3 p-2.5 rounded-lg bg-slate-800/50 hover:bg-slate-800 transition-colors"
          >
            <div
              className="w-2 h-8 rounded-full shrink-0"
              style={{ background: SENTIMENT_COLOR[c.dominant_sentiment] ?? "#6366f1" }}
            />
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-slate-100 truncate">{c.theme}</p>
              <p className="text-xs text-slate-500">
                {c.message_count} msgs · impact {(c.avg_business_impact * 100).toFixed(0)}
              </p>
            </div>
            <div className="text-right shrink-0">
              <p className="text-xs text-slate-400">{(c.avg_severity * 100).toFixed(0)} sev</p>
              <div className="flex gap-1 mt-1">
                {(c.top_terms ?? []).slice(0, 2).map((t) => (
                  <span key={t} className="text-xs bg-slate-700 text-slate-400 px-1.5 py-0.5 rounded">
                    {t}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
