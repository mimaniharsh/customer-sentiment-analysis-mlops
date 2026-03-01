"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { Network } from "lucide-react";

const SENTIMENT_COLOR: Record<string, string> = {
  positive: "#10b981",
  neutral: "#f59e0b",
  negative: "#ef4444",
};

export function ClusterInsights() {
  const { data, isLoading } = useQuery({
    queryKey: ["clusters"],
    queryFn: () => api.get("/analytics/clusters").then((r) => r.data),
  });

  const clusters: any[] = (data?.clusters ?? []).slice(0, 6);

  return (
    <div className="card h-full">
      <div className="flex items-center gap-2 mb-4">
        <Network className="w-4 h-4 text-indigo-400" />
        <h3 className="text-sm font-semibold text-slate-100">Root-Cause Clusters</h3>
        <span className="ml-auto text-xs text-slate-500">{clusters.length} clusters</span>
      </div>
      {isLoading ? (
        <div className="space-y-2">
          {[1,2,3].map((i) => <div key={i} className="h-14 bg-slate-800 rounded-lg animate-pulse" />)}
        </div>
      ) : (
        <div className="space-y-2">
          {clusters.map((c: any) => (
            <div key={c.cluster_id} className="flex items-center gap-3 p-2.5 rounded-lg bg-slate-800/50 hover:bg-slate-800 transition-colors cursor-default">
              <div className="w-1.5 h-10 rounded-full shrink-0" style={{ background: SENTIMENT_COLOR[c.dominant_sentiment] ?? "#6366f1" }} />
              <div className="flex-1 min-w-0">
                <p className="text-xs font-semibold text-slate-100 truncate">{c.theme}</p>
                <p className="text-xs text-slate-500 mt-0.5">{c.message_count} messages · impact {(c.avg_business_impact * 100).toFixed(0)}%</p>
              </div>
              <div className="shrink-0 text-right">
                <p className="text-xs text-slate-400 font-medium">sev {(c.avg_severity * 100).toFixed(0)}</p>
                <div className="flex gap-1 mt-1 justify-end">
                  {(c.top_terms ?? []).slice(0, 2).map((t: string) => (
                    <span key={t} className="text-xs bg-slate-700 text-slate-400 px-1.5 py-0.5 rounded">{t}</span>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
