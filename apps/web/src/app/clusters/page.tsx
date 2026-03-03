"use client";
import { ClusterInsights } from "@/components/dashboard/index";
export default function ClustersPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Root-Cause Clusters</h1>
        <p className="text-slate-400 text-sm mt-1">Embedding-based topic clustering for root-cause analysis</p>
      </div>
      <ClusterInsights />
    </div>
  );
}
