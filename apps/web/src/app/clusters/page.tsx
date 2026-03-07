"use client";
import { BubbleChart } from "@/components/charts/BubbleChart";
import { RootCausePinpointer } from "@/components/charts/RootCausePinpointer";
import { ClusterInsights } from "@/components/dashboard/ClusterInsights";
import { AttributeHeatmap } from "@/components/charts/AttributeHeatmap";

export default function ClustersPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Root-Cause Analysis</h1>
        <p className="text-slate-400 text-sm mt-1">
          Multi-level breakdown — from category to exact issue to recommended action
        </p>
      </div>

      <div>
        <p className="text-xs text-slate-600 uppercase tracking-widest mb-3">
          Step 1 — Prioritise the category
        </p>
        <BubbleChart />
      </div>

      <div>
        <p className="text-xs text-slate-600 uppercase tracking-widest mb-3">
          Step 2 — Pinpoint the exact issue
        </p>
        <RootCausePinpointer />
      </div>

      <div>
        <p className="text-xs text-slate-600 uppercase tracking-widest mb-3">
          Step 3 — Contextual signals
        </p>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <ClusterInsights />
          <AttributeHeatmap />
        </div>
      </div>
    </div>
  );
}
