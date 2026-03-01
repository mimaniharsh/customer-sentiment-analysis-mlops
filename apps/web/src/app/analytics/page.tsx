"use client";
import { SentimentTrendChart } from "@/components/charts/SentimentTrendChart";
import { AttributeHeatmap } from "@/components/charts/AttributeHeatmap";
import { UrgencyBreakdown } from "@/components/charts/UrgencyBreakdown";

export default function AnalyticsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Analytics</h1>
        <p className="text-slate-400 text-sm mt-1">Deep-dive sentiment and attribute analysis</p>
      </div>
      <SentimentTrendChart />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <AttributeHeatmap />
        <UrgencyBreakdown />
      </div>
    </div>
  );
}
