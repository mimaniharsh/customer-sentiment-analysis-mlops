"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { FileText, Download, Plus, Clock } from "lucide-react";

export default function ReportsPage() {
  const { data } = useQuery({
    queryKey: ["reports"],
    queryFn: () => api.get("/reports").then((r) => r.data),
  });

  const reports: any[] = data?.reports ?? [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Reports</h1>
          <p className="text-slate-400 text-sm mt-1">Generate and download board-ready intelligence reports</p>
        </div>
        <button className="flex items-center gap-2 px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white text-sm font-medium rounded-lg transition-colors">
          <Plus className="w-4 h-4" />
          Generate Report
        </button>
      </div>

      {/* Report type cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          { title: "Executive Summary", desc: "30-day sentiment overview with KPIs and trend analysis", icon: "📊" },
          { title: "Drift Analysis Report", desc: "Model stability analysis with PSI and KS statistics", icon: "📈" },
          { title: "Cluster Intelligence", desc: "Root-cause analysis with top themes and business impact", icon: "🔍" },
        ].map((rt) => (
          <div key={rt.title} className="card hover:border-indigo-500/40 transition-colors cursor-pointer group">
            <div className="text-2xl mb-3">{rt.icon}</div>
            <h3 className="text-sm font-semibold text-slate-100 group-hover:text-indigo-300 transition-colors">{rt.title}</h3>
            <p className="text-xs text-slate-500 mt-1">{rt.desc}</p>
            <button className="mt-4 text-xs text-indigo-400 hover:text-indigo-300 font-medium">Generate →</button>
          </div>
        ))}
      </div>

      {/* Reports list */}
      <div className="card">
        <h3 className="text-sm font-semibold text-slate-100 mb-4">Recent Reports</h3>
        {reports.length === 0 ? (
          <div className="text-center py-12">
            <FileText className="w-10 h-10 text-slate-700 mx-auto mb-3" />
            <p className="text-slate-500 text-sm">No reports generated yet</p>
            <p className="text-slate-600 text-xs mt-1">Click Generate Report to create your first intelligence report</p>
          </div>
        ) : (
          <div className="divide-y divide-slate-800">
            {reports.map((r: any) => (
              <div key={r.id} className="flex items-center gap-4 py-3">
                <FileText className="w-4 h-4 text-slate-500 shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-slate-200 truncate">{r.title}</p>
                  <p className="text-xs text-slate-500 flex items-center gap-1 mt-0.5">
                    <Clock className="w-3 h-3" /> {r.created_at}
                  </p>
                </div>
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  r.status === "ready" ? "bg-emerald-500/15 text-emerald-400" : "bg-amber-500/15 text-amber-400"
                }`}>{r.status}</span>
                <button className="p-1.5 text-slate-500 hover:text-slate-200 hover:bg-slate-800 rounded-lg transition-colors">
                  <Download className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
