"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";

export function AttributeHeatmap() {
  const { data } = useQuery({
    queryKey: ["attributes"],
    queryFn: () => api.get("/analytics/attributes?days=30").then((r) => r.data),
  });

  const heatmap: any[] = data?.heatmap ?? [];
  const attrs = [...new Set(heatmap.map((h: any) => h.attribute))];
  const sentiments = ["positive", "neutral", "negative"];
  const COLOR: Record<string, string> = {
    positive: "#10b981",
    neutral: "#f59e0b",
    negative: "#ef4444",
  };

  return (
    <div className="card">
      <div className="mb-4">
        <h3 className="text-sm font-semibold text-slate-100">Attribute Heatmap</h3>
        <p className="text-xs text-slate-500 mt-0.5">Sentiment distribution by business attribute</p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr>
              <th className="text-left text-slate-500 pb-2 pr-4 font-medium">Attribute</th>
              {sentiments.map((s) => (
                <th key={s} className="text-center pb-2 px-2 font-medium capitalize" style={{ color: COLOR[s] }}>
                  {s}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="space-y-1">
            {attrs.map((attr) => (
              <tr key={attr} className="border-t border-slate-800">
                <td className="py-2 pr-4 text-slate-300 font-medium capitalize">{attr}</td>
                {sentiments.map((s) => {
                  const cell = heatmap.find((h: any) => h.attribute === attr && h.sentiment === s);
                  const count = cell?.count ?? 0;
                  const maxCount = Math.max(...heatmap.map((h: any) => h.count));
                  const intensity = count / maxCount;
                  return (
                    <td key={s} className="text-center py-2 px-2">
                      <div
                        className="inline-flex items-center justify-center w-12 h-7 rounded font-semibold text-xs"
                        style={{
                          background: `${COLOR[s]}${Math.round(intensity * 40 + 10).toString(16).padStart(2, "0")}`,
                          color: intensity > 0.5 ? COLOR[s] : "#64748b",
                          border: `1px solid ${COLOR[s]}22`,
                        }}
                      >
                        {count}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
