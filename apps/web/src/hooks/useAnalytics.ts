// apps/web/src/hooks/useAnalytics.ts
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { OverviewData, TimeSeriesEntry, ClusterEntry, DriftRecord, HeatmapEntry } from "@/types";

export function useOverview(days = 30) {
  return useQuery<OverviewData>({
    queryKey: ["overview", days],
    queryFn: () => api.get(`/analytics/overview?days=${days}`).then((r) => r.data),
    refetchInterval: 60_000,
  });
}

export function useTimeSeries(days = 30) {
  return useQuery<{ series: TimeSeriesEntry[] }>({
    queryKey: ["timeseries", days],
    queryFn: () => api.get(`/analytics/timeseries?days=${days}`).then((r) => r.data),
    refetchInterval: 120_000,
  });
}

export function useClusters() {
  return useQuery<{ clusters: ClusterEntry[] }>({
    queryKey: ["clusters"],
    queryFn: () => api.get("/analytics/clusters").then((r) => r.data),
    staleTime: 300_000,
  });
}

export function useDrift() {
  return useQuery<{ drift_history: DriftRecord[]; current_psi: number }>({
    queryKey: ["drift"],
    queryFn: () => api.get("/analytics/drift").then((r) => r.data),
    staleTime: 300_000,
  });
}

export function useAttributes(days = 30) {
  return useQuery<{ heatmap: HeatmapEntry[] }>({
    queryKey: ["attributes", days],
    queryFn: () => api.get(`/analytics/attributes?days=${days}`).then((r) => r.data),
    staleTime: 120_000,
  });
}

export function useUrgency(days = 7) {
  return useQuery({
    queryKey: ["urgency", days],
    queryFn: () => api.get(`/analytics/urgency?days=${days}`).then((r) => r.data),
    staleTime: 60_000,
  });
}
