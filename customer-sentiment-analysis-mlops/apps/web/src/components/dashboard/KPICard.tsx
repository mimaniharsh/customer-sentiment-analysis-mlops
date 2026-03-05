"use client";
import { clsx } from "clsx";
import { ReactNode } from "react";

interface KPICardProps {
  title: string;
  value: string;
  subtitle?: string;
  icon: ReactNode;
  accent?: "emerald" | "red" | "amber" | "blue" | "violet" | "green";
  loading?: boolean;
}

const ACCENT_CLASSES: Record<string, string> = {
  emerald: "border-emerald-500/20 hover:border-emerald-500/40",
  red: "border-red-500/20 hover:border-red-500/40",
  amber: "border-amber-500/20 hover:border-amber-500/40",
  blue: "border-blue-500/20 hover:border-blue-500/40",
  violet: "border-violet-500/20 hover:border-violet-500/40",
  green: "border-green-500/20 hover:border-green-500/40",
};

export function KPICard({ title, value, subtitle, icon, accent, loading }: KPICardProps) {
  return (
    <div className={clsx(
      "kpi-card",
      accent ? ACCENT_CLASSES[accent] : "hover:border-slate-700"
    )}>
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <p className="text-xs text-slate-500 font-medium uppercase tracking-wide">{title}</p>
          {loading ? (
            <div className="h-7 w-20 bg-slate-800 rounded animate-pulse" />
          ) : (
            <p className="text-2xl font-bold text-white">{value}</p>
          )}
          {subtitle && <p className="text-xs text-slate-500">{subtitle}</p>}
        </div>
        <div className="p-2 rounded-lg bg-slate-800">{icon}</div>
      </div>
    </div>
  );
}
