"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard, BarChart2, FileText, Network,
  Settings, Zap, Activity, ChevronRight, FlaskConical,
} from "lucide-react";
import { clsx } from "clsx";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/analytics", label: "Analytics", icon: BarChart2 },
  { href: "/clusters", label: "Clusters", icon: Network },
  { href: "/reports", label: "Reports", icon: FileText },
  { href: "/demo", label: "Live Demo", icon: FlaskConical },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex flex-col w-64 bg-slate-900 border-r border-slate-800 shrink-0">
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 py-5 border-b border-slate-800">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center">
          <Zap className="w-4 h-4 text-white" />
        </div>
        <div>
          <p className="text-sm font-bold text-white leading-tight">Sentient</p>
          <p className="text-xs text-slate-500">Intelligence Platform</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={clsx(
                "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all group",
                active
                  ? "bg-indigo-600/20 text-indigo-300 border border-indigo-500/30"
                  : "text-slate-400 hover:text-slate-100 hover:bg-slate-800"
              )}
            >
              <Icon className={clsx("w-4 h-4", active ? "text-indigo-400" : "text-slate-500 group-hover:text-slate-300")} />
              {label}
              {active && <ChevronRight className="w-3 h-3 ml-auto text-indigo-400" />}
            </Link>
          );
        })}
      </nav>

      {/* Status bar */}
      <div className="px-4 py-4 border-t border-slate-800">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-xs text-slate-400">Endpoint InService</span>
          <Activity className="w-3 h-3 text-slate-600 ml-auto" />
        </div>
      </div>
    </aside>
  );
}
