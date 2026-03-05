// TopBar.tsx
"use client";
import { Bell, Search, User } from "lucide-react";
export function TopBar() {
  return (
    <header className="h-14 bg-slate-900 border-b border-slate-800 flex items-center px-6 gap-4 shrink-0">
      <div className="flex items-center gap-2 flex-1 max-w-md">
        <Search className="w-4 h-4 text-slate-500" />
        <input
          placeholder="Search messages, reports…"
          className="bg-slate-800 text-sm text-slate-300 placeholder-slate-600 rounded-lg px-3 py-1.5 flex-1 border border-slate-700 focus:outline-none focus:border-indigo-500"
        />
      </div>
      <div className="flex items-center gap-3 ml-auto">
        <button className="relative p-2 rounded-lg text-slate-400 hover:text-slate-100 hover:bg-slate-800 transition-colors">
          <Bell className="w-4 h-4" />
          <span className="absolute top-1 right-1 w-1.5 h-1.5 bg-red-500 rounded-full" />
        </button>
        <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center">
          <User className="w-4 h-4 text-white" />
        </div>
      </div>
    </header>
  );
}
