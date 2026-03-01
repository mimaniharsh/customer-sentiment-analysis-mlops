"use client";
import { Settings, Shield, Bell, Database, Key, Webhook } from "lucide-react";

const SECTIONS = [
  {
    icon: Shield,
    title: "Security & Access",
    items: ["RBAC role assignments", "API key management", "Session timeout policy"],
  },
  {
    icon: Bell,
    title: "Alerts & Notifications",
    items: ["Drift alert threshold (PSI > 0.2)", "Critical urgency email alerts", "Weekly digest report"],
  },
  {
    icon: Database,
    title: "Data & Retention",
    items: ["Message retention policy", "Export data as CSV / JSON", "GDPR data deletion requests"],
  },
  {
    icon: Webhook,
    title: "Integrations",
    items: ["Webhook endpoints for predictions", "Slack alert integration", "Zapier / Make connectors"],
  },
];

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Settings</h1>
        <p className="text-slate-400 text-sm mt-1">Configure your tenant, alerts, and integrations</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {SECTIONS.map(({ icon: Icon, title, items }) => (
          <div key={title} className="card">
            <div className="flex items-center gap-2 mb-4">
              <div className="p-1.5 bg-slate-800 rounded-lg">
                <Icon className="w-4 h-4 text-indigo-400" />
              </div>
              <h3 className="text-sm font-semibold text-slate-100">{title}</h3>
            </div>
            <ul className="space-y-2">
              {items.map((item) => (
                <li key={item} className="flex items-center justify-between py-2 border-t border-slate-800">
                  <span className="text-xs text-slate-400">{item}</span>
                  <button className="text-xs text-indigo-400 hover:text-indigo-300 font-medium transition-colors">Configure</button>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="card border-red-500/20">
        <div className="flex items-center gap-2 mb-3">
          <Key className="w-4 h-4 text-red-400" />
          <h3 className="text-sm font-semibold text-red-300">Danger Zone</h3>
        </div>
        <p className="text-xs text-slate-500 mb-4">
          These actions are irreversible. Proceed with caution.
        </p>
        <div className="flex gap-3">
          <button className="px-3 py-1.5 text-xs font-medium text-red-400 border border-red-500/30 rounded-lg hover:bg-red-500/10 transition-colors">
            Delete all predictions
          </button>
          <button className="px-3 py-1.5 text-xs font-medium text-red-400 border border-red-500/30 rounded-lg hover:bg-red-500/10 transition-colors">
            Delete tenant data
          </button>
        </div>
      </div>
    </div>
  );
}
