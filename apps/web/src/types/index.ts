// apps/web/src/types/index.ts

export interface PredictionResult {
  sentiment: "positive" | "neutral" | "negative";
  sentiment_score: number;
  confidence: number;
  probabilities: {
    positive: number;
    neutral: number;
    negative: number;
  };
  urgency: "critical" | "high" | "medium" | "low";
  urgency_score: number;
  severity_score: number;
  business_impact_score: number;
  attributes: string[];
  attribute_scores: Record<string, number>;
  explanation: {
    sentiment_label: string;
    top_tokens: Array<{ token: string; importance: number }>;
    method: string;
  };
  model_version: string;
  latency_ms: number;
}

export interface KPIs {
  total_messages: number;
  positive_count: number;
  neutral_count: number;
  negative_count: number;
  positive_rate: number;
  negative_rate: number;
  critical_urgency_count: number;
  avg_business_impact: number;
  avg_confidence: number;
}

export interface OverviewData {
  period_days: number;
  kpis: KPIs;
  trend: "improving" | "worsening" | "stable";
  sentiment_health_score: number;
}

export interface TimeSeriesEntry {
  date: string;
  total: number;
  positive: number;
  neutral: number;
  negative: number;
  positive_rate: number;
  negative_rate: number;
  avg_sentiment_score: number;
  avg_confidence: number;
  critical_count: number;
  avg_business_impact: number;
}

export interface ClusterEntry {
  cluster_id: number;
  theme: string;
  dominant_sentiment: "positive" | "neutral" | "negative";
  message_count: number;
  avg_severity: number;
  avg_business_impact: number;
  top_terms: string[];
}

export interface DriftRecord {
  week_start: string;
  psi_score: number;
  ks_statistic: number;
  drift_detected: boolean;
  alert: boolean;
}

export interface HeatmapEntry {
  attribute: string;
  sentiment: "positive" | "neutral" | "negative";
  count: number;
  avg_score: number;
}
