// apps/web/src/lib/api.ts
import axios from "axios";

// Always use relative URL — Next.js rewrites proxy it to http://api:8000
// This works in localhost, GitHub Codespaces, and production identically.
export const api = axios.create({
  baseURL: "/api/v1",
  timeout: 30_000,
  headers: { "Content-Type": "application/json" },
});

api.interceptors.request.use((config) => {
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("access_token");
    if (token) config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (r) => r,
  (err) => {
    console.error("[API Error]", err.response?.status, err.response?.data?.detail);
    return Promise.reject(err);
  }
);
