// apps/web/src/app/api/predict/route.ts
// Server-side proxy for predictions — keeps API URL server-only

import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.INTERNAL_API_URL ?? process.env.NEXT_PUBLIC_API_URL ?? "http://api:8000";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const tenantId = request.headers.get("X-Tenant-Id") ?? "default";

    const res = await fetch(`${API_URL}/api/v1/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Tenant-Id": tenantId,
      },
      body: JSON.stringify(body),
    });

    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (err) {
    console.error("[API Proxy Error]", err);
    return NextResponse.json({ error: "Upstream API unavailable" }, { status: 502 });
  }
}
