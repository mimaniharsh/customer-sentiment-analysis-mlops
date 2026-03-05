#!/bin/bash
# ============================================================
# Sentient Platform — Codespaces / DevContainer Bootstrap
# Runs automatically after the container is created.
# ============================================================
set -euo pipefail

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Sentient Intelligence Platform — Dev Setup     ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── Python dependencies ──────────────────────────────────────────────────────
echo "📦 Installing Python dependencies..."
cd /workspaces/sentient-platform/apps/api
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
pip install --quiet ruff mypy pytest pytest-asyncio pytest-cov httpx
echo "   ✅ Python deps installed"

# ── Node dependencies ─────────────────────────────────────────────────────────
echo ""
echo "📦 Installing Node dependencies..."
cd /workspaces/sentient-platform/apps/web
npm install --silent
echo "   ✅ Node deps installed"

# ── Create .env if not present ───────────────────────────────────────────────
echo ""
cd /workspaces/sentient-platform
if [ ! -f .env ]; then
  cp .env.example .env
  echo "   ⚠️  Created .env from .env.example"
  echo "   ➡️  Open .env and fill in your AWS credentials before starting."
else
  echo "   ✅ .env already exists"
fi

# ── Generate secret keys if placeholders still present ───────────────────────
if grep -q "<generate-with" .env 2>/dev/null; then
  APP_KEY=$(openssl rand -hex 32)
  JWT_KEY=$(openssl rand -hex 32)
  sed -i "s|APP_SECRET_KEY=<generate-with: openssl rand -hex 32>|APP_SECRET_KEY=${APP_KEY}|" .env
  sed -i "s|JWT_SECRET_KEY=<generate-with: openssl rand -hex 32>|JWT_SECRET_KEY=${JWT_KEY}|" .env
  echo "   ✅ Generated APP_SECRET_KEY and JWT_SECRET_KEY"
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║   Setup complete! Next steps:                    ║"
echo "║                                                  ║"
echo "║   1. Edit .env — add your AWS credentials        ║"
echo "║   2. docker compose up --build                   ║"
echo "║   3. Open http://localhost:3000                  ║"
echo "╚══════════════════════════════════════════════════╝"
