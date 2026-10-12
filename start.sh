#!/usr/bin/env bash
# Start the SJCET weather app: FastAPI backend (port 8000) + web frontend (port 5173).
#
#   ./start.sh            start the app (installs what's missing on first run)
#   ./start.sh --refresh  first fetch the latest NASA POWER data and remake today's live forecast (~1 min, needs internet)
#
# Then open http://localhost:5173. Press Ctrl+C in this terminal to stop everything.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
API_PORT=8000
WEB_PORT=5173
LOG="$ROOT/.start-logs"
mkdir -p "$LOG"

say()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31mxx\033[0m %s\n' "$*"; exit 1; }
up()   { curl -s -m 2 -o /dev/null "$1"; }

command -v uv  >/dev/null || die "uv is not installed: curl -LsSf https://astral.sh/uv/install.sh | sh"
command -v npm >/dev/null || die "Node.js/npm is not installed"
unset VIRTUAL_ENV  # each part uses its own .venv

# --- what the app shows: produced by the digitization and model steps (not regenerated here)
[ -f "$ROOT/data/sjcet/sjcet_daily.csv" ]           || warn "data/sjcet/sjcet_daily.csv missing: History page will be empty (run digitization/assemble.py)"
[ -f "$ROOT/model/outputs/metrics.json" ]           || warn "model/outputs/metrics.json missing: Results page will be empty (run model/run_all.sh)"
[ -f "$ROOT/model/outputs/forecasts/index.json" ]   || warn "replay forecasts missing (run: cd model && uv run python predict.py --export)"

# --- optional: today's live forecast
if [ "${1:-}" = "--refresh" ]; then
  say "Fetching the latest data and making today's live forecast (about a minute)…"
  (cd "$ROOT/model" && uv sync -q && uv run python -W ignore live.py) || warn "live forecast failed (no internet?); the app shows the last one"
fi
[ -f "$ROOT/model/outputs/forecasts/live.json" ] || warn "no live forecast yet: run ./start.sh --refresh (or use Replay mode)"

# --- backend
if up "http://localhost:$API_PORT/api/health"; then
  say "API already running on port $API_PORT"
else
  say "Starting API on port $API_PORT…"
  (cd "$ROOT/frontend/api" && uv sync -q && exec uv run uvicorn main:app --port "$API_PORT") > "$LOG/api.log" 2>&1 &
  API_PID=$!
  for _ in $(seq 1 60); do up "http://localhost:$API_PORT/api/health" && break; sleep 1; done
  up "http://localhost:$API_PORT/api/health" || die "API did not start, see $LOG/api.log"
fi

# --- frontend
if up "http://localhost:$WEB_PORT/"; then
  say "Web app already running on port $WEB_PORT"
else
  [ -d "$ROOT/frontend/web/node_modules" ] || { say "Installing web dependencies (first run only)…"; (cd "$ROOT/frontend/web" && npm install --silent); }
  say "Starting web app on port $WEB_PORT…"
  (cd "$ROOT/frontend/web" && exec npx vite --port "$WEB_PORT" --strictPort) > "$LOG/web.log" 2>&1 &
  WEB_PID=$!
  for _ in $(seq 1 60); do up "http://localhost:$WEB_PORT/" && break; sleep 1; done
  up "http://localhost:$WEB_PORT/" || die "web app did not start, see $LOG/web.log"
fi

URL="http://localhost:$WEB_PORT"
say "Ready: $URL   (Forecast · History · Results · About)"
# open the browser: WSL -> Windows browser, else the desktop default
if command -v wslview >/dev/null; then wslview "$URL" >/dev/null 2>&1 || true
elif command -v cmd.exe >/dev/null; then cmd.exe /c start "$URL" >/dev/null 2>&1 || true
elif command -v xdg-open >/dev/null; then xdg-open "$URL" >/dev/null 2>&1 || true
fi

# keep running until Ctrl+C, then stop what this script started
cleanup() { say "Stopping…"; kill ${API_PID:-} ${WEB_PID:-} 2>/dev/null || true; }
trap cleanup INT TERM EXIT
if [ -n "${API_PID:-}${WEB_PID:-}" ]; then
  say "Press Ctrl+C to stop."
  wait
fi
