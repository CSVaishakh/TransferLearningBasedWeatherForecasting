# Frontend: SJCET weather app

FastAPI backend (`api/`) + React/Vite/TypeScript app (`web/`), following
[`docs/frontend-requirements.md`](../docs/frontend-requirements.md).

```bash
cd frontend/api && uv sync && uv run uvicorn main:app --port 8000     # API on :8000
cd frontend/web && npm install && npm run dev                           # app on http://localhost:5173 (proxies /api)
```

The API serves files produced by the other tracks and computes nothing itself:

| Endpoint | Source |
|---|---|
| `/api/forecast?date=&model=`, `/api/forecast/range?from=&to=`, `/api/forecast/dates`, `/api/models` | `model/outputs/forecasts/` (`predict.py --export [--run <id>]`) |
| `/api/metrics`, `/api/graph` | `model/outputs/metrics.json`, `graph.json` (`export.py`) |
| `/api/results/test?model=` | `model/outputs/runs/<run>/local_test_pred.npz` |
| `/api/history?from=&to=&vars=` | `data/sjcet/sjcet_daily.csv` (`digitization/assemble.py`) |
| `/api/review/status` | `data/sjcet/cells.csv` (not produced yet) |

## Pages
- **Forecast:** replay forecasts for the test period (2015–2016), compared with the register.
  The model forecasts **one day ahead**, so the three cards are the next-day forecasts issued
  on the selected date and the two days after it.
- **History:** coverage heatmap, daily readings (suspect values hollow), monthly and annual rainfall, annual temperature means, CSV download.
- **Results:** error vs amount of SJCET data, model table, forecast vs observed, error by season.
- **Review Tool:** shows an empty state until OCR'd cells (`cells.csv`) exist. The 2010–2016 register was transcribed by hand.
- **About:** pipeline and data sources. Team names are not filled in yet (`web/src/pages/About.tsx`).
