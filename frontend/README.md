# Frontend: SJCET weather app

The web app for the project: a **FastAPI** backend in [`api/`](api/) and a **React + Vite +
TypeScript** app in [`web/`](web/). It shows the live forecast, replays past forecasts, browses the
digitized register and compares the models.

The API computes nothing itself. It serves files produced by the digitization and model parts
of the repo. The one exception is the live-forecast refresh, which runs `model/live.py`.

## Starting it

The easiest way is the script in the repository root, which starts both servers and opens the browser:

```bash
./start.sh            # from the repo root
./start.sh --refresh  # also remake today's live forecast first (~1 min, needs internet)
```

To run the two servers yourself, use two terminals:

```bash
# API on http://localhost:8000 (interactive docs at http://localhost:8000/docs)
cd frontend/api
uv sync
uv run uvicorn main:app --reload --port 8000

# Web app on http://localhost:5173
cd frontend/web
npm install
npm run dev
```

The Vite dev server forwards `/api/*` to port 8000 ([vite.config.ts](web/vite.config.ts)), and
the API allows CORS from `http://localhost:5173`. Keep these ports or change both places.

Requirements: [uv](https://docs.astral.sh/uv/) (Python 3.11/3.12) and Node.js 18+.

### Other commands

```bash
cd frontend/web
npm run build     # type-check and build to dist/
npm run preview   # serve the built app
npm run lint      # eslint
```

To serve a different set of model outputs (for example the archived next-day results), set
`SJCET_MODEL_OUT` before starting the API:

```bash
SJCET_MODEL_OUT=../../model/outputs_h1 uv run uvicorn main:app --port 8000
```

## What the app needs on disk

| File(s) | Made by | Without it |
|---|---|---|
| `data/sjcet/sjcet_daily.csv` | `digitization/assemble.py` | History page empty |
| `model/outputs/metrics.json`, `graph.json` | `model/export.py` | Results page and network map empty |
| `model/outputs/forecasts/*.json` | `model/predict.py --export [--run <id>]` | Replay mode empty |
| `model/outputs/forecasts/live.json` | `model/live.py` | Live mode shows "Make one now" |
| `model/outputs/runs/<run>/local_test_pred.npz` | `model/train.py` | Forecast-vs-observed charts empty |
| `data/sjcet/cells.csv` | not produced yet | Review Tool shows an empty state |

All paths are set in [api/paths.py](api/paths.py).

## API endpoints

| Endpoint | Returns | Source |
|---|---|---|
| `GET /api/health` | `{"ok": true}` | – |
| `GET /api/models` | models available for replay | `model/outputs/forecasts/` |
| `GET /api/forecast/dates?model=` | issue dates that have a forecast | `model/outputs/forecasts/` |
| `GET /api/forecast?date=&model=` | one 6-day forecast | `model/outputs/forecasts/<date>.json` |
| `GET /api/forecast/range?from=&to=&model=` | forecasts issued in a date range | same |
| `GET /api/forecast/live` | the latest live forecast | `model/outputs/forecasts/live.json` |
| `POST /api/forecast/live/refresh` | reruns `live.py` and returns the new forecast (~1–2 min) | NASA POWER + NOAA GSOD |
| `GET /api/metrics` | scores per model, lead day, ablation and season | `model/outputs/metrics.json` |
| `GET /api/graph` | graph nodes and edges for the map | `model/outputs/graph.json` |
| `GET /api/results/test?model=&lead=` | test-period predictions vs observations | `model/outputs/runs/<run>/local_test_pred.npz` |
| `GET /api/history?from=&to=&vars=` | daily register rows | `data/sjcet/sjcet_daily.csv` |
| `GET /api/review/status` | OCR review progress | `data/sjcet/cells.csv` |

Routes live in [api/routes/](api/routes/).

## Pages

| Page | Source file | What it shows |
|---|---|---|
| Forecast | [Forecast.tsx](web/src/pages/Forecast.tsx) | **Live:** today + next two days from the latest data, with a refresh button. **Replay:** pick an issue date in 2015–2016 and a model, and compare the 6-day forecast with the register. Network map in both modes. |
| History | [History.tsx](web/src/pages/History.tsx) | Coverage heatmap, daily readings (suspect values hollow) with a year range and variable toggles, monthly/annual rainfall, annual temperature means, CSV download. |
| Results | [Results.tsx](web/src/pages/Results.tsx) | Error vs lead time, live-like test (SJCET inputs hidden), error vs amount of SJCET data, model table, forecast vs observed, error by season. Variable and lead-day pickers at the top. |
| Review Tool | [Review.tsx](web/src/pages/Review.tsx) | Empty state until OCR'd cells exist. The 2010–2016 register was transcribed by hand. |
| About | [About.tsx](web/src/pages/About.tsx) | Pipeline and data sources. Team names are not filled in yet. |

Step-by-step instructions for using each page are in the [root README](../README.md#using-the-app).

## Code layout

```
web/src/
├── App.tsx            # sidebar, routes, light/dark theme toggle
├── api.ts             # typed fetch helpers for every endpoint
├── components/        # DayCard, ForecastChart, NodeMap (Leaflet), loading/empty/error states
├── pages/             # one file per page
└── styles.css         # all styles; colour tokens with light and dark themes
```

Charts use Recharts, the map uses Leaflet via react-leaflet, and the icons are from lucide-react.
