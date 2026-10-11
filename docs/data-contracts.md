# Data Contracts

The three tracks (Digitization, Model, Frontend) work in parallel. They only
touch each other through the files and API below. **Change a contract only by
agreement of all three tracks**, and bump the `version` line when you do.

`version: 1 (2026-10-10)`

```
Digitization ──cells.csv / sjcet_daily.csv──► Model ──forecast / metrics JSON──► Frontend
      ▲                                                                            │
      └──────────────────── corrections.csv (review tool) ◄────────────────────────┘
```

---

## 1. `data/sjcet/sjcet_daily.csv` — the digitized register (Digitization → Model, Frontend)

One row per calendar day, **including days with no reading** (all value columns empty).

| column | type | unit | notes |
|---|---|---|---|
| `date` | `YYYY-MM-DD` | – | the date written in the register |
| `obs_time` | `HH:MM` | IST | time of reading, empty if not written |
| `rain_mm` | float | mm | **register is in cm → multiply by 10**. `0` = "Nil"/"–" written; empty = not recorded |
| `dry_bulb_c` | float | °C | |
| `wet_bulb_c` | float | °C | |
| `tmin_c` | float | °C | |
| `tmax_c` | float | °C | |
| `rh_pct` | float | % | derived from dry/wet bulb (see digitization guide §11), never OCR'd |
| `source_page` | string | – | `page_id` from `pages.csv` |
| `qc_flag` | enum | – | `ok` · `corrected` (human fixed) · `suspect` (kept but doubtful) |
| `qc_note` | string | – | free text, optional |

Rules:
- Missing ≠ zero. An empty cell means "not observed"; `0` means "observed, no rain".
- 8:30 AM register convention: `rain_mm` on date D is rain from D-1 08:30 to D 08:30;
  `tmax_c` on date D is normally the max of day D-1. The model track handles the
  shift; the CSV stores values **exactly against the date written**.

## 2. `data/sjcet/cells.csv` — every OCR'd cell (Digitization → Frontend review tool)

| column | example | notes |
|---|---|---|
| `cell_id` | `p0123_r05_c3` | unique, stable |
| `page_id` | `p0123` | |
| `page_path` | `digitization/work/pages/p0123.png` | relative to repo root |
| `crop_path` | `digitization/work/cells/p0123_r05_c3.png` | |
| `bbox` | `412,980,530,1032` | `x0,y0,x1,y1` in page pixels |
| `date` | `2012-07-05` | from row → day mapping |
| `field` | `tmax_c` | one of the value columns above, or `date` |
| `ocr_text` | `29.5` | raw model output |
| `ocr_conf` | `0.83` | 0–1 |
| `value` | `29.5` | parsed number (empty if unparseable) |
| `qc_flag` | `suspect` | `ok` / `suspect` / `fail` |
| `qc_reason` | `tmax<tmin` | short machine reason |

## 3. `data/sjcet/corrections.csv` — human review results (Review tool → Digitization)

Append-only. The latest row per `cell_id` wins.

| column | notes |
|---|---|
| `cell_id` | |
| `value` | corrected text as it should be read, e.g. `29.5` |
| `status` | `accepted` (OCR was right) · `corrected` · `illegible` · `blank` |
| `reviewer` | initials |
| `timestamp` | ISO 8601 |

## 4. Forecast JSON (Model → Frontend)

Served at `GET /api/forecast?date=YYYY-MM-DD`, also written to
`model/outputs/forecasts/<date>.json`. `issued` is the last day of input data.

```json
{
  "version": 1,
  "issued": "2015-07-10",
  "station": {"id": "SJCET", "name": "SJCET Palai (Choondacherry)", "lat": 9.727, "lon": 76.727},
  "model": {"name": "gnn-stage3", "run_id": "2026-11-02_a"},
  "days": [
    {"date": "2015-07-11", "tmax_c": 29.1, "tmin_c": 22.8, "tmean_c": 26.0, "dewpoint_c": 23.4,
     "rh_pct": 86, "rain_mm": 12.4, "rain_prob": 0.71, "rain_expected": true,
     "observed": {"tmax_c": 28.6, "tmin_c": 23.0, "tmean_c": 25.7, "dewpoint_c": 23.1, "rh_pct": 87, "rain_mm": 9.0}}
  ]
}
```
`observed` is present only when the date is in the past (replay mode) and data exist; any
value may be `null`. `rh_pct` (relative humidity) is calculated from `tmean_c` and `dewpoint_c`.

## 5. Metrics JSON (Model → Frontend)

`model/outputs/metrics.json`:

```json
{
  "version": 1,
  "test_period": ["2015-01-01", "2016-12-31"],
  "models": [
    {"name": "persistence", "group": "baseline",
     "metrics": {"tmax_c": {"rmse": 1.9, "mae": 1.4, "bias": 0.0, "r2": 0.31},
                 "tmin_c": {"rmse": 1.2, "mae": 0.9, "bias": 0.0, "r2": 0.40},
                 "rain":   {"rmse": 14.2, "mae": 7.1, "hit_rate": 0.62, "false_alarm": 0.35},
                 "rh_pct": {"rmse": 3.8, "mae": 2.9, "bias": 0.1, "r2": 0.5}}}
  ],
  "ablation": [
    {"model": "gnn-1-2-3", "local_fraction": 0.25, "seed": 0, "tmax_rmse": 1.3, "tmin_rmse": 0.9, "rain_rmse": 12.1}
  ],
  "by_season": [
    {"model": "gnn-1-2-3", "season": "SW monsoon (Jun–Sep)", "tmax_rmse": 1.1}
  ]
}
```

Seasons (IMD): Winter Jan–Feb · Pre-monsoon Mar–May · SW monsoon Jun–Sep · NE monsoon Oct–Dec.

## 6. Graph JSON (Model → Frontend map)

`model/outputs/graph.json`:

```json
{"nodes": [{"id": 0, "kind": "era5", "lat": 9.75, "lon": 76.75},
           {"id": 401, "kind": "gsod", "lat": 8.48, "lon": 76.95, "name": "THIRUVANANTHAPURAM"},
           {"id": 402, "kind": "sjcet", "lat": 9.727, "lon": 76.727}],
 "edges": [[0, 402], [401, 0]]}
```
`kind` ∈ `era5` · `gsod` · `sjcet`.
