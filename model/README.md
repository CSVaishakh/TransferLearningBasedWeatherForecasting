# Model: staged transfer-learning GNN

Working implementation of `docs/model-guide.md`, with the changes below made to hit the
deadline.

## What differs from the guide (and why)

| Guide | Implemented | Why |
|---|---|---|
| ERA5 from Copernicus CDS | **NASA POWER** daily point data (MERRA-2 based reanalysis) | CDS requests sat in the queue for over an hour without starting; POWER returns 36 years per point in ~4 s, no key. `data/download_era5.py` still works if you want ERA5 later. |
| Kerala fine grid 0.25° | 0.5° (POWER's native resolution) | |
| SJCET node as stage-3 target | **SJCET** (digitized register, Jul 2010–Dec 2016) since 2026-10-11 | Earlier runs used the COCHIN NOAA station (43353099999) as a stand-in; set `local.node_id` back to it to reproduce them. |
| H = 1–3 days | H = 1 (next day) | |

## Graph
1,062 reanalysis nodes (962 at 1° over India, 99 at 0.5° over Kerala, 1 at SJCET) + 6 NOAA
stations (Bangalore, Coimbatore, Mangalore, Thiruvananthapuram, Kozhikode, Cochin), k = 8
nearest-neighbour edges (symmetric) with offset/distance/elevation-difference features.

Inputs per node: last 7 days of 9 variables (tmax, tmin, tmean, dewpoint, rain, surface
pressure, u10, v10, cloud) + masks + lat/lon/elevation/node type + day of year.
Targets: next-day tmax, tmin, rain (log1p), tmean, dewpoint and P(rain ≥ 2.5 mm).
**Relative humidity is not predicted directly**: it is calculated from the forecast tmean and
dewpoint with the Magnus formula (`wx.rh_from`). Checkpoints from before this change
(tmax/tmin/rain only) still load: `wx.load_weights` copies their outputs and the new
tmean/dewpoint outputs are learned during the next fine-tuning stage.

Model: MLP encoder → 4 × (GATv2 message passing + feed-forward, residual) → MLP decoder;
~0.4 M parameters; trains on CPU.

## Stages
| Stage | Visible data | Loss on |
|---|---|---|
| 1 | reanalysis nodes | all reanalysis nodes |
| 2 | + NOAA stations except Cochin | Kerala 0.5° nodes (1), stations (2), India 1° (0.1) |
| 3 | + Cochin (a fraction of its training months) | Cochin only, on its 4-hop subgraph |

Splits: train 1990–2013, val 2014, test 2015–2016 (identical in every stage).

## SJCET (stage-3 target)

`data/build_dataset.py` adds `data/sjcet/sjcet_daily.csv` as one station node:
- **08:30 convention:** register rain and Tmax on date D are moved to D-1; Tmin stays on D.
- **Dewpoint:** from the 08:30 dry/wet bulb. tmean = (Tmax + Tmin) / 2.
- **Dropped:** `suspect` values, and rain read after an unread day (weekend gauge accumulation).

That leaves train 2010-07–2013 (716 temperature / 249 rain days), val 2014, test 2015–2016.

Results, SJCET test 2015–2016, all training months (RMSE; `outputs/metrics.json`):

| Model | Tmax °C | Tmin °C | Rain mm | Rain/no-rain acc. |
|---|---|---|---|---|
| GNN from scratch (SJCET only) | 2.09 | 2.82 | 21.5 | 0.68 |
| GNN pretrained 1→3 | **1.51** | **1.09** | 20.6 | 0.63 |
| GNN pretrained 1→2→3 | 1.54 | 1.23 | **20.6** | 0.67 |
| XGBoost (station + nearest grid) | 1.44 | 1.07 | 21.5 | 0.52 |
| Climatology | 1.92 | 1.24 | 21.4 | 0.51 |

- **Pretraining is what makes SJCET learnable:** with 10% of its months, pretrained models
  reach Tmax 2.0 / Tmin 1.8–2.0 °C, against 3.0 / 2.5 °C from scratch.
- **Persistence is hard to beat on temperature.** On the same days it scores
  Tmax 1.12 vs 1.46 °C and Tmin 0.90 vs 1.08 °C. The register holds whole degrees and the
  max thermometer often repeats for weeks, so "tomorrow = today" is very strong here.
- **RH is not meaningful at SJCET:** it is derived from the tmean output, and the register
  has no true daily mean, only (Tmax + Tmin) / 2 from fewer days. The tmean output is
  biased about -2.3 °C.

`./run_stage3.sh` reruns baselines + the stage-3 ablation + export from the existing
`s1`/`s12` checkpoints (about 2 h on CPU).

## Run

```bash
cd model
uv sync
uv run python data/download_power.py      # ~1 h, resumable
./run_all.sh                              # build data, baselines, stages 1-2, 30 stage-3 runs, export
uv run python predict.py 2015-07-10       # one forecast (JSON)
uv run python predict.py --export         # test-period forecasts -> outputs/forecasts/
```

## Outputs
- `outputs/metrics.json`: models, ablation and per-season results (data-contracts §5)
- `outputs/graph.json`: nodes/edges for the frontend map (§6)
- `outputs/forecasts/<date>.json`: forecasts (§4)
- `outputs/figures/`: `ablation.png`, `test_timeseries.png`, `graph_map.png`
- `outputs/runs/<run>/`: checkpoint, training log, result.json
