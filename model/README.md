# Model: staged transfer-learning GNN

Working implementation of `docs/model-guide.md`, with the changes below made to hit the
deadline.

## What differs from the guide (and why)

| Guide | Implemented | Why |
|---|---|---|
| ERA5 from Copernicus CDS | **NASA POWER** daily point data (MERRA-2 based reanalysis) | CDS requests sat in the queue for over an hour without starting; POWER returns 36 years per point in ~4 s, no key. `data/download_era5.py` still works if you want ERA5 later. |
| Kerala fine grid 0.25° | 0.5° (POWER's native resolution) | |
| SJCET node as stage-3 target | **COCHIN NOAA station (43353099999)** as a stand-in | SJCET register is not digitized yet. Cochin is the nearest real station (~50 km) and is hidden in stages 1–2, so it plays exactly SJCET's role: a data-poor local site. Swap in SJCET by changing `local.node_id` in `config.yaml` once `sjcet_daily.csv` exists. |
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
