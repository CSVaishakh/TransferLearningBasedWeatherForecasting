# Model: staged transfer-learning GNN

Forecasts daily weather at SJCET (Palai, Kerala) **6 days ahead** with a graph neural network
trained in three stages: first on regional reanalysis, then on NOAA stations, and finally on the
short digitized SJCET register. The question it answers: does pretraining on data-rich regional
sources help forecast a data-poor station?

## Quick start

Requirements: [uv](https://docs.astral.sh/uv/) and Python 3.11/3.12. Everything runs on CPU
(PyTorch CPU build, PyTorch Geometric).

```bash
cd model
uv sync

# Data
uv run python data/download_power.py      # NASA POWER for every grid node, ~1 h, resumable -> raw/power/
uv run python data/build_dataset.py       # stack grid + stations + SJCET into processed/dataset.npz

# Training and evaluation (the full experiment, several hours on CPU, resumable)
./run_all.sh                              # SEEDS="0" ./run_all.sh for a quicker single-seed run

# Forecasts
uv run python predict.py 2015-07-10       # print one forecast (JSON), issued on that date
uv run python predict.py --export         # all test-period forecasts -> outputs/forecasts/
uv run python live.py                     # forecast from the latest data -> outputs/forecasts/live.json
```

`run_all.sh` builds the dataset if missing, so `build_dataset.py` only needs running by hand
after changing the data. The web app ([../frontend](../frontend/README.md)) reads the files in
`outputs/`; start it with `./start.sh` from the repo root.

### Input data

| Path | Contents | How to get it |
|---|---|---|
| `raw/power/` | NASA POWER daily point data, 1990–2025 | `data/download_power.py` |
| `raw/gsod/kerala/<year>/<id>.csv`, `kerala_stations.csv`, `isd-history.csv` | NOAA GSOD files for the 6 stations | Copied from the NOAA GSOD archive (`../Dataset/`) |
| `../data/sjcet/sjcet_daily.csv` | The digitized SJCET register | `../digitization/assemble.py` |

None of these are in git.

## Files

| File | Purpose |
|---|---|
| [config.yaml](config.yaml) | Regions, grid steps, variables, train/val/test splits, `horizon`, the local target node |
| [wx.py](wx.py) | Shared code: dataset loading, the `WeatherGNN` model, loss, metrics, RH formula |
| [data/download_power.py](data/download_power.py) | Downloads NASA POWER for every grid node |
| [data/download_era5.py](data/download_era5.py) | ERA5 download from Copernicus CDS (not used, see below) |
| [data/build_dataset.py](data/build_dataset.py) | Builds the node array and k-NN graph -> `processed/dataset.npz` |
| [train.py](train.py) | Trains one stage (`--stage 1/2/3`, `--init`, `--fraction`, `--seed`, `--run-id`) |
| [baselines.py](baselines.py) | Persistence, climatology, raw reanalysis, XGBoost |
| [export.py](export.py) | Collects all runs into `outputs/metrics.json`, `graph.json` and figures |
| [predict.py](predict.py) | Replay forecasts for past issue dates |
| [live.py](live.py) | Live forecast from the newest NASA POWER and GSOD data |
| [run_all.sh](run_all.sh) | Full experiment: data, baselines, stages 1–2, stage-3 ablation, export |
| [run_stage3.sh](run_stage3.sh) | Reruns baselines + stage-3 ablation + export from existing `s1`/`s12` checkpoints (~2 h) |
| [run_climfill.sh](run_climfill.sh) | Stage-3 runs with SJCET climatology fed in place of missing readings |

## Changes from the original plan

| Plan | Implemented | Why |
|---|---|---|
| ERA5 from Copernicus CDS | **NASA POWER** daily point data (MERRA-2 based reanalysis) | CDS requests sat in the queue for over an hour without starting. POWER returns 36 years per point in ~4 s with no API key. `data/download_era5.py` still works if ERA5 is needed later. |
| Kerala fine grid at 0.25° | 0.5° | POWER's native resolution |
| SJCET as the stage-3 target | **SJCET** (digitized register, Jul 2010 – Dec 2016) | Earlier runs used the COCHIN NOAA station (`43353099999`) as a stand-in. Set `local.node_id` back to it in `config.yaml` to reproduce them. |
| Forecast 1–3 days ahead | **6 days** (`horizon` in `config.yaml`) | NASA POWER publishes ~3 days late, so 6 lead days are needed for a live forecast to reach today plus the next two days. |

## Graph

1,069 nodes:
- 1,062 reanalysis nodes: 962 at 1° over India, 99 at 0.5° over Kerala and 1 at SJCET's location
- 6 NOAA GSOD stations: Bangalore, Coimbatore, Mangalore, Thiruvananthapuram, Kozhikode, Cochin
- 1 SJCET node

Each node connects to its k = 8 nearest neighbours (edges made symmetric). Edge features are the
lat/lon offset, distance and elevation difference.

**Inputs per node:** the last 7 days of 9 variables (tmax, tmin, tmean, dewpoint, rain, surface
pressure, u10, v10, cloud), their missing-value masks, lat/lon/elevation, node type and day of year.

**Outputs per node and lead day (t+1 … t+6):** tmax, tmin, rain (log1p), tmean, dewpoint and
the probability of rain ≥ 2.5 mm. **Relative humidity is not predicted directly.** It is
calculated from the forecast tmean and dewpoint with the Magnus formula (`wx.rh_from`).

**Model:** MLP encoder → 4 × (GATv2 message passing + feed-forward, residual) → MLP decoder.
About 0.4 M parameters, trains on CPU. The decoder emits one block of outputs per lead day from
the same 7-day input window, and the loss averages over lead days.

## Training stages

| Stage | Data the model sees | Loss on |
|---|---|---|
| 1 | reanalysis nodes only | all reanalysis nodes |
| 2 | + NOAA stations | Kerala 0.5° nodes (weight 1), stations (2), India 1° nodes (0.1) |
| 3 | + SJCET (a fraction of its training months) | SJCET only, on its 4-hop subgraph |

SJCET is hidden in stages 1 and 2. Splits are the same in every stage:
**train 1990–2013, validation 2014, test 2015–2016**.

Stage 3 is run as an ablation: three strategies (from scratch, 1→3, 1→2→3) × four fractions of
SJCET's training months (10%, 25%, 50%, 100%) × three seeds (one seed at 100%). Run IDs follow
`<strategy>_f<fraction>_s<seed>`, e.g. `s123_f0.25_s0`. Checkpoints from earlier configurations
(next-day only, or without tmean/dewpoint outputs) still load: `wx.load_weights` copies the old
outputs into the new ones and the rest are learned during the next stage.

## SJCET data handling

`data/build_dataset.py` adds `data/sjcet/sjcet_daily.csv` as one station node:
- **08:30 reading convention:** register rain and Tmax on date D are moved to D-1; Tmin stays on D.
- **Dewpoint** is computed from the 08:30 dry/wet bulb; tmean = (Tmax + Tmin) / 2.
- **Dropped:** values flagged `suspect`, and rain read after an unread day (weekend gauge accumulation).

That leaves training data for July 2010 – 2013 (716 temperature days, 249 rain days),
validation 2014 and test 2015–2016.

## Results

SJCET test period 2015–2016, all training months, RMSE (from `outputs/metrics.json`):

| Model | Tmax °C day 1 | Tmin °C day 1 | Rain mm day 1 | Rain/no-rain acc. day 1 | Tmax °C day 6 | Tmin °C day 6 |
|---|---|---|---|---|---|---|
| GNN from scratch (SJCET only) | 2.17 | 1.72 | 20.8 | 0.63 | 1.85 | 1.43 |
| GNN pretrained 1→3 | **1.42** | 1.10 | 20.9 | 0.63 | **1.55** | 1.25 |
| GNN pretrained 1→2→3 | 1.43 | 1.08 | **20.8** | 0.64 | 1.58 | **1.21** |
| XGBoost (station + nearest grid) | 1.48 | **1.04** | 21.4 | 0.54 | 1.67 | 1.26 |
| Climatology | 1.92 | 1.24 | 21.4 | 0.51 | 1.89 | 1.23 |
| Persistence ("same as today") | 1.12 | 0.90 | 30.1 | 0.75 | 1.71 | 1.33 |

- **Pretraining makes SJCET learnable from little data.** With 10% of its training months
  (day 1, mean of 3 seeds), pretrained models reach Tmax 2.0–2.1 / Tmin 2.1–2.5 °C, against
  2.6 / 3.7 °C from scratch.
- **Persistence is hard to beat on temperature at day 1.** The register holds whole degrees and
  the max thermometer often repeats for weeks, so "tomorrow = today" is very strong. By day 6
  the pretrained GNNs are ahead of it. Persistence can only be scored on days that follow a
  reading, so it covers fewer days.
- **RH is not meaningful at SJCET.** It is derived from the tmean output, but the register has
  no true daily mean, only (Tmax + Tmin) / 2 from fewer days. The tmean output is biased by
  about −2.3 °C.

The archived next-day (H = 1) results are in `outputs_h1/`.

## Live forecasts

`live.py` makes a forecast from the latest data:
1. Finds the reanalysis nodes in SJCET's 4-hop neighbourhood (the only nodes that affect its output).
2. Fetches recent NASA POWER data for them and the latest NOAA GSOD files for the stations.
3. Issues the forecast from the last complete day, so today is usually lead day 4.
4. Writes `outputs/forecasts/live.json`, including the last 14 days of reanalysis at SJCET.

The SJCET register has no readings after 2016, so SJCET's own inputs are empty and the live
forecast relies on regional data. The model saw such gaps on every weekend and holiday during
training. The Results page's "live-like test" scores the models under the same conditions.

Options: `--run <run-id>` (default `s123_f1.0_s0`), `--workers N` (parallel downloads).
The web app's **Refresh with latest data** button runs this script.

## Outputs

| Path | Contents |
|---|---|
| `outputs/metrics.json` | All models: overall, per lead day, ablation, per season |
| `outputs/ablation.csv` | Every stage-3 run × fraction × seed × lead day |
| `outputs/baselines.json` | Baseline scores |
| `outputs/graph.json` | Nodes and edges for the frontend map |
| `outputs/forecasts/<date>.json` | Replay forecasts for the default model; other models in `outputs/forecasts/<run>/` |
| `outputs/forecasts/live.json` | Latest live forecast |
| `outputs/figures/` | `ablation.png`, `test_timeseries.png`, `graph_map.png` |
| `outputs/runs/<run>/` | `best.pt` checkpoint, `log.csv` training log, `result.json`, `local_test_pred.npz` |
