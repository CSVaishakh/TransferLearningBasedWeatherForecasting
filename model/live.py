"""Live forecast for SJCET from the most recent data available (not replay).

    uv run python live.py [--run s123_f1.0_s0] [--workers 6]

Only the local node's `layers`-hop neighbourhood affects its forecast, so we fetch
recent NASA POWER data for those grid nodes and the latest NOAA GSOD files for those
stations, build the 7-day input window ending on the last day POWER has complete,
and run the fine-tuned model. The SJCET register has no readings after 2016, so the
SJCET node's own inputs are empty (the model saw such gaps on every weekend/holiday).

Writes outputs/forecasts/live.json (data-contracts §4 plus `mode`, `data_through`
and `recent`: the last 14 days of reanalysis at SJCET for context).
"""
import argparse
import io
import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import k_hop_subgraph

from train import predict as run_model
from train import stage_setup
from wx import CFG, H, K, NREG, OUTS, RAIN, RAIN_THR, Data, WeatherGNN, load_weights, rh_from

POWER_URL = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters={params}"
             "&community=AG&longitude={lon}&latitude={lat}&start={start}&end={end}&format=JSON")
GSOD_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{id}.csv"
IST = timezone(timedelta(hours=5, minutes=30))
CORE = ["tmax", "tmin", "tmean", "td", "rain", "ps", "u10", "v10"]  # must be present for a day to count as complete


def fetch_power(lat, lon, start, end):
    p = CFG["power"]
    url = POWER_URL.format(params=",".join(v for v, _ in p["variables"]), lat=lat, lon=lon,
                           start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d"))
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                d = json.load(r)
            break
        except Exception:
            if attempt == 3:
                raise
    par = d["properties"]["parameter"]
    df = pd.DataFrame({name: par[v] for v, name in p["variables"]})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.replace(-999.0, np.nan)
    df["ps"] = df["ps"] * 10  # kPa -> hPa, as in build_dataset
    return df


def fetch_gsod(sid, years):
    """Latest GSOD rows for a station (converted like build_dataset); empty if not published yet."""
    frames = []
    for y in years:
        try:
            with urllib.request.urlopen(GSOD_URL.format(year=y, id=sid), timeout=60) as r:
                frames.append(pd.read_csv(io.BytesIO(r.read()), dtype=str))
        except urllib.error.HTTPError:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames)
    df.index = pd.to_datetime(df.DATE)
    num = lambda c, miss: pd.to_numeric(df[c], errors="coerce").where(lambda s: s < miss - 0.5)
    f2c = lambda c: (num(c, 9999.9) - 32) * 5 / 9
    prcp = num("PRCP", 99.99).where(df.PRCP_ATTRIBUTES.str.strip() != "I") * 25.4
    return pd.DataFrame({"tmax": f2c("MAX"), "tmin": f2c("MIN"), "tmean": f2c("TEMP"), "td": f2c("DEWP"), "rain": prcp})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="s123_f1.0_s0")
    ap.add_argument("--init", default=None, help="checkpoint path (default outputs/runs/<run>/best.pt)")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    ck = torch.load(a.init or OUTS / "runs" / a.run / "best.pt")
    D = Data()
    layers = ck["args"]["layers"]
    vis_full = stage_setup(D, 3, 1.0, 0)[0]
    sub, ei, _, emask = k_hop_subgraph(D.local, layers, D.edge_index, relabel_nodes=True, num_nodes=D.N)
    sub = sub.numpy()
    D.restrict(sub, ei, D.edge_attr[emask])
    clim_fill = bool(ck["args"].get("clim_fill"))
    if clim_fill:  # SJCET's own day-of-year climatology (register 2010-2013) stands in for its missing inputs
        D.set_local_clim(vis_full[:, sub])
    nodes = D.nodes
    vars_ = D.vars

    today = datetime.now(IST).date()
    start = today - timedelta(days=K + 14)
    power_idx = np.flatnonzero(nodes.kind == "power")
    print(f"{len(sub)} nodes in SJCET's {layers}-hop neighbourhood: {len(power_idx)} reanalysis, "
          f"{(nodes.kind == 'gsod').sum()} NOAA, SJCET; fetching POWER {start}..{today}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        power = list(ex.map(lambda i: fetch_power(nodes.lat[i], nodes.lon[i], start, today), power_idx))

    # issue date: last day every fetched node has the core variables
    complete = pd.concat([df[CORE].notna().all(axis=1) for df in power], axis=1).all(axis=1)
    issued = complete[complete].index.max().date()
    dates = pd.date_range(issued - timedelta(days=K - 1), issued + timedelta(days=H))
    X = np.full((len(dates), len(nodes), len(vars_)), np.nan, np.float32)
    for i, df in zip(power_idx, power):
        X[:, i] = df.reindex(dates)[vars_].to_numpy(np.float32)
    X[K:] = np.nan  # nothing after the issue date is an input

    stations = []
    for i in np.flatnonzero(nodes.kind == "gsod"):
        g = fetch_gsod(nodes.id[i], sorted({issued.year, (issued - timedelta(days=K)).year}))
        n = 0
        if len(g):
            g = g.reindex(dates[:K])
            for v in g.columns:
                X[:K, i, vars_.index(v)] = g[v].to_numpy(np.float32)
            n = int(g.notna().any(axis=1).sum())
        stations.append({"id": nodes.id[i], "name": nodes.name[i], "days_with_data": n})

    # swap the live window into the restricted Data object and run the model
    ri = vars_.index("rain")
    D.rain_raw = X[:, :, ri].copy()
    X[:, :, ri] = np.log1p(np.clip(X[:, :, ri], 0, None))
    D.mask = ~np.isnan(X)
    D.Xn = np.nan_to_num((X - D.mean) / D.std).astype(np.float32)
    D.dates, D.T = dates, len(dates)
    doy = dates.dayofyear.to_numpy()
    D.doy = np.stack([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)], 1).astype(np.float32)

    model = WeatherGNN(ck["in_dim"], hidden=ck["args"]["hidden"], layers=layers)
    load_weights(model, ck["model"])
    t = K - 1
    p = run_model(model, D, np.array([t]), np.ones((D.T, D.N), bool), D.edge_index, D.edge_attr)[0, D.local]
    phys = D.denorm(p[:, :NREG])
    prob = 1 / (1 + np.exp(-p[:, NREG]))

    # recent reanalysis at the SJCET grid point (context for the chart; not station observations)
    sj = int(np.flatnonzero((nodes.kind == "power") & (nodes.region == "sjcet"))[0])
    sj_df = power[list(power_idx).index(sj)]
    recent = [{"date": d.strftime("%Y-%m-%d"), "tmax_c": round(float(r.tmax), 1), "tmin_c": round(float(r.tmin), 1),
               "rain_mm": round(float(r.rain), 1)} for d, r in sj_df.loc[:pd.Timestamp(issued)].tail(14).iterrows()]

    node = nodes.iloc[D.local]
    out = {
        "version": 1, "mode": "live",
        "generated": datetime.now(IST).isoformat(timespec="minutes"),
        "issued": issued.isoformat(), "data_through": issued.isoformat(),
        "station": {"id": CFG["local"]["node_id"], "name": node["name"], "lat": float(node.lat), "lon": float(node.lon)},
        "model": {"name": a.run, "run_id": a.run},
        "days": [{"date": (issued + timedelta(days=d + 1)).isoformat(),
                  "lead_days": d + 1,
                  "tmax_c": round(float(f[0]), 1), "tmin_c": round(float(f[1]), 1),
                  "tmean_c": round(float(f[3]), 1), "dewpoint_c": round(float(f[4]), 1),
                  "rh_pct": round(float(rh_from(f[3], f[4])), 0),
                  "rain_mm": round(float(f[2]), 1), "rain_prob": round(float(pr), 2),
                  "rain_expected": bool(pr >= 0.5)} for d, (f, pr) in enumerate(zip(phys, prob))],
        "recent": recent,
        "inputs": {"reanalysis_nodes": int(len(power_idx)), "noaa_stations": stations,
                   "sjcet_register": "no readings after 2016-12-31",
                   "sjcet_climatology": clim_fill},
        "notes": (f"Issued from data through {issued} (NASA POWER publishes with a ~3-day lag). "
                  f"rain_prob = P(rain >= {RAIN_THR} mm). SJCET's own readings are not available for this period."),
    }
    dst = OUTS / "forecasts"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "live.json").write_text(json.dumps(out, indent=1))
    print(f"issued {issued} (today {today}); NOAA stations with recent data: "
          f"{sum(s['days_with_data'] > 0 for s in stations)}/{len(stations)}")
    for d in out["days"]:
        print(f"  {d['date']} (+{d['lead_days']})  max {d['tmax_c']}  min {d['tmin_c']}  rain {d['rain_mm']} mm  P(rain) {d['rain_prob']}")


if __name__ == "__main__":
    main()
