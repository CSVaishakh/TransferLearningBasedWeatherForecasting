"""Download NASA POWER daily point data for every graph node location.

    uv run python data/download_power.py [--workers 4]

Nodes: a 1.0° grid over the India box (skipping cells inside the Kerala box)
plus a 0.5° grid over the Kerala box, plus the SJCET point. One CSV per node in
raw/power/, nodes listed in raw/power/nodes.csv. Re-run safely: finished nodes
are skipped.
"""
import argparse
import json
import pathlib
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yaml

HERE = pathlib.Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())
OUT = HERE / "raw" / "power"
URL = ("https://power.larc.nasa.gov/api/temporal/daily/point?parameters={params}"
       "&community=AG&longitude={lon}&latitude={lat}&start={start}0101&end={end}1231&format=JSON")


def grid(box, step):
    lats = np.arange(box["south"], box["north"] + 1e-9, step)
    lons = np.arange(box["west"], box["east"] + 1e-9, step)
    return [(round(float(a), 3), round(float(o), 3)) for a in lats for o in lons]


def nodes():
    p, r = CFG["power"], CFG["regions"]
    k = r["kerala"]
    inside = lambda a, o: k["south"] <= a <= k["north"] and k["west"] <= o <= k["east"]
    rows = [("coarse", a, o) for a, o in grid(r["india"], p["coarse_step"]) if not inside(a, o)]
    rows += [("fine", a, o) for a, o in grid(k, p["fine_step"])]
    rows.append(("sjcet", CFG["sjcet"]["lat"], CFG["sjcet"]["lon"]))
    df = pd.DataFrame(rows, columns=["kind", "lat", "lon"])
    df["file"] = [f"{a:.3f}_{o:.3f}.csv" for a, o in zip(df.lat, df.lon)]
    return df


def fetch(lat, lon, target, retries=5):
    if target.exists():
        return "skip", None
    p = CFG["power"]
    url = URL.format(params=",".join(v for v, _ in p["variables"]), lat=lat, lon=lon,
                     start=p["years"][0], end=p["years"][1])
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                d = json.load(r)
            break
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(5 * 2 ** attempt)
    par = d["properties"]["parameter"]
    df = pd.DataFrame({name: par[v] for v, name in p["variables"]})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.replace(-999.0, np.nan)
    df.index.name = "date"
    tmp = target.with_suffix(".part")
    df.to_csv(tmp, float_format="%.2f")
    tmp.rename(target)
    return "ok", d["geometry"]["coordinates"][2]  # POWER's grid-cell elevation (m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = nodes()
    elev_f = OUT / "elevation.json"
    elev = json.loads(elev_f.read_text()) if elev_f.exists() else {}
    print(f"{len(df)} nodes: {df.kind.value_counts().to_dict()}", flush=True)
    failed = []
    with ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(fetch, r.lat, r.lon, OUT / r.file): r.file for r in df.itertuples()}
        for i, fut in enumerate(as_completed(futs), 1):
            f = futs[fut]
            try:
                status, z = fut.result()
                if z is not None:
                    elev[f] = z
                if i % 50 == 0 or i == len(df):
                    print(f"[{i}/{len(df)}] {status} {f}", flush=True)
                    elev_f.write_text(json.dumps(elev))
            except Exception as exc:
                failed.append(f)
                print(f"[{i}/{len(df)}] FAILED {f}: {exc}", flush=True)
    elev_f.write_text(json.dumps(elev))
    df["elev"] = df.file.map(elev)
    df.to_csv(OUT / "nodes.csv", index=False)
    print(f"done: {len(df) - len(failed)} ok, {len(failed)} failed", flush=True)


if __name__ == "__main__":
    main()
