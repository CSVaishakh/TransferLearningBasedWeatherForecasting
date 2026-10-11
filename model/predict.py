"""Next-day forecast for the local station in the data-contracts §4 JSON format (replay mode).

    uv run python predict.py 2015-07-10                 # print one forecast
    uv run python predict.py --export                   # write outputs/forecasts/<date>.json for the test period
    uv run python predict.py 2015-07-10 --run s123_f1.0_s0

`date` is the last day of input data ("issued"); the forecast is for the next day.
"""
import argparse
import json

import numpy as np
import torch

from train import predict as run_model
from train import stage_setup
from wx import CFG, NREG, OUTS, RAIN, RAIN_THR, Data, WeatherGNN, load_weights, rh_from

_cache = {}


def _load(run):
    if run not in _cache:
        ck = torch.load(OUTS / "runs" / run / "best.pt")
        a = ck["args"]
        D = Data()
        model = WeatherGNN(ck["in_dim"], hidden=a["hidden"], layers=a["layers"])
        load_weights(model, ck["model"])
        vis, _, _, _ = stage_setup(D, 3, 1.0, 0)  # every observation up to the issue date is usable
        _cache[run] = (D, model, vis)
    return _cache[run]


def _forecasts(run, ts):
    D, model, vis = _load(run)
    p = run_model(model, D, np.asarray(ts), vis, D.edge_index, D.edge_attr)[:, D.local]
    phys = D.denorm(p[:, :NREG])  # tmax tmin rain tmean td
    prob = 1 / (1 + np.exp(-p[:, NREG]))
    out = []
    for t, f, pr in zip(ts, phys, prob):
        nxt = t + 1
        m = D.mask[nxt, D.local, D.ti]
        obs = [float(D.Xn[nxt, D.local, j] * D.std[j] + D.mean[j]) if m[k] else None for k, j in enumerate(D.ti)]
        obs[RAIN] = None if np.isnan(D.rain_raw[nxt, D.local]) else float(D.rain_raw[nxt, D.local])
        obs_rh = None if obs[3] is None or obs[4] is None else float(rh_from(obs[3], obs[4]))
        r1 = lambda v: None if v is None else round(v, 1)
        node = D.nodes.iloc[D.local]
        out.append({
            "version": 1,
            "issued": D.dates[t].strftime("%Y-%m-%d"),
            "station": {"id": CFG["local"]["node_id"], "name": f"{node['name']} (stand-in for SJCET)",
                        "lat": float(node.lat), "lon": float(node.lon)},
            "model": {"name": run, "run_id": run},
            "days": [{"date": D.dates[nxt].strftime("%Y-%m-%d"),
                      "tmax_c": round(float(f[0]), 1), "tmin_c": round(float(f[1]), 1),
                      "tmean_c": round(float(f[3]), 1), "dewpoint_c": round(float(f[4]), 1),
                      "rh_pct": round(float(rh_from(f[3], f[4])), 0),
                      "rain_mm": round(float(f[2]), 1), "rain_prob": round(float(pr), 2),
                      "rain_expected": bool(pr >= 0.5),
                      "observed": {"tmax_c": r1(obs[0]), "tmin_c": r1(obs[1]), "rain_mm": r1(obs[2]),
                                   "tmean_c": r1(obs[3]), "dewpoint_c": r1(obs[4]),
                                   "rh_pct": None if obs_rh is None else round(obs_rh)}}],
            "notes": f"rain_prob = P(rain >= {RAIN_THR} mm); rh_pct is derived from tmean_c and dewpoint_c",
        })
    return out


def predict(date: str, model_name: str = "s123_f1.0_s0") -> dict:
    D, _, _ = _load(model_name)
    t = int(np.flatnonzero(D.dates == np.datetime64(date))[0])
    return _forecasts(model_name, [t])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?")
    ap.add_argument("--run", default="s123_f1.0_s0")
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    if a.export:
        D, _, _ = _load(a.run)
        ts = D.split_days("test")
        dst = OUTS / "forecasts"
        dst.mkdir(parents=True, exist_ok=True)
        fc = _forecasts(a.run, ts)
        for f in fc:
            (dst / f"{f['issued']}.json").write_text(json.dumps(f, indent=1))
        (dst / "index.json").write_text(json.dumps([f["issued"] for f in fc]))
        print(f"wrote {len(fc)} forecasts to {dst}")
    else:
        print(json.dumps(predict(a.date, a.run), indent=1))


if __name__ == "__main__":
    main()
