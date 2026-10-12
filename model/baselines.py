"""Baselines for the local station, on the same test days and training months as the GNN.

    uv run python baselines.py [--fractions 0.1 0.25 0.5 1.0] [--seeds 0 1 2]

persistence    day t+d = today (observed at the station)
climatology    day-of-year mean of the station's training months, smoothed +-15 days
reanalysis     NASA POWER value at the nearest grid node on the target day (not a forecast:
               shows how far gridded reanalysis is from the station)
xgboost        station's last 7 days + nearest grid node's last 7 days + day of year, one model per lead day
Writes outputs/baselines.json (one entry per model / fraction / seed, metrics per lead day).
"""
import argparse
import json

import numpy as np
import pandas as pd
import xgboost as xgb

from train import days_for, stage_setup
from wx import CFG, H, K, NREG, OUTS, RAIN, Data, metrics


def physical(D):
    """[T, N, NREG] targets in physical units, NaN where missing."""
    out = np.stack([D.Xn[:, :, j] * D.std[j] + D.mean[j] for j in D.ti], -1)
    out[~D.mask[:, :, D.ti]] = np.nan
    out[..., RAIN] = D.rain_raw
    return out


def all_vars(D, n):
    """[T, V] physical values of every variable at node n (rain as log1p), NaN where missing."""
    v = D.Xn[:, n] * D.std + D.mean
    v[~D.mask[:, n]] = np.nan
    return v


def climatology(P, dates, train_days, test_t):
    doy = dates.dayofyear.to_numpy()
    clim = np.full((367, P.shape[-1]), np.nan)
    for d in range(1, 367):
        dist = np.minimum(np.abs(doy[train_days] - d), 366 - np.abs(doy[train_days] - d))
        clim[d] = np.nanmean(P[train_days][dist <= 15], axis=0)
    return clim[doy[test_t + 1]]


def features(D, ts, loc, near):
    win = ts[:, None] + np.arange(-K + 1, 1)[None]
    a = all_vars(D, loc)[win][:, :, :5].reshape(len(ts), -1)     # tmax tmin tmean td rain(log1p)
    b = all_vars(D, near)[win].reshape(len(ts), -1)
    return np.concatenate([a, b, D.doy[ts + 1]], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    D = Data()
    loc = D.local
    fine = np.flatnonzero(D.is_power)
    d = np.hypot(D.nodes.lat[fine] - D.nodes.lat[loc], D.nodes.lon[fine] - D.nodes.lon[loc])
    near = int(fine[np.argmin(d.to_numpy())])
    P = physical(D)
    vis_all, w3, _, _ = stage_setup(D, 3, 1.0, 0)
    te = days_for(D, "test", vis_all, w3)
    a_, b_ = (pd.Timestamp(s) for s in CFG["splits"]["train"])
    in_train = (D.dates >= a_) & (D.dates <= b_)
    leads = range(1, H + 1)

    rows = []

    def add(name, frac, seed, preds):
        """preds: one [len(te), NREG] array per lead day. `metrics` = lead day 1."""
        by_lead = [metrics(p, P[te + d, loc]) for d, p in zip(leads, preds)]
        rows.append({"model": name, "local_fraction": frac, "seed": seed, "metrics": by_lead[0],
                     "metrics_by_lead": by_lead})
        print(f"{name:12s} f={frac:<5} s={seed}  " + "  ".join(
            f"d{d}: tmax {m['tmax_c']['rmse']:.2f} tmin {m['tmin_c']['rmse']:.2f} rain {m['rain']['rmse']:.2f}"
            for d, m in zip(leads, by_lead)), flush=True)

    add("persistence", 1.0, 0, [P[te, loc] for d in leads])
    add("reanalysis", 1.0, 0, [P[te + d, near] for d in leads])
    for frac in a.fractions:
        for seed in a.seeds if frac < 1 else [0]:
            vis, _, _, _ = stage_setup(D, 3, frac, seed)
            train_days = np.flatnonzero(vis[:, loc] & in_train)
            add("climatology", frac, seed, [climatology(P[:, loc], D.dates, train_days, te + d - 1) for d in leads])
            tr_all = days_for(D, "train", vis, w3)
            Xte = features(D, te, loc, near)
            preds = []
            for d in leads:  # one model per lead day and target
                tr = tr_all[vis[tr_all + d, loc]]
                Xtr = features(D, tr, loc, near)
                pred = np.full((len(te), NREG), np.nan)
                for j in range(NREG):
                    y = P[tr + d, loc, j]
                    y = np.log1p(y) if j == RAIN else y
                    ok = ~np.isnan(y)
                    m = xgb.XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                                         colsample_bytree=0.8, random_state=seed, n_jobs=8)
                    m.fit(Xtr[ok], y[ok])
                    p = m.predict(Xte)
                    pred[:, j] = np.clip(np.expm1(p), 0, None) if j == RAIN else p
                preds.append(pred)
            add("xgboost", frac, seed, preds)
    (OUTS / "baselines.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
