"""Shared pieces: data loading/normalisation, day sampling, the GNN, loss and metrics."""
import json
import pathlib

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch_geometric.nn import GATv2Conv

HERE = pathlib.Path(__file__).resolve().parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())
PROC, OUTS = HERE / "processed", HERE / "outputs"

K = 7                                  # input window (days)
TARGETS = ["tmax", "tmin", "rain", "tmean", "td"]  # regression targets; + rain occurrence (>= RAIN_THR mm)
NAMES = ["tmax_c", "tmin_c", "rain", "tmean_c", "td_c"]
NREG = len(TARGETS)                    # outputs [:NREG] regression, [NREG] rain logit
NOUT = NREG + 1
RAIN = TARGETS.index("rain")
RAIN_THR = 2.5                         # IMD rainy-day threshold


class Data:
    """Normalised arrays + node metadata, shared by every stage."""

    def __init__(self):
        d = np.load(PROC / "dataset.npz", allow_pickle=True)
        X = d["X"]
        self.vars = list(d["vars"])
        self.dates = pd.to_datetime(d["dates"])
        self.nodes = pd.read_csv(PROC / "nodes.csv", dtype={"id": str}, keep_default_na=False)
        g = torch.load(PROC / "graph.pt")
        self.edge_index, self.edge_attr = g["edge_index"], g["edge_attr"]
        self.T, self.N, self.V = X.shape
        self.local = int(np.flatnonzero(self.nodes.id == CFG["local"]["node_id"])[0])
        self.is_power = (self.nodes.kind == "power").to_numpy()
        self.is_fine = (self.nodes.region == "fine").to_numpy() | (self.nodes.region == "sjcet").to_numpy()
        self.is_station = (self.nodes.kind == "gsod").to_numpy()

        ri = self.vars.index("rain")
        self.rain_raw = X[:, :, ri].copy()
        X[:, :, ri] = np.log1p(np.clip(X[:, :, ri], 0, None))
        self.mask = ~np.isnan(X)
        norm_f = PROC / "norm.json"
        if norm_f.exists():
            n = json.loads(norm_f.read_text())
        else:  # stage-1 training years, reanalysis nodes only; reused by every stage
            tr = self.split_days("train")
            sub = X[tr][:, self.is_power]
            n = {"mean": np.nanmean(sub, axis=(0, 1)).tolist(), "std": np.nanstd(sub, axis=(0, 1)).tolist()}
            norm_f.write_text(json.dumps(n))
        self.mean, self.std = np.array(n["mean"], np.float32), np.array(n["std"], np.float32)
        self.Xn = np.nan_to_num((X - self.mean) / self.std).astype(np.float32)
        del X
        self.ti = [self.vars.index(v) for v in TARGETS]

        nd = self.nodes
        self.static = np.stack([(nd.lat - 20) / 10, (nd.lon - 83) / 10, pd.to_numeric(nd.elev, errors="coerce").fillna(0) / 1000,
                                self.is_power, self.is_station], 1).astype(np.float32)
        doy = self.dates.dayofyear.to_numpy()
        self.doy = np.stack([np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)], 1).astype(np.float32)
        self.in_dim = 2 * K * self.V + self.static.shape[1] + 2

    def restrict(self, sub, edge_index, edge_attr):
        """Keep only nodes `sub` (e.g. the local station's k-hop neighbourhood) with a relabelled graph."""
        sub = np.asarray(sub)
        self.Xn, self.mask, self.rain_raw = self.Xn[:, sub], self.mask[:, sub], self.rain_raw[:, sub]
        self.static = self.static[sub]
        self.is_power, self.is_fine, self.is_station = self.is_power[sub], self.is_fine[sub], self.is_station[sub]
        self.nodes = self.nodes.iloc[sub].reset_index(drop=True)
        self.local = int(np.flatnonzero(sub == self.local)[0])
        self.N = len(sub)
        self.edge_index, self.edge_attr = edge_index, edge_attr

    def split_days(self, split):
        """Indices t ("today") whose window and next day lie inside the split period."""
        a, b = (pd.Timestamp(s) for s in CFG["splits"][split])
        t = np.arange(K - 1, self.T - 1)
        return t[(self.dates[t - K + 1] >= a) & (self.dates[t + 1] <= b)]

    def batch(self, ts, visible):
        """Inputs/targets for days ts. visible: bool [T, N], which observations exist for this stage."""
        ts = np.asarray(ts)
        win = ts[:, None] + np.arange(-K + 1, 1)[None]                    # [B, K]
        vis = visible[win][..., None]                                       # [B, K, N, 1]
        x = self.Xn[win] * vis
        m = (self.mask[win] & vis).astype(np.float32)
        B = len(ts)
        x = x.transpose(0, 2, 1, 3).reshape(B, self.N, K * self.V)
        m = m.transpose(0, 2, 1, 3).reshape(B, self.N, K * self.V)
        st = np.broadcast_to(self.static, (B, self.N, self.static.shape[1]))
        dy = np.broadcast_to(self.doy[ts + 1][:, None], (B, self.N, 2))
        feats = np.concatenate([x, m, st, dy], -1)

        nxt = ts + 1
        vis_y = visible[nxt][..., None]
        y = self.Xn[nxt][:, :, self.ti]
        rain = self.rain_raw[nxt]
        occ = (np.nan_to_num(rain) >= RAIN_THR).astype(np.float32)[..., None]
        y = np.concatenate([y, occ], -1)
        ym = self.mask[nxt][:, :, self.ti]
        ym = np.concatenate([ym, ym[..., RAIN:RAIN + 1]], -1) & vis_y
        f = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))
        return f(feats), f(y), f(ym)

    def denorm(self, y):
        """[..., NREG] normalised targets -> physical units (degC, mm for rain)."""
        out = y * self.std[self.ti] + self.mean[self.ti]
        out[..., RAIN] = np.clip(np.expm1(out[..., RAIN]), 0, None)
        return out


class WeatherGNN(nn.Module):
    def __init__(self, in_dim, edge_dim=4, hidden=128, layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.convs = nn.ModuleList(
            GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=edge_dim) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.ffs = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2 * hidden), nn.GELU(),
                          nn.Linear(2 * hidden, hidden)) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.dec = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, NOUT))

    def forward(self, x, edge_index, edge_attr):
        h = self.enc(x)
        for conv, norm, ff in zip(self.convs, self.norms, self.ffs):
            h = h + self.drop(conv(norm(h), edge_index, edge_attr))
            h = h + self.drop(ff(h))
        return self.dec(h)  # [:, :NREG] regression (normalised), [:, NREG] rain logit


def load_weights(model, state):
    """Load a checkpoint, also one saved with fewer regression targets (older runs predicted
    tmax/tmin/rain only): existing outputs are copied across, new target rows keep their init."""
    own = model.state_dict()
    for k in ("dec.2.weight", "dec.2.bias"):
        old = state[k]
        if old.shape[0] != own[k].shape[0]:
            new = own[k].clone()
            n_old = old.shape[0] - 1                 # old regression targets
            new[:n_old] = old[:n_old]
            new[NREG] = old[n_old]                   # rain logit
            state = {**state, k: new}
    model.load_state_dict(state)


def batch_graph(edge_index, edge_attr, B, N):
    E = edge_index.size(1)
    offs = (torch.arange(B) * N).repeat_interleave(E)
    return edge_index.repeat(1, B) + offs, edge_attr.repeat(B, 1)


def loss_fn(pred, y, m, w, rain_pos_weight=2.0):
    """pred/y/m: [B*N, NOUT]; w: [B*N] per-node weight. Masked, weighted MSE + BCE."""
    w = w[:, None]
    r = slice(0, NREG)
    reg = ((pred[:, r] - y[:, r]) ** 2 * m[:, r] * w).sum() / (m[:, r] * w).sum().clamp(min=1)
    bce = nn.functional.binary_cross_entropy_with_logits(
        pred[:, NREG], y[:, NREG], pos_weight=torch.tensor(rain_pos_weight), reduction="none")
    cls = (bce * m[:, NREG] * w[:, 0]).sum() / (m[:, NREG] * w[:, 0]).sum().clamp(min=1)
    return reg + 0.5 * cls


def rh_from(t, td):
    """Relative humidity (%) from air temperature and dewpoint (degC), Magnus formula."""
    e = lambda x: 6.112 * np.exp(17.62 * x / (243.12 + x))
    return np.clip(100 * e(td) / e(t), 0, 100)


def metrics(pred, obs, rain_prob=None):
    """pred/obs: [n, NREG] physical units (NaN = missing), columns in TARGETS order.
    Relative humidity is derived from the tmean and td columns, never predicted directly."""
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    cols = {name: j for j, name in enumerate(NAMES[: pred.shape[1]])}
    if "tmean_c" in cols and "td_c" in cols:
        pred = np.column_stack([pred, rh_from(pred[:, cols["tmean_c"]], pred[:, cols["td_c"]])])
        obs = np.column_stack([obs, rh_from(obs[:, cols["tmean_c"]], obs[:, cols["td_c"]])])
        cols["rh_pct"] = pred.shape[1] - 1
    out = {}
    for name, j in cols.items():
        ok = ~np.isnan(obs[:, j]) & ~np.isnan(pred[:, j])
        p, o = pred[ok, j], obs[ok, j]
        if len(o) == 0:
            continue
        e = p - o
        r = {"n": int(len(o)), "rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()),
             "bias": float(e.mean()), "r2": float(1 - (e ** 2).sum() / ((o - o.mean()) ** 2).sum())}
        if name == "rain":
            po = rain_prob[ok] >= 0.5 if rain_prob is not None else p >= RAIN_THR
            oo = o >= RAIN_THR
            r["hit_rate"] = float((po & oo).sum() / max(oo.sum(), 1))
            r["false_alarm"] = float((po & ~oo).sum() / max(po.sum(), 1))
            r["accuracy"] = float((po == oo).mean())
            if rain_prob is not None:
                r["brier"] = float(((rain_prob[ok] - oo) ** 2).mean())
        else:
            r["mape"] = float(np.abs(e / o).mean() * 100)
        out[name] = r
    return out
