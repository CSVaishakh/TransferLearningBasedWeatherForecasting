"""Train one stage of the staged transfer-learning GNN and evaluate it.

    uv run python train.py --stage 1 --run-id s1
    uv run python train.py --stage 2 --init outputs/runs/s1/best.pt --run-id s12
    uv run python train.py --stage 3 --init outputs/runs/s12/best.pt --fraction 0.25 --seed 0 --run-id s123_f0.25_s0

Stage 1: reanalysis (NASA POWER) nodes only, loss on all of them.
Stage 2: + GSOD stations (except the held-out local station); loss on the Kerala
         fine grid and stations.
Stage 3: + the local station (stand-in for SJCET), loss on it only, trained on a
         `fraction` of its training months.
Writes outputs/runs/<run-id>/{best.pt, log.csv, result.json}.
"""
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import k_hop_subgraph

from wx import CFG, NOUT, NREG, OUTS, RAIN, Data, WeatherGNN, batch_graph, load_weights, loss_fn, metrics


def stage_setup(D, stage, fraction, seed):
    """visible [T, N]: which observations the model may see; w [N]: loss weight per node."""
    vis = np.ones((D.T, D.N), bool)
    w = np.zeros(D.N, np.float32)
    if stage == 1:
        vis[:, ~D.is_power] = False
        w[D.is_power] = 1.0
    elif stage == 2:
        vis[:, D.local] = False
        w[D.is_power & D.is_fine] = 1.0
        w[D.is_power & ~D.is_fine] = 0.1
        w[D.is_station] = 2.0
        w[D.local] = 0.0
    else:
        a, b = (pd.Timestamp(s) for s in CFG["splits"]["train"])
        in_train = (D.dates >= a) & (D.dates <= b)
        has = D.mask[:, D.local, D.ti].any(1)
        months = pd.PeriodIndex(D.dates, freq="M")
        avail = np.unique(months[in_train & has])
        rng = np.random.default_rng(seed)
        keep = rng.permutation(avail)[: max(1, math.ceil(fraction * len(avail)))]
        sel = np.isin(months, keep)
        vis[in_train & ~sel, D.local] = False
        w[D.local] = 1.0
        return vis, w, len(keep), len(avail)
    return vis, w, None, None


def days_for(D, split, vis, w):
    """Days whose next day has at least one weighted, visible target."""
    ts = D.split_days(split)
    nxt = ts + 1
    tgt = D.mask[nxt][:, :, D.ti].any(-1) & vis[nxt] & (w > 0)[None]
    return ts[tgt.any(1)]


@torch.no_grad()
def predict(model, D, ts, vis, ei, ea, bs=32):
    model.eval()
    preds = []
    for i in range(0, len(ts), bs):
        x, _, _ = D.batch(ts[i:i + bs], vis)
        B = x.shape[0]
        e, a = batch_graph(ei, ea, B, D.N)
        preds.append(model(x.view(B * D.N, -1), e, a).view(B, D.N, NOUT).numpy())
    return np.concatenate(preds)


def evaluate(model, D, ts, vis, ei, ea, nodes):
    """Metrics in physical units at the given node(s), pooled over days."""
    p = predict(model, D, ts, vis, ei, ea)
    phys = D.denorm(p[..., :NREG])
    prob = 1 / (1 + np.exp(-p[..., NREG]))
    nxt = ts + 1
    obs = np.stack([D.Xn[nxt][:, :, j] * D.std[j] + D.mean[j] for j in D.ti], -1)
    obs[~D.mask[nxt][:, :, D.ti]] = np.nan
    obs[..., RAIN] = D.rain_raw[nxt]  # NaN where missing
    sel = lambda a: a[:, nodes].reshape(-1, *a.shape[2:])
    return metrics(sel(phys), sel(obs), sel(prob)), phys, prob, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--init", default=None)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps", type=int, default=150, help="batches per epoch (random days)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--freeze", type=int, default=0, help="freeze encoder + first N conv layers")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    a = ap.parse_args()
    defaults = {1: (12, 1e-3), 2: (10, 3e-4), 3: (25, 1e-4 if a.init else 1e-3)}
    epochs = a.epochs or defaults[a.stage][0]
    lr = a.lr or defaults[a.stage][1]

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(8)
    out = OUTS / "runs" / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    D = Data()
    vis, w, n_months, n_avail = stage_setup(D, a.stage, a.fraction, a.seed)
    if a.stage == 3:  # only the local node is scored: its `layers`-hop neighbourhood gives identical outputs
        sub, ei_s, _, emask = k_hop_subgraph(D.local, a.layers, D.edge_index, relabel_nodes=True, num_nodes=D.N)
        sub = sub.numpy()
        D.restrict(sub, ei_s, D.edge_attr[emask])
        vis, w = vis[:, sub], w[sub]
        print(f"stage 3 subgraph: {D.N} nodes, {ei_s.shape[1]} edges", flush=True)
    tr, va, te = (days_for(D, s, vis, w) for s in ("train", "val", "test"))
    # stage 3: evaluate the local station on all val/test days where it has observations
    print(f"stage {a.stage} | train {len(tr)} val {len(va)} test {len(te)} days"
          + (f" | local months {n_months}/{n_avail}" if n_months else ""), flush=True)

    model = WeatherGNN(D.in_dim, hidden=a.hidden, layers=a.layers)
    if a.init:
        load_weights(model, torch.load(a.init)["model"])
    if a.freeze:
        for p in list(model.enc.parameters()) + [p for c in model.convs[: a.freeze] for p in c.parameters()]:
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    steps_per_epoch = min(a.steps, math.ceil(len(tr) / a.batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.1)
    ei, ea = D.edge_index, D.edge_attr
    wt = torch.from_numpy(w)

    def run_loss(ts, train):
        x, y, m = D.batch(ts, vis)
        B = x.shape[0]
        e, at = batch_graph(ei, ea, B, D.N)
        pred = model(x.view(B * D.N, -1), e, at)
        return loss_fn(pred, y.view(B * D.N, NOUT), m.view(B * D.N, NOUT), wt.repeat(B))

    log, best, bad = [], float("inf"), 0
    rng = np.random.default_rng(a.seed)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        order = rng.permutation(tr)[: steps_per_epoch * a.batch]
        tl = []
        for i in range(0, len(order), a.batch):
            loss = run_loss(order[i:i + a.batch], True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            tl.append(loss.item())
        model.eval()
        with torch.no_grad():
            vl = float(np.mean([run_loss(va[i:i + 32], False).item() for i in range(0, len(va), 32)]))
        log.append({"epoch": ep, "train_loss": float(np.mean(tl)), "val_loss": vl, "sec": round(time.time() - t0, 1)})
        print(f"ep {ep:2d} train {log[-1]['train_loss']:.4f} val {vl:.4f} ({log[-1]['sec']} s)", flush=True)
        if vl < best:
            best, bad = vl, 0
            torch.save({"model": model.state_dict(), "args": vars(a), "in_dim": D.in_dim}, out / "best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print("early stop", flush=True)
                break
    pd.DataFrame(log).to_csv(out / "log.csv", index=False)

    model.load_state_dict(torch.load(out / "best.pt")["model"])
    # local station: in stages 1-2 it is hidden, so this is a zero-shot forecast at its location
    vis3, w3, _, _ = stage_setup(D, 3, 1.0, 0)
    vis_eval = vis.copy()
    if a.stage < 3:
        vis_eval[:, D.local] = False
    te_local = days_for(D, "test", vis3, w3)
    va_local = days_for(D, "val", vis3, w3)
    res = {"run_id": a.run_id, "args": vars(a), "best_val_loss": best,
           "local_node": CFG["local"]["node_id"], "n_train_days": int(len(tr)),
           "local_months": n_months, "local_months_available": n_avail}
    res["local_val"], *_ = evaluate(model, D, va_local, vis_eval, ei, ea, [D.local])
    res["local_test"], phys, prob, obs = evaluate(model, D, te_local, vis_eval, ei, ea, [D.local])
    np.savez_compressed(out / "local_test_pred.npz", dates=D.dates[te_local + 1].strftime("%Y-%m-%d").to_numpy(),
                        pred=phys[:, D.local], rain_prob=prob[:, D.local], obs=obs[:, D.local])
    if a.stage < 3:
        fine = np.flatnonzero(D.is_power & D.is_fine)
        res["fine_grid_test"], *_ = evaluate(model, D, D.split_days("test"), vis_eval, ei, ea, fine)
    (out / "result.json").write_text(json.dumps(res, indent=1))
    lt = res["local_test"]
    print("local test RMSE  tmax {:.2f}  tmin {:.2f}  rain {:.2f}  RH {:.1f} | rain acc {:.2f}".format(
        lt["tmax_c"]["rmse"], lt["tmin_c"]["rmse"], lt["rain"]["rmse"], lt.get("rh_pct", {}).get("rmse", np.nan),
        lt["rain"]["accuracy"]), flush=True)


if __name__ == "__main__":
    main()
