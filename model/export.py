"""Collect run results into outputs/metrics.json (docs/data-contracts.md §5) and figures.

    uv run python export.py
"""
import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from wx import CFG, H, OUTS, metrics, rh_from  # noqa: E402

FIG = OUTS / "figures"
STRATEGIES = {"scratch": "GNN, SJCET only (no pretraining)", "s13": "GNN, Region → SJCET (1→3)",
              "s123": "GNN, Region → Stations → SJCET (1→2→3)",
              # same, with SJCET's day-of-year climatology in place of its missing inputs (train.py --clim-fill)
              "scratchc": "GNN, SJCET only + SJCET climatology", "s123c": "GNN, Region → Stations → SJCET + SJCET climatology"}
SEASONS = {"Winter (Jan–Feb)": [1, 2], "Pre-monsoon (Mar–May)": [3, 4, 5],
           "SW monsoon (Jun–Sep)": [6, 7, 8, 9], "NE monsoon (Oct–Dec)": [10, 11, 12]}


def local_name():
    """Display name of the stage-3 target station."""
    nodes = pd.read_csv(OUTS.parent / "processed" / "nodes.csv", dtype={"id": str}, keep_default_na=False)
    node = nodes[nodes.id == CFG["local"]["node_id"]].iloc[0]
    return node["name"] if node["kind"] == "sjcet" else f"{node['name']} (GSOD), stand-in for SJCET"


def load_runs():
    runs = {}
    for f in sorted((OUTS / "runs").glob("*/result.json")):
        if f.parent.name.startswith("_"):
            continue
        r = json.loads(f.read_text())
        if r.get("local_node") != CFG["local"]["node_id"]:  # scored at a different station: not comparable
            continue
        runs[f.parent.name] = r
    return runs


def flat(m):
    return {"tmax_rmse": m["tmax_c"]["rmse"], "tmin_rmse": m["tmin_c"]["rmse"], "rain_rmse": m["rain"]["rmse"],
            "rh_rmse": m.get("rh_pct", {}).get("rmse"),
            "rain_accuracy": m["rain"]["accuracy"]}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    base = json.loads((OUTS / "baselines.json").read_text())

    # `metrics` = lead day 1; `by_lead` = one metrics dict per lead day (1..H)
    models = [{"name": b["model"], "group": "baseline", "metrics": b["metrics"],
               "by_lead": b.get("metrics_by_lead", [b["metrics"]])}
              for b in base if b["local_fraction"] == 1.0]
    gnn = [(rid, label) for rid, label in [("s1", "GNN, Region only (1; no SJCET data)"),
                                           ("s12", "GNN, Region → Stations (1→2; no SJCET data)")]]
    gnn += [(f"{key}_f1.0_s0", label) for key, label in STRATEGIES.items()]
    for rid, label in gnn:
        if rid in runs:
            r = runs[rid]
            models.append({"name": label, "group": "gnn", "metrics": r["local_test"],
                           "by_lead": r.get("local_test_by_lead", [r["local_test"]]),
                           # live-like: SJCET's own inputs hidden for the whole test period
                           **({"by_lead_hidden": r["local_test_hidden_by_lead"]} if "local_test_hidden_by_lead" in r else {})})

    ablation = [{"model": b["model"], "local_fraction": b["local_fraction"], "seed": b["seed"], "lead": d + 1, **flat(m)}
                for b in base if b["model"] in ("xgboost", "climatology")
                for d, m in enumerate(b.get("metrics_by_lead", [b["metrics"]]))]
    for rid, r in runs.items():
        key = rid.split("_f")[0]
        if key in STRATEGIES:
            for d, m in enumerate(r.get("local_test_by_lead", [r["local_test"]])):
                ablation.append({"model": STRATEGIES[key], "local_fraction": r["args"]["fraction"],
                                 "seed": r["args"]["seed"], "lead": d + 1, **flat(m)})

    by_season = []
    for key, label in STRATEGIES.items():
        f = OUTS / "runs" / f"{key}_f1.0_s0" / "local_test_pred.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        pred, obs, prob = (z[k][:, 0] if z[k].ndim > (2 if k != "rain_prob" else 1) else z[k]
                           for k in ("pred", "obs", "rain_prob"))
        months = pd.to_datetime(z["dates"]).month
        for season, ms in SEASONS.items():
            sel = np.isin(months, ms)
            m = metrics(pred[sel], obs[sel], prob[sel])
            by_season.append({"model": label, "season": season, **flat(m)})

    out = {"version": 1, "horizon": H, "test_period": [str(d) for d in CFG["splits"]["test"]],
           "local_station": {"id": CFG["local"]["node_id"], "name": local_name()},
           "models": models, "ablation": ablation, "by_season": by_season}
    (OUTS / "metrics.json").write_text(json.dumps(out, indent=1))
    pd.DataFrame(ablation).to_csv(OUTS / "ablation.csv", index=False)

    # --- figures ---------------------------------------------------------------
    ab = pd.DataFrame(ablation)
    ab = ab[ab.lead == 1] if len(ab) else ab  # figures show lead day 1
    if len(ab):
        fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
        for ax, (col, title) in zip(axes, [("tmax_rmse", "Max temp RMSE (°C)"), ("tmin_rmse", "Min temp RMSE (°C)"),
                                           ("rain_rmse", "Rainfall RMSE (mm)"),
                                           ("rh_rmse", "Rel. humidity RMSE (%)")]):
            for name, g in ab.groupby("model"):
                s = g.groupby("local_fraction")[col].agg(["mean", "std"]).fillna(0)
                ax.errorbar(s.index * 100, s["mean"], yerr=s["std"], marker="o", capsize=3, label=name)
            pers = next(b for b in base if b["model"] == "persistence")["metrics"]
            key = {"tmax_rmse": "tmax_c", "tmin_rmse": "tmin_c", "rain_rmse": "rain", "rh_rmse": "rh_pct"}[col]
            ax.axhline(pers[key]["rmse"], color="grey", ls="--", lw=1, label="persistence")
            ax.set_xscale("log")
            ax.set_xticks([10, 25, 50, 100], ["10%", "25%", "50%", "100%"])
            ax.set_xlabel("local training data used")
            ax.set_title(title)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Test error vs amount of local data ({local_name()}, test 2015–2016)")
        fig.tight_layout()
        fig.savefig(FIG / "ablation.png", dpi=150)
        plt.close(fig)

    f = OUTS / "runs" / "s123_f1.0_s0" / "local_test_pred.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        d = pd.to_datetime(z["dates"])
        pr, ob = z["pred"], z["obs"]
        if pr.ndim == 3:  # [n, H, NREG]: plot lead day 1
            pr, ob = pr[:, 0], ob[:, 0]
        pr = np.column_stack([pr[:, :3], rh_from(pr[:, 3], pr[:, 4])])
        ob = np.column_stack([ob[:, :3], rh_from(ob[:, 3], ob[:, 4])])
        z = {"pred": pr, "obs": ob}
        fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
        for j, (ax, lab) in enumerate(zip(axes, ["Max temp (°C)", "Min temp (°C)", "Rainfall (mm)",
                                                  "Rel. humidity (%)"])):
            if j == 2:
                ax.bar(d, z["obs"][:, j], color="#9ecae1", label="observed", width=1)
                ax.plot(d, z["pred"][:, j], color="#08519c", lw=0.8, label="forecast")
            else:
                ax.plot(d, z["obs"][:, j], color="#555", lw=0.8, label="observed")
                ax.plot(d, z["pred"][:, j], color="#d94801", lw=0.8, label="forecast")
            ax.set_ylabel(lab)
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
        fig.suptitle(f"Next-day forecast vs observed, {local_name()}, pretrained GNN (1→2→3), test period")
        fig.tight_layout()
        fig.savefig(FIG / "test_timeseries.png", dpi=150)
        plt.close(fig)

    g = json.loads((OUTS / "graph.json").read_text())
    nd = pd.DataFrame(g["nodes"])
    fig, ax = plt.subplots(figsize=(7, 7))
    for s, t in g["edges"]:
        a, b = nd.iloc[s], nd.iloc[t]
        ax.plot([a.lon, b.lon], [a.lat, b.lat], color="#ccc", lw=0.3, zorder=0)
    for (kind, region), sty in {("power", "coarse"): dict(s=6, c="#888", label="reanalysis 1.0°"),
                                ("power", "fine"): dict(s=10, c="#3182bd", label="reanalysis 0.5° (Kerala)"),
                                ("power", "sjcet"): dict(s=60, c="#e6550d", marker="*", label="SJCET"),
                                ("gsod", "station"): dict(s=40, c="#31a354", marker="s", label="NOAA stations")}.items():
        sel = nd[(nd.kind == kind) & (nd.region == region)]
        ax.scatter(sel.lon, sel.lat, zorder=2, **sty)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title(f"Graph: {len(nd)} nodes, {len(g['edges'])} edges")
    fig.tight_layout()
    fig.savefig(FIG / "graph_map.png", dpi=150)
    plt.close(fig)

    print(pd.DataFrame([{"model": m["name"], **flat(m["metrics"])} for m in models]).round(3).to_string(index=False))
    if len(ab):
        print(ab.groupby(["model", "local_fraction"])[["tmax_rmse", "tmin_rmse", "rain_rmse"]].mean().round(3).to_string())
    by_lead = pd.DataFrame([{"model": m["name"], "lead": d + 1, "tmax_rmse": x["tmax_c"]["rmse"], "rain_acc": x["rain"]["accuracy"]}
                            for m in models for d, x in enumerate(m["by_lead"])])
    print(by_lead.pivot(index="model", columns="lead", values="tmax_rmse").round(2).to_string())


if __name__ == "__main__":
    main()
