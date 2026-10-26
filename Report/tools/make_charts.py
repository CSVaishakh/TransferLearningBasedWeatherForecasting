"""Charts for the report, computed from model/outputs/metrics.json (copied to data/)
and the digitized register data/sjcet_daily.csv when present."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
M = json.loads((ROOT / "data" / "metrics.json").read_text())
INK, MUTED, GRID = "#2B2B2B", "#6B6B6B", "#E4E0D8"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)

SERIES = [  # (metrics.json name, label, color, marker, linestyle)
    ("GNN, Region → Stations → SJCET (1→2→3)", "GNN 1→2→3", PAL[0], "o", "-"),
    ("GNN, Region → SJCET (1→3)", "GNN 1→3", PAL[1], "s", "-"),
    ("GNN, SJCET only (no pretraining)", "GNN scratch", PAL[2], "^", "-"),
    ("xgboost", "XGBoost", PAL[3], "D", "--"),
    ("climatology", "Climatology", PAL[4], "v", "--"),
    ("persistence", "Persistence", PAL[5], "P", "--"),
]
models = {m["name"]: m for m in M["models"]}

# 1. RMSE vs lead day
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.4), dpi=200, sharey=True)
for ax, var, title in zip(axs, ("tmax_c", "tmin_c"), ("Maximum temperature", "Minimum temperature")):
    for name, lab, c, mk, ls in SERIES:
        y = [l[var]["rmse"] for l in models[name]["by_lead"]]
        ax.plot(range(1, 7), y, color=c, marker=mk, ls=ls, lw=2, ms=6, label=lab)
    ax.set_title(title, fontsize=10, color=INK); ax.set_xlabel("Lead day"); ax.set_xticks(range(1, 7)); style(ax)
axs[0].set_ylabel("RMSE (°C)"); axs[0].set_ylim(0.6, 2.6)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(FIG / "rmse_lead.png"); plt.close(fig)

# 2. Data efficiency: day-1 RMSE vs fraction of SJCET training months (mean ± sd over seeds)
a = pd.DataFrame(M["ablation"]); a = a[a.lead == 1]
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.4), dpi=200, sharey=True)
for ax, var, title in zip(axs, ("tmax_rmse", "tmin_rmse"), ("Maximum temperature", "Minimum temperature")):
    for name, lab, c, mk, ls in SERIES[:5]:
        g = a[a.model == name].groupby("local_fraction")[var].agg(["mean", "std"]).fillna(0)
        x = (g.index * 100).to_numpy()
        ax.errorbar(x, g["mean"], yerr=g["std"], color=c, marker=mk, ls=ls, lw=2, ms=6, capsize=3, label=lab)
    ax.set_xscale("log"); ax.set_xticks([10, 25, 50, 100]); ax.set_xticklabels(["10%", "25%", "50%", "100%"])
    ax.set_xlabel("Share of SJCET training months"); ax.set_title(title, fontsize=10, color=INK); style(ax)
axs[0].set_ylabel("Day-1 RMSE (°C)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(FIG / "data_efficiency.png"); plt.close(fig)

# 3. Register coverage per year
csv = ROOT / "data" / "sjcet_daily.csv"
if csv.exists():
    d = pd.read_csv(csv, parse_dates=["date"])
    obs = d[d[["dry_bulb_c", "tmax_c", "rain_mm"]].notna().any(axis=1)]
    cnt = obs.groupby(obs.date.dt.year).size()
    yrs = [("%d\n(Jul–Dec)" % y) if y == 2010 else str(y) for y in cnt.index]
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)
    b = ax.bar(yrs, cnt.values, color=PAL[0], width=0.62)
    for r, v in zip(b, cnt.values):
        ax.text(r.get_x() + r.get_width() / 2, v + 4, str(v), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_ylabel("Observed days"); ax.set_ylim(0, 300); style(ax)
    fig.tight_layout(); fig.savefig(FIG / "register_coverage.png"); plt.close(fig)
print("ok")
