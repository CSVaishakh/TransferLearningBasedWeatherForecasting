"""Stack NASA POWER grid nodes, GSOD station nodes and the digitized SJCET register into one array + graph.

    uv run python data/build_dataset.py

Writes processed/dataset.npz (X [T, N, V] float32 with NaN = missing, dates),
processed/nodes.csv, processed/graph.pt and outputs/graph.json.
"""
import json
import pathlib

import numpy as np
import pandas as pd
import torch
import yaml

HERE = pathlib.Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())
RAW, PROC, OUTS = HERE / "raw", HERE / "processed", HERE / "outputs"
VARS = [name for _, name in CFG["power"]["variables"]]  # tmax tmin tmean td rain ps u10 v10 tcc


def load_power(dates):
    nodes = pd.read_csv(RAW / "power" / "nodes.csv")
    X = np.full((len(dates), len(nodes), len(VARS)), np.nan, np.float32)
    for i, f in enumerate(nodes.file):
        df = pd.read_csv(RAW / "power" / f, index_col="date", parse_dates=True).reindex(dates)
        df["ps"] = df["ps"] * 10  # kPa -> hPa
        X[:, i] = df[VARS].to_numpy(np.float32)
    nodes = nodes.assign(id=nodes.file.str.removesuffix(".csv"), name="",
                         kind=nodes.kind.map({"coarse": "power", "fine": "power", "sjcet": "power"}),
                         region=nodes.kind)
    return X, nodes[["id", "kind", "region", "name", "lat", "lon", "elev"]]


def f_to_c(s, missing=9999.9):
    s = pd.to_numeric(s, errors="coerce")
    return ((s.where(s < missing - 1)) - 32) * 5 / 9


def load_gsod(dates):
    g = CFG["gsod"]
    st = pd.read_csv(RAW / "gsod" / "kerala_stations.csv", dtype={"id": str})
    st = st[(st.coverage_pct >= g["min_coverage_pct"]) & ~st.id.isin(g["exclude"])]
    reg = pd.read_csv(RAW / "gsod" / "isd-history.csv", dtype=str)
    reg["id"] = reg.USAF + reg.WBAN
    elev = reg.set_index("id")["ELEV(M)"].astype(float)
    X = np.full((len(dates), len(st), len(VARS)), np.nan, np.float32)
    for i, sid in enumerate(st.id):
        files = sorted((RAW / "gsod" / "kerala").glob(f"*/{sid}.csv"))
        df = pd.concat(pd.read_csv(f, dtype=str) for f in files)
        df.index = pd.to_datetime(df.DATE)
        df = df[~df.index.duplicated()].reindex(dates)
        prcp = pd.to_numeric(df.PRCP, errors="coerce")
        prcp = prcp.where((prcp < 99) & (df.PRCP_ATTRIBUTES.str.strip() != "I")) * 25.4  # in -> mm
        cols = {"tmax": f_to_c(df.MAX), "tmin": f_to_c(df.MIN), "tmean": f_to_c(df.TEMP),
                "td": f_to_c(df.DEWP), "rain": prcp}
        # basic QC: physical ranges, Tmin <= Tmean <= Tmax (1 degC slack), rain <= 500 mm
        bad_t = ~(cols["tmax"].between(15, 45) & cols["tmin"].between(5, 35)
                  & (cols["tmin"] <= cols["tmax"]))
        for v in ("tmax", "tmin", "tmean"):
            cols[v] = cols[v].mask(bad_t)
        cols["rain"] = cols["rain"].where(cols["rain"] <= 500)
        for name, s in cols.items():
            X[:, i, VARS.index(name)] = s.to_numpy(np.float32)
    nodes = pd.DataFrame({"id": st.id.values, "kind": "gsod", "region": "station", "name": st.name.values,
                          "lat": st.lat.values, "lon": st.lon.values, "elev": elev.reindex(st.id).values})
    return X, nodes


# qc_note reasons that make a register row's temperatures doubtful (a rain-total
# mismatch for the month only casts doubt on the rain value)
TEMP_DOUBT = r"doubtful|ambiguous|or \d|uncertain|unreadable|obscured|overwritten|struck|as written|likely \d|shifted|corrected|unclear|wet>dry|min>=max|min>dry"


def load_sjcet(dates):
    """The digitized SJCET register (data/sjcet/sjcet_daily.csv) as one station node.

    The register is read at ~08:30 IST on date D: rain on D fell from D-1 08:30 to
    D 08:30 and the max thermometer holds the maximum of D-1, so both are moved to
    D-1. The minimum (early morning of D) stays on D. Dewpoint comes from the 08:30
    dry/wet bulb pair. tmean = (tmax + tmin) / 2. Rows flagged `suspect` lose their
    rain value, and also their temperatures when the doubt is about a reading.
    Rain read after an unread day (multi-day accumulation) is dropped.
    """
    s = CFG["sjcet"]
    df = pd.read_csv(HERE.parent / "data" / "sjcet" / "sjcet_daily.csv", parse_dates=["date"]).set_index("date")
    sus = df.qc_flag.eq("suspect")
    t_bad = sus & df.qc_note.fillna("").str.contains(TEMP_DOUBT, case=False, regex=True)
    for c in ("dry_bulb_c", "wet_bulb_c", "tmin_c", "tmax_c"):
        df.loc[t_bad, c] = np.nan
    df.loc[sus, "rain_mm"] = np.nan
    # the gauge is emptied only when read: after a weekend/holiday the reading holds
    # several days of rain, so keep it only when the previous day was read too
    read = df[["dry_bulb_c", "tmax_c", "rain_mm"]].notna().any(axis=1)
    df.loc[~read.shift(1, freq="D").reindex(df.index, fill_value=False), "rain_mm"] = np.nan
    t, tw = df.dry_bulb_c, df.wet_bulb_c
    es = lambda x: 6.112 * np.exp(17.67 * x / (x + 243.5))
    e = (es(tw) - 0.000799 * 1005.0 * (t - tw)).clip(lower=0.1)
    td = 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))
    out = pd.DataFrame(index=dates)
    out["tmax"] = df.tmax_c.shift(-1, freq="D").reindex(dates)
    out["rain"] = df.rain_mm.shift(-1, freq="D").reindex(dates)
    out["tmin"] = df.tmin_c.reindex(dates)
    out["td"] = td.reindex(dates)
    out["tmean"] = (out.tmax + out.tmin) / 2
    X = np.full((len(dates), 1, len(VARS)), np.nan, np.float32)
    for v in ("tmax", "tmin", "tmean", "td", "rain"):
        X[:, 0, VARS.index(v)] = out[v].to_numpy(np.float32)
    nodes = pd.DataFrame({"id": ["SJCET"], "kind": ["sjcet"], "region": ["station"], "name": ["SJCET Palai"],
                          "lat": [s["lat"]], "lon": [s["lon"]], "elev": [s["elev"]]})
    return X, nodes


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def build_graph(nodes, k):
    lat, lon, elev = nodes.lat.to_numpy(), nodes.lon.to_numpy(), nodes.elev.fillna(0).to_numpy()
    d = haversine(lat[:, None], lon[:, None], lat[None], lon[None])
    np.fill_diagonal(d, np.inf)
    nbr = np.argsort(d, axis=1)[:, :k]
    src = np.repeat(np.arange(len(nodes)), k)
    dst = nbr.ravel()
    pairs = np.unique(np.concatenate([np.stack([src, dst], 1), np.stack([dst, src], 1)]), axis=0)
    s, t = pairs[:, 0], pairs[:, 1]
    dy = (lat[s] - lat[t]) * 111.2
    dx = (lon[s] - lon[t]) * 111.2 * np.cos(np.radians((lat[s] + lat[t]) / 2))
    attr = np.stack([dx / 100, dy / 100, d[s, t] / 100, (elev[s] - elev[t]) / 1000], 1)
    # messages flow source -> target: edge_index[0] = s (neighbour), [1] = t (receiver)
    return torch.tensor(pairs.T, dtype=torch.long), torch.tensor(attr, dtype=torch.float32)


def main():
    PROC.mkdir(exist_ok=True)
    OUTS.mkdir(exist_ok=True)
    y0, y1 = CFG["power"]["years"]
    dates = pd.date_range(f"{y0}-01-01", f"{y1}-12-31", freq="D")
    Xp, np_ = load_power(dates)
    Xg, ng = load_gsod(dates)
    Xs, ns = load_sjcet(dates)
    X = np.concatenate([Xp, Xg, Xs], axis=1)
    nodes = pd.concat([np_, ng, ns], ignore_index=True)
    nodes.index.name = "node"
    np.savez_compressed(PROC / "dataset.npz", X=X, dates=dates.strftime("%Y-%m-%d").to_numpy(), vars=np.array(VARS))
    nodes.to_csv(PROC / "nodes.csv")
    ei, ea = build_graph(nodes, CFG["graph"]["k"])
    torch.save({"edge_index": ei, "edge_attr": ea}, PROC / "graph.pt")
    json.dump({"nodes": [{"id": int(i), "kind": r.kind, "region": r.region, "lat": r.lat, "lon": r.lon,
                          **({"name": r["name"]} if r["name"] else {})} for i, r in nodes.iterrows()],
               "edges": ei.T.tolist()}, open(OUTS / "graph.json", "w"))

    print("X", X.shape, "| nodes", nodes.kind.value_counts().to_dict(), "| edges", ei.shape[1])
    deg = np.bincount(ei[1].numpy(), minlength=len(nodes))
    print("in-degree min/median/max", deg.min(), int(np.median(deg)), deg.max())
    miss = np.isnan(X).mean(axis=0)
    print("missing fraction per var (power nodes):", dict(zip(VARS, miss[: len(np_)].mean(0).round(3))))
    print("missing fraction per var (gsod nodes):", dict(zip(VARS, miss[len(np_):].mean(0).round(3))))
    print(pd.concat([ng, ns])[["id", "name", "lat", "lon", "elev"]].to_string(index=False))
    print("SJCET observed days per var:", dict(zip(VARS, (~np.isnan(Xs[:, 0])).sum(0))))


if __name__ == "__main__":
    main()
