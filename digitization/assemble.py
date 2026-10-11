"""Assemble hand transcriptions into data/sjcet/sjcet_daily.csv (data-contracts §1).

    uv run python assemble.py

Reads work/transcripts/m*.csv (one per register page, values exactly as
written, rain in cm) and work/monthly_totals.csv (rainfall totals written on
the monthly summary pages). Writes:
  ../data/sjcet/sjcet_daily.csv   one row per calendar day, 2010-07-01..2016-12-31
  work/pages.csv                  page inventory (page_id, source, year, month, ...)
  work/qc_report.csv              per-month rain checksum and flag counts
"""
import calendar
import re
from pathlib import Path

import numpy as np
import pandas as pd

from check_transcripts import read

HERE = Path(__file__).parent
TDIR = HERE / "work" / "transcripts"
OUT = HERE.parent / "data" / "sjcet"

# Remarks that mean the value was hard to read or is physically doubtful
DOUBT = re.compile(r"doubtful|ambiguous|or \d|uncertain|unreadable|obscured|overwritten|struck|"
                   r"as written|likely|shifted|corrected|unclear|stray", re.I)
ACCUM = re.compile(r"accumulated", re.I)
SUM_TOL_CM = 0.1


def rh_pct(t, tw, p=1005.0, a=0.000799):
    """Relative humidity from dry/wet bulb (°C), Stevenson-screen psychrometer."""
    es = lambda x: 6.112 * np.exp(17.67 * x / (x + 243.5))
    e = es(tw) - a * p * (t - tw)
    return np.clip(100 * e / es(t), 0, 100)


def hhmm(s):
    if not isinstance(s, str) or ":" not in s:
        return ""
    h, m = s.split(":")
    return f"{int(h):02d}:{int(m):02d}"


def main():
    totals = pd.read_csv(HERE / "work" / "monthly_totals.csv").set_index(["year", "month"])
    rows, pages, report = [], [], []
    for path in sorted(TDIR.glob("m*.csv")):
        meta, df = read(path)
        pid = path.stem
        y, m = int(meta["year"]), int(meta["month"])
        pages.append(dict(page_id=pid, source=f"Main/{meta['source']}", year=y, month=m,
                          layout=meta.get("columns", ""), duplicate_of=meta.get("duplicate_of", "")))
        if "duplicate_of" in meta:
            continue
        written = totals.total_cm.get((y, m), np.nan)
        rain_sum = round(df.rain_cm.sum(), 2)
        observed = df[["dry", "wet", "min", "max", "rain_cm"]].notna().any(axis=1)
        sum_ok = observed.sum() == 0 or abs(rain_sum - written) <= SUM_TOL_CM
        for _, r in df.iterrows():
            flags, notes = [], []
            rem = r.remarks if isinstance(r.remarks, str) else ""
            vals = {k: r[k] for k in ("dry", "wet", "min", "max", "rain_cm")}
            has = any(pd.notna(v) for v in vals.values())
            if has:
                if DOUBT.search(rem):
                    flags.append("suspect")
                if ACCUM.search(rem):
                    flags.append("suspect")
                    notes.append("rain may be accumulated over unobserved days")
                if pd.notna(vals["wet"]) and pd.notna(vals["dry"]) and vals["wet"] > vals["dry"]:
                    flags.append("suspect")
                    notes.append("wet>dry")
                if pd.notna(vals["min"]) and pd.notna(vals["max"]) and vals["min"] >= vals["max"]:
                    flags.append("suspect")
                    notes.append("min>=max")
                if pd.notna(vals["min"]) and pd.notna(vals["dry"]) and vals["min"] > vals["dry"] + 1:
                    flags.append("suspect")
                    notes.append("min>dry")
                if pd.notna(vals["rain_cm"]) and not sum_ok:
                    flags.append("suspect")
                    notes.append(f"month rain sum {rain_sum:.2f} cm != written total {written:.2f} cm")
            if rem:
                notes.insert(0, rem)
            t, tw = vals["dry"], vals["wet"]
            rh = rh_pct(t, tw) if pd.notna(t) and pd.notna(tw) else np.nan
            rows.append(dict(
                date=f"{y}-{m:02d}-{int(r.day):02d}", obs_time=hhmm(r.time) if has else "",
                rain_mm=round(vals["rain_cm"] * 10, 1) if pd.notna(vals["rain_cm"]) else np.nan,
                dry_bulb_c=t, wet_bulb_c=tw, tmin_c=vals["min"], tmax_c=vals["max"],
                rh_pct=round(rh, 1) if pd.notna(rh) else np.nan,
                source_page=pid, qc_flag="suspect" if flags else "ok",
                qc_note="; ".join(notes)))
        report.append(dict(year=y, month=m, page_id=pid, rain_sum_cm=rain_sum, written_total_cm=written,
                           diff_cm=round(rain_sum - written, 2), checksum_ok=sum_ok,
                           days_observed=int(observed.sum())))

    daily = pd.DataFrame(rows)
    # complete calendar (also covers any day a page has no row for)
    cal = pd.DataFrame({"date": pd.date_range("2010-07-01", "2016-12-31").strftime("%Y-%m-%d")})
    daily = cal.merge(daily, on="date", how="left")
    assert daily.date.is_unique
    OUT.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT / "sjcet_daily.csv", index=False, float_format="%.6g")
    pd.DataFrame(pages).to_csv(HERE / "work" / "pages.csv", index=False)
    rep = pd.DataFrame(report).sort_values(["year", "month"])
    rep.to_csv(HERE / "work" / "qc_report.csv", index=False)

    obs = daily[daily[["dry_bulb_c", "tmax_c", "rain_mm"]].notna().any(axis=1)]
    print(f"{len(daily)} days, {len(obs)} with observations "
          f"({len(obs) / len(daily):.0%}); qc: {obs.qc_flag.value_counts().to_dict()}")
    print(f"rain checksum: {rep.checksum_ok.sum()}/{len(rep)} months match")
    print(rep[~rep.checksum_ok].to_string(index=False))
    print("RH:", obs.rh_pct.describe().round(1).to_dict())


if __name__ == "__main__":
    main()
