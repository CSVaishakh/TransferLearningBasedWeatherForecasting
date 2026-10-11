"""Sanity-check hand transcriptions in work/transcripts/*.csv.

    uv run python check_transcripts.py [m05 m11 ...]

For each month: sum of daily rain vs the total written on the page, the
calendar (row count = days in month), and per-row range/consistency rules.
"""
import calendar
import io
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
TDIR = HERE / "work" / "transcripts"


def read(path):
    """Header comment (key=value pairs) + the CSV body."""
    lines = path.read_text().splitlines()
    meta = dict(kv.split("=", 1) for kv in lines[0].lstrip("# ").split() if "=" in kv)
    body = "\n".join(l for l in lines[1:] if l.strip())
    df = pd.read_csv(io.StringIO(body), dtype={"time": str}) if body else None
    return meta, df


def check(path):
    meta, df = read(path)
    tag = path.stem
    if "duplicate_of" in meta:
        return f"{tag}: duplicate of {meta['duplicate_of']}", []
    y, m = int(meta["year"]), int(meta["month"])
    issues = []
    ndays = calendar.monthrange(y, m)[1]
    if list(df.day) != list(range(1, ndays + 1)):
        issues.append(f"days {df.day.min()}..{df.day.max()} n={len(df)}, expected 1..{ndays}")
    for _, r in df.iterrows():
        d = int(r.day)
        v = {k: r[k] for k in ("dry", "wet", "min", "max", "rain_cm")}
        if pd.notna(v["wet"]) and pd.notna(v["dry"]) and v["wet"] > v["dry"]:
            issues.append(f"day {d}: wet {v['wet']} > dry {v['dry']}")
        if pd.notna(v["min"]) and pd.notna(v["max"]) and v["min"] >= v["max"]:
            issues.append(f"day {d}: min {v['min']} >= max {v['max']}")
        if pd.notna(v["min"]) and pd.notna(v["dry"]) and v["min"] > v["dry"] + 1:
            issues.append(f"day {d}: min {v['min']} > dry {v['dry']}")
        for k, lo, hi in (("dry", 15, 40), ("wet", 13, 40), ("min", 14, 30), ("max", 22, 41), ("rain_cm", 0, 40)):
            if pd.notna(v[k]) and not lo <= v[k] <= hi:
                issues.append(f"day {d}: {k}={v[k]} out of [{lo},{hi}]")
    total = round(df.rain_cm.sum(), 2)
    written = float(meta["written_total_cm"]) if meta.get("written_total_cm") else None
    ok = written is not None and abs(total - written) < 0.005
    head = f"{tag}: {y}-{m:02d} rain sum {total:.2f} vs written {written} {'OK' if ok else 'MISMATCH %+.2f' % (total - written) if written is not None else '(no total)'}"
    return head, issues


def main():
    names = sys.argv[1:]
    paths = [TDIR / f"{n}.csv" for n in names] if names else sorted(TDIR.glob("m*.csv"))
    bad = 0
    for p in paths:
        head, issues = check(p)
        bad += "MISMATCH" in head
        print(head)
        for i in issues:
            print("   ", i)
    print(f"\n{len(paths)} pages, {bad} rain-total mismatches")


if __name__ == "__main__":
    main()
