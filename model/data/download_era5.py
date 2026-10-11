"""Download ERA5 daily statistics for the India box from the Copernicus CDS.

    uv run python data/download_era5.py [--years 1990 2007] [--workers 4]

One NetCDF per (variable, statistic, year) in raw/era5/, plus raw/era5/static.nc.
Re-run safely after failures: finished files are skipped, and failures are
listed in raw/era5/_failed.txt. Needs ~/.cdsapirc and the dataset licence
accepted on the CDS website.
"""
import argparse
import logging
import pathlib
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import cdsapi
import yaml

HERE = pathlib.Path(__file__).resolve().parent.parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())
OUT = HERE / "raw" / "era5"
lock = threading.Lock()


def area():
    b = CFG["regions"]["india"]
    return [b["north"], b["west"], b["south"], b["east"]]


def request(variables, stat, years, months=range(1, 13), days=range(1, 32)):
    e = CFG["era5"]
    return {
        "product_type": "reanalysis",
        "variable": variables,
        "daily_statistic": stat,
        "year": [str(y) for y in years],
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in days],
        "time_zone": e["time_zone"],
        "frequency": e["frequency"],
        "area": area(),
    }


def fetch(client, req, target):
    if target.exists() and target.stat().st_size > 0:
        return "skip"
    tmp = target.with_suffix(".part")
    client.retrieve(CFG["era5"]["dataset"], req, str(tmp))
    tmp.rename(target)
    if zipfile.is_zipfile(target):  # multi-variable requests come back as a zip of NetCDFs
        with zipfile.ZipFile(target) as z:
            z.extractall(target.with_suffix(""))
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs=2, type=int, default=CFG["era5"]["years"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(OUT / "_download.log"), logging.StreamHandler()])
    log = logging.getLogger("era5")
    client = cdsapi.Client(quiet=True, progress=False)

    jobs = [(request(CFG["era5"]["static"], "daily_mean", [2000], [1], [1]), OUT / "static.nc")]
    for var, stat, _ in CFG["era5"]["variables"]:
        for year in range(a.years[0], a.years[1] + 1):
            jobs.append((request([var], stat, [year]), OUT / f"{var}_{stat}_{year}.nc"))

    failed = []
    with ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(fetch, client, req, tgt): tgt for req, tgt in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            tgt = futs[fut]
            try:
                log.info("[%d/%d] %s %s", i, len(jobs), fut.result(), tgt.name)
            except Exception as exc:  # keep going; re-run picks up the rest
                log.error("[%d/%d] FAILED %s: %s", i, len(jobs), tgt.name, exc)
                with lock:
                    failed.append(f"{tgt.name}\t{exc}")
    (OUT / "_failed.txt").write_text("\n".join(failed) + ("\n" if failed else ""))
    log.info("done: %d ok/skipped, %d failed", len(jobs) - len(failed), len(failed))


if __name__ == "__main__":
    main()
