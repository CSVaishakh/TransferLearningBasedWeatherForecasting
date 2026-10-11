"""The digitized SJCET register (data-contracts §1)."""
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from paths import SJCET_DAILY, clean

router = APIRouter()
VALUE_COLS = ["rain_mm", "dry_bulb_c", "wet_bulb_c", "tmin_c", "tmax_c", "rh_pct"]
_cache: dict = {}


def load() -> pd.DataFrame:
    if not SJCET_DAILY.exists():
        raise HTTPException(404, "sjcet_daily.csv not found: run digitization/assemble.py")
    mtime = SJCET_DAILY.stat().st_mtime
    if _cache.get("mtime") != mtime:
        _cache.update(mtime=mtime, df=pd.read_csv(SJCET_DAILY, dtype={"obs_time": str}))
    return _cache["df"]


@router.get("/history")
def history(start: str | None = Query(None, alias="from"), end: str | None = Query(None, alias="to"),
            vars: str | None = None):
    df = load()
    if start:
        df = df[df.date >= start]
    if end:
        df = df[df.date <= end]
    cols = ["date"] + ([v for v in vars.split(",") if v in VALUE_COLS] if vars else VALUE_COLS) \
        + ["obs_time", "source_page", "qc_flag", "qc_note"]
    return [{k: clean(v) for k, v in r.items()} for r in df[cols].to_dict("records")]
