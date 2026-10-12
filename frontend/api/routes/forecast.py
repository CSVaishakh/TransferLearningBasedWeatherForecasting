"""Forecasts (data-contracts §4): replay forecasts precomputed by model/predict.py --export,
and the live forecast written by model/live.py."""
import subprocess
import threading
from datetime import date, timedelta

from fastapi import APIRouter, HTTPException, Query

from paths import DEFAULT_MODEL, FORECASTS, MODEL_DIR, MODEL_LABELS, forecast_dir, read_json

router = APIRouter()


def _check_model(model: str):
    if model not in MODEL_LABELS or not (forecast_dir(model) / "index.json").exists():
        raise HTTPException(404, f"no forecasts for model {model!r}")


@router.get("/models")
def models():
    return [{"id": m, "label": label, "default": m == DEFAULT_MODEL}
            for m, label in MODEL_LABELS.items() if (forecast_dir(m) / "index.json").exists()]


@router.get("/forecast/dates")
def dates(model: str = DEFAULT_MODEL):
    _check_model(model)
    return read_json(forecast_dir(model) / "index.json")


@router.get("/forecast")
def forecast(date: str = Query(..., pattern=r"^\d{4}-\d{2}-\d{2}$"), model: str = DEFAULT_MODEL):
    _check_model(model)
    f = forecast_dir(model) / f"{date}.json"
    if not f.exists():
        raise HTTPException(404, f"no forecast issued on {date}")
    return read_json(f)


@router.get("/forecast/range")
def forecast_range(start: str = Query(..., alias="from"), end: str = Query(..., alias="to"),
                   model: str = DEFAULT_MODEL):
    """Every forecast issued between `from` and `to` (inclusive); missing issue dates are skipped."""
    _check_model(model)
    d, stop, out = date.fromisoformat(start), date.fromisoformat(end), []
    if (stop - d).days > 400:
        raise HTTPException(400, "range too long (max 400 days)")
    while d <= stop:
        f = forecast_dir(model) / f"{d.isoformat()}.json"
        if f.exists():
            out.append(read_json(f))
        d += timedelta(days=1)
    return out


_live_lock = threading.Lock()


@router.get("/forecast/live")
def live():
    f = FORECASTS / "live.json"
    if not f.exists():
        raise HTTPException(404, "no live forecast yet: run model/live.py or POST /api/forecast/live/refresh")
    return read_json(f)


@router.post("/forecast/live/refresh")
def refresh_live():
    """Fetch the latest NASA POWER/NOAA data and rerun the model (about 1-2 minutes)."""
    if not _live_lock.acquire(blocking=False):
        raise HTTPException(409, "a refresh is already running")
    try:
        r = subprocess.run(["uv", "run", "python", "-W", "ignore", "live.py"], cwd=MODEL_DIR,
                           capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        raise HTTPException(504, "refresh timed out")
    finally:
        _live_lock.release()
    if r.returncode != 0:
        raise HTTPException(500, f"live.py failed: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}")
    return live()
