"""Replay forecasts (data-contracts §4), precomputed by model/predict.py --export."""
from datetime import date, timedelta

from fastapi import APIRouter, HTTPException, Query

from paths import DEFAULT_MODEL, MODEL_LABELS, forecast_dir, read_json

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
