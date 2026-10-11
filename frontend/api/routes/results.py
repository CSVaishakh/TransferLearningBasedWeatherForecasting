"""Model results: metrics (§5), graph (§6) and test-period predictions."""
import numpy as np
from fastapi import APIRouter, HTTPException

from paths import DEFAULT_MODEL, MODEL_LABELS, MODEL_OUT, RUNS, clean, read_json

router = APIRouter()
TARGETS = ["tmax_c", "tmin_c", "rain_mm", "tmean_c", "dewpoint_c"]  # column order of the pred/obs arrays


def _file(name):
    f = MODEL_OUT / name
    if not f.exists():
        raise HTTPException(404, f"{name} not found: run model/export.py")
    return f


@router.get("/metrics")
def metrics():
    return read_json(_file("metrics.json"))


@router.get("/graph")
def graph():
    return read_json(_file("graph.json"))


@router.get("/results/test")
def test_predictions(model: str = DEFAULT_MODEL):
    """Next-day predictions vs observations at SJCET for the test period."""
    f = RUNS / model / "local_test_pred.npz"
    if model not in MODEL_LABELS or not f.exists():
        raise HTTPException(404, f"no test predictions for {model!r}")
    z = np.load(f, allow_pickle=True)
    pred, obs, prob = z["pred"], z["obs"], z["rain_prob"]
    return [{"date": str(d), "rain_prob": clean(round(float(p), 3)),
             **{f"pred_{t}": clean(round(float(pred[i, j]), 2)) for j, t in enumerate(TARGETS)},
             **{f"obs_{t}": clean(round(float(obs[i, j]), 2)) for j, t in enumerate(TARGETS)}}
            for i, (d, p) in enumerate(zip(z["dates"], prob))]
