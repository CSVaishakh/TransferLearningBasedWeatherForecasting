"""Locations of the files the API serves (all produced by the digitization and model tracks)."""
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SJCET_DAILY = ROOT / "data" / "sjcet" / "sjcet_daily.csv"
CELLS = ROOT / "data" / "sjcet" / "cells.csv"
MODEL_DIR = ROOT / "model"
MODEL_OUT = Path(os.environ.get("SJCET_MODEL_OUT", MODEL_DIR / "outputs"))  # e.g. model/outputs_h1 to serve archived results
FORECASTS = MODEL_OUT / "forecasts"
RUNS = MODEL_OUT / "runs"

DEFAULT_MODEL = "s123_f1.0_s0"
MODEL_LABELS = {
    "s123_f1.0_s0": "GNN, Region → Stations → SJCET (1→2→3)",
    "s13_f1.0_s0": "GNN, Region → SJCET (1→3)",
    "scratch_f1.0_s0": "GNN, SJCET only (no pretraining)",
    "s123c_f1.0_s0": "GNN, Region → Stations → SJCET + SJCET climatology",
    "scratchc_f1.0_s0": "GNN, SJCET only + SJCET climatology",
}


def forecast_dir(model: str) -> Path:
    return FORECASTS if model == DEFAULT_MODEL else FORECASTS / model


def read_json(path: Path):
    return json.loads(path.read_text())


def clean(v):
    """NaN/inf -> None so the response is valid JSON."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v
