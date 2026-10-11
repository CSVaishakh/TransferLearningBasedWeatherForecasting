"""Locations of the files the API serves (all produced by the digitization and model tracks)."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SJCET_DAILY = ROOT / "data" / "sjcet" / "sjcet_daily.csv"
CELLS = ROOT / "data" / "sjcet" / "cells.csv"
MODEL_OUT = ROOT / "model" / "outputs"
FORECASTS = MODEL_OUT / "forecasts"
RUNS = MODEL_OUT / "runs"

DEFAULT_MODEL = "s123_f1.0_s0"
MODEL_LABELS = {
    "s123_f1.0_s0": "GNN pretrained 1→2→3",
    "s13_f1.0_s0": "GNN pretrained 1→3",
    "scratch_f1.0_s0": "GNN from scratch (SJCET only)",
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
