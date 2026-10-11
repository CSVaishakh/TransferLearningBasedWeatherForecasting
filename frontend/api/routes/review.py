"""Review tool queue. The 2010-2016 register was transcribed by hand, so there are no OCR'd
cells yet; the queue becomes available once digitization writes data/sjcet/cells.csv."""
import pandas as pd
from fastapi import APIRouter

from paths import CELLS

router = APIRouter()


@router.get("/review/status")
def status():
    if not CELLS.exists():
        return {"available": False, "cells": 0,
                "reason": "No OCR'd cells yet: the 2010–2016 register was transcribed by hand "
                          "(digitization/work/transcripts). The queue opens when cells.csv exists."}
    df = pd.read_csv(CELLS)
    todo = df[(df.qc_flag != "ok") | (df.ocr_conf < 0.7)]
    return {"available": True, "cells": int(len(df)), "queue": int(len(todo))}
