"""SJCET weather API.

    uv run uvicorn main:app --reload --port 8000

Serves files written by the other tracks (see paths.py); nothing is computed here.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes import forecast, history, results, review

app = FastAPI(title="SJCET Weather API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])

for r in (forecast, history, results, review):
    app.include_router(r.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"ok": True}
