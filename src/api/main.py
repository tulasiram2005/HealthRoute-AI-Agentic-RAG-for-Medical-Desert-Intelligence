"""FastAPI app."""

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import DATA_PATH, router

app = FastAPI(title="Virtue Foundation IDP Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.middleware("http")
async def request_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time"] = str(round(time.perf_counter() - start, 4))
    return response


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "dataset_loaded": DATA_PATH.exists(), "agent_ready": True}

