from __future__ import annotations

import os
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from analysis import analyze

ORIGINS = [o.strip() for o in os.environ.get(
    "ALLOWED_ORIGINS", "https://signify5.github.io,http://localhost:8000,http://localhost:8771").split(",") if o.strip()]
MAX_CHARS = 400

app = FastAPI(title="Signify gloss service")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["POST"], allow_headers=["content-type"])


class Sentence(BaseModel):
    text: str


@app.get("/health")
def health() -> Dict[str, bool]:
    return {"ok": True}


@app.post("/analyze")
def analyze_text(body: Sentence) -> Dict[str, List[Dict[str, object]]]:
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text is empty")
    if len(text) > MAX_CHARS:
        raise HTTPException(status_code=413, detail="text too long")
    return {"words": analyze(text)}
