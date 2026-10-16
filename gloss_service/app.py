from __future__ import annotations

import json
import os
from typing import Dict, List

import psycopg

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from analysis import analyze

ORIGINS = [o.strip() for o in os.environ.get(
    "ALLOWED_ORIGINS", "https://signify5.github.io,http://localhost:8000,http://localhost:8771").split(",") if o.strip()]
MAX_CHARS = 400
DATABASE_URL = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
SOURCES = {"voice", "typed"}
SCHEMA = """CREATE TABLE IF NOT EXISTS sentences (
    id bigserial PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT now(),
    source text NOT NULL,
    text text NOT NULL,
    words jsonb NOT NULL
)"""

app = FastAPI(title="Signify gloss service")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["POST"], allow_headers=["content-type"])


class Sentence(BaseModel):
    text: str
    source: str = "typed"


def record(text: str, source: str, words: List[Dict[str, object]]) -> None:
    if not DATABASE_URL:
        return
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=3, autocommit=True) as db:
            db.execute(SCHEMA)
            db.execute("INSERT INTO sentences (source, text, words) VALUES (%s, %s, %s::jsonb)",
                       (source, text, json.dumps(words)))
    except Exception as e:
        print("log failed:", type(e).__name__)


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
    words = analyze(text)
    record(text, body.source if body.source in SOURCES else "typed", words)
    return {"words": words}
