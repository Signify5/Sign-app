from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

_PHRASES = json.loads(
    (Path(__file__).resolve().parent / "phrases.json").read_text(encoding="ascii")
)["phrases"]

_WORD = re.compile(r"[A-Za-z0-9']+")


def words(text: str) -> List[str]:
    text = text.translate({ord("\u2018"): "'", ord("\u2019"): "'", ord("`"): "'"})
    return [m.group().upper().strip("'") for m in _WORD.finditer(text) if m.group().strip("'")]


def apply_phrases(toks: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(toks):
        hit = None
        for p in _PHRASES:
            en = p["en"]
            n = len(en)
            if n and i + n <= len(toks) and toks[i:i + n] == en:
                hit = p
                break
        if hit:
            out.extend(hit["gloss"])
            i += len(hit["en"])
        else:
            out.append(toks[i])
            i += 1
    return [x for x in out if x]


def phrase_gloss(text: str) -> List[str]:
    return apply_phrases(words(text))
