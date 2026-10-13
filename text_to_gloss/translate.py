from __future__ import annotations

import re
from typing import Dict, List, Optional

_WORD = re.compile(r"[A-Za-z0-9']+")
_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "`": "'"})


def _words(text: str) -> List[re.Match]:
    return [m for m in _WORD.finditer(text.translate(_QUOTES)) if m.group().strip("'")]


def analyze(text: str, nlp=None) -> List[Dict[str, object]]:
    if nlp is None:
        from text_to_gloss.gloss import nlp
    text = text.translate(_QUOTES)
    doc = nlp(text)
    out = []
    for m in _words(text):
        start, end = m.start(), m.end()
        toks = [t for t in doc if t.idx < end and t.idx + len(t.text) > start]
        if not toks:
            out.append({"word": m.group().strip("'").upper(), "lemma": "", "pos": "", "tag": "", "neg": False})
            continue
        head = toks[0]
        out.append({
            "word": m.group().strip("'").upper(),
            "lemma": head.lemma_.upper(),
            "pos": head.pos_,
            "tag": head.tag_,
            "neg": any(t.dep_ == "neg" for t in toks),
        })
    return out


def try_analyze(text: str) -> Optional[List[Dict[str, object]]]:
    try:
        return analyze(text)
    except (ImportError, OSError):
        return None
