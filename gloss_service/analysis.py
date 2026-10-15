from __future__ import annotations

import re
from typing import Dict, List

import spacy

nlp = spacy.load("en_core_web_sm", disable=["ner"])

_WORD = re.compile(r"[A-Za-z0-9']+")
_QUOTES = str.maketrans({"‘": "'", "’": "'", "`": "'"})


def analyze(text: str) -> List[Dict[str, object]]:
    text = text.translate(_QUOTES)
    doc = nlp(text)
    out = []
    for m in _WORD.finditer(text):
        word = m.group().strip("'")
        if not word:
            continue
        start, end = m.start(), m.end()
        toks = [t for t in doc if t.idx < end and t.idx + len(t.text) > start]
        if not toks:
            out.append({"word": word.upper(), "lemma": "", "pos": "", "tag": "", "neg": False})
            continue
        head = toks[0]
        out.append({
            "word": word.upper(),
            "lemma": head.lemma_.upper(),
            "pos": head.pos_,
            "tag": head.tag_,
            "neg": any(t.dep_ == "neg" for t in toks),
        })
    return out
