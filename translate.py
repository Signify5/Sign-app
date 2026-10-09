"""English -> ASL signs: step 2's spaCy reading of the sentence feeding step 3's sign rules (gloss.js).

spaCy (loaded by gloss.py) knows what each word is in this sentence: its dictionary form (WENT -> go,
CHILDREN -> child), whether it is a noun or a verb here (my BOOKS vs he BOOKS a room) and its tense.
gloss.js knows the ASL side: which signs exist, sign order (time first, WH-sign last), FINISH for past,
plurals signed twice. `analyze` lines spaCy's tokens up with the words gloss.js reads, one entry per word, and
the page passes them to toGloss. gloss.js still works on its own (the shareable page) without them.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

# The words gloss.js reads: its words() keeps letters, digits and apostrophes and strips quotes at the ends.
_WORD = re.compile(r"[A-Za-z0-9']+")
_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "`": "'"})


def _words(text: str) -> List[re.Match]:
    return [m for m in _WORD.finditer(text.translate(_QUOTES)) if m.group().strip("'")]


def analyze(text: str, nlp=None) -> List[Dict[str, object]]:
    """One entry per word gloss.js will read, in order: {word, lemma, pos, tag, neg}.

    `pos` and `tag` are spaCy's (NOUN / VERB / ADJ ..., NNS / VBD / VBN ...) for the word's first token;
    `neg` is true when the word holds a negation (DON'T, CAN'T). `nlp` defaults to gloss.py's model.
    """
    if nlp is None:
        from gloss import nlp
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
    """analyze, or None when spaCy or its English model is not installed (gloss.js then works alone)."""
    try:
        return analyze(text)
    except (ImportError, OSError):
        return None
