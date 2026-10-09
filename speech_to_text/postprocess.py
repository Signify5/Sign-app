from __future__ import annotations

import re

_SENTENCE_END = (".", "!", "?")


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if not text.endswith(_SENTENCE_END):
        text += "."
    return text
