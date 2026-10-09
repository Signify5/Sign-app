"""Write gloss_to_asl/assets/hand/nouns.txt: the signed glosses that are mainly nouns in English.

The page uses it to tell a plural noun (CATS, signed twice) from a verb with -s (WANTS, signed once).
A gloss counts as a noun when WordNet's tagged sense counts favour its noun senses over its verb senses
(ties go to whichever has more senses). Needs nltk with the wordnet corpus, only to regenerate the file:

    pip install nltk && python -c "import nltk; nltk.download('wordnet')"
    python gloss_to_asl/tools/build_nouns.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # the repo root
sys.path.insert(0, str(ROOT))

from nltk.corpus import wordnet as wn  # noqa: E402

from gloss_to_asl.clips import ClipIndex  # noqa: E402
from gloss_to_asl.library import vocabulary  # noqa: E402

OUT = ROOT / "gloss_to_asl" / "assets" / "hand" / "nouns.txt"


def is_noun(word: str) -> bool:
    w = word.lower()
    score = {}
    for pos in (wn.NOUN, wn.VERB):
        lemmas = [l for s in wn.synsets(w, pos) for l in s.lemmas() if l.name().lower() == w]
        score[pos] = (sum(l.count() for l in lemmas), len(lemmas))
    return score[wn.NOUN] > score[wn.VERB]


def main() -> None:
    glosses = sorted(g for g in vocabulary(ClipIndex.load()) if " " not in g and g.isalpha())
    nouns = [g for g in glosses if is_noun(g)]
    OUT.write_text("\n".join(nouns) + "\n", encoding="utf-8")
    print(f"{len(nouns)} of {len(glosses)} glosses are nouns -> {OUT}")


if __name__ == "__main__":
    main()
