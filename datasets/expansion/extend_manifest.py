from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_manifest as bm
from nltk.corpus import wordnet as wn

MANIFEST = HERE / "new_words_manifest.csv"
REMOVED = HERE / "manifest_removed.csv"
ORIGINAL = HERE / "new_words_manifest_original.csv"
FIELDS = ["word", "source", "clip_id_or_url", "license_or_terms",
          "clip_available", "confidence", "signbank_id_gloss", "signbank_cve_id"]


def read(path):
    with path.open(newline="", encoding="ascii") as fh:
        return list(csv.DictReader(fh))


def base_of(w):
    if " " in w:
        return None
    b = wn.morphy(w.lower())
    return b.upper() if b and b.upper() != w else None


def main():
    signs, known = bm.load_known()
    kept = read(MANIFEST)
    done = {r["word"] for r in kept} | {r["word"] for r in read(REMOVED)} | {r["word"] for r in read(ORIGINAL)}
    taken_signs = {r["signbank_cve_id"] for r in kept}
    taken_glosses = {r["signbank_id_gloss"] for r in kept}
    words = {r["word"] for r in kept} | known
    seen = set(done)
    new, removed = [], []

    for cid, gloss, desc in bm.load_signbank():
        te = [bm.clean_word(x) for x in re.split(r"[,;/]", desc)]
        te = [x for x in te if x]
        bg = bm.clean_word(bm.base_gloss(gloss))
        ranked = []
        for w in te:
            if w not in ranked:
                ranked.append(w)
        if bg and bg not in ranked:
            ranked.append(bg)
        for i, w in enumerate(ranked):
            if w in known or w in seen or w.count(" ") > 1:
                continue
            if w != bg or not bm.in_wordnet(w):
                continue
            seen.add(w)
            conf = "high"
            row = {
                "word": w, "source": "ASL Signbank",
                "clip_id_or_url": f"https://aslsignbank.com/dictionary/gloss/{cid}.html",
                "license_or_terms": "CC BY-NC-SA 4.0 (aslsignbank.com/about/conditions)",
                "clip_available": "yes", "confidence": conf,
                "signbank_id_gloss": gloss, "signbank_cve_id": cid,
            }
            b = base_of(w)
            if b and b in words:
                removed.append((w, gloss, "form_of_known_word"))
                continue
            if cid in taken_signs or gloss in taken_glosses:
                removed.append((w, gloss, "shares_sign_with_other_word"))
                continue
            taken_signs.add(cid)
            taken_glosses.add(gloss)
            words.add(w)
            new.append(row)

    new.sort(key=lambda r: (0 if r["confidence"] == "high" else 1, " " in r["word"], r["word"]))
    with MANIFEST.open("a", newline="", encoding="ascii") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        for r in new:
            w.writerow(r)
    with REMOVED.open("a", newline="", encoding="ascii") as fh:
        w = csv.writer(fh)
        for r in removed:
            w.writerow(r)
    print("added", len(new), "removed", len(removed))
    from collections import Counter
    print(Counter(r["word"][0] for r in new))
    print(Counter(r["confidence"] for r in new))


if __name__ == "__main__":
    main()
