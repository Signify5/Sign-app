from __future__ import annotations

import csv
import gzip
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from nltk.corpus import wordnet as wn

ROOT = Path(__file__).resolve().parents[2]
ECV = ROOT / "datasets" / "signbank" / "asl.ecv"
PACK = ROOT / "gloss_to_asl" / "assets" / "signs.json.gz"
WLASL = ROOT / "datasets" / "wlasl" / "WLASL_v0.3.json"
OUT = Path(__file__).resolve().parent / "new_words_manifest.csv"

SUFFIX_RE = re.compile(
    r"(IX|NOIX|ARC|STR|FLEX|WIG|NEUT|SYMM?|ASYM|ONEX|ROT|CK|BB|BD|BK|PA|"
    r"PRO|SUP|EL|FULL|REP|TWIST|MO|TH|NS)$"
)
NON_ALPHA = re.compile(r"[^A-Z' ]+")
LETTERISH = re.compile(r"^(?:[A-Z] )+[A-Z]$")
STOP = {
    "A", "AN", "THE", "TO", "OF", "AND", "OR", "BE", "IS", "ARE", "AM", "WAS",
    "WERE", "BEEN", "BEING", "DO", "DOES", "DID", "HAVE", "HAS", "HAD", "NOT",
    "NO", "YES", "IN", "ON", "AT", "FOR", "WITH", "FROM", "BY", "AS", "IT",
    "ITS", "THIS", "THAT", "THESE", "THOSE", "IF", "SO", "BUT", "THAN", "THEN",
}


def base_gloss(g: str) -> str:
    g = g.upper().replace("_", "-")
    parts = g.split("-")
    cleaned = []
    for p in parts:
        p2 = SUFFIX_RE.sub("", p)
        p2 = re.sub(r"[0-9]+$", "", p2)
        if p2:
            cleaned.append(p2)
    return " ".join(cleaned)


def clean_word(t: str) -> str | None:
    t = NON_ALPHA.sub("", t.upper()).strip()
    t = re.sub(r"\s+", " ", t)
    if not t or len(t) < 2 or t in STOP:
        return None
    if LETTERISH.match(t):
        return None
    if not re.fullmatch(r"[A-Z][A-Z' ]*", t):
        return None
    if any(len(p) == 1 for p in t.split()):
        return None
    return t


def in_wordnet(w: str) -> bool:
    key = w.lower().replace(" ", "_")
    return bool(wn.synsets(key))


def load_signbank():
    root = ET.parse(ECV).getroot()
    rows = []
    for el in root.iter("CV_ENTRY_ML"):
        cid = el.get("CVE_ID") or ""
        gloss = ""
        desc = ""
        for c in el:
            if c.tag == "CVE_VALUE":
                gloss = (c.text or "").strip()
                desc = c.get("DESCRIPTION") or ""
        if gloss and cid:
            rows.append((cid, gloss, desc))
    return rows


def load_known():
    pack = json.load(gzip.open(PACK, "rt", encoding="utf-8"))
    signs = {k.upper() for k in pack["signs"]}
    known = signs | {k.upper() for k in pack["alias"]} | {k.upper() for k in pack["drop"]}
    return signs, known


def main() -> None:
    signs, known = load_known()
    sb = load_signbank()
    wlasl = {e["gloss"].upper(): e for e in json.load(open(WLASL, encoding="utf-8"))}

    candidates = []
    seen = set()

    for cid, gloss, desc in sb:
        te = [clean_word(x) for x in re.split(r"[,;/]", desc)]
        te = [x for x in te if x]
        bg = clean_word(base_gloss(gloss))
        ranked = []
        for w in te:
            if w not in ranked:
                ranked.append(w)
        if bg and bg not in ranked:
            ranked.append(bg)
        for i, w in enumerate(ranked):
            if w in known or w in seen:
                continue
            if " " in w and w.count(" ") > 1:
                continue
            if not in_wordnet(w) and i > 0:
                continue
            seen.add(w)
            conf = "high" if in_wordnet(w) and (w in te[:2] or w == bg) else (
                "high" if in_wordnet(w) else "medium"
            )
            candidates.append({
                "word": w,
                "source": "ASL Signbank",
                "clip_id_or_url": f"https://aslsignbank.com/dictionary/gloss/{cid}.html",
                "license_or_terms": "CC BY-NC-SA 4.0 (aslsignbank.com/about/conditions)",
                "clip_available": "yes",
                "confidence": conf,
                "signbank_id_gloss": gloss,
                "signbank_cve_id": cid,
            })

    for gloss, entry in wlasl.items():
        g = gloss.upper()
        if g in known or g in seen or len(g) == 1:
            continue
        instances = entry.get("instances") or []
        url = ""
        for inst in instances:
            u = inst.get("url") or ""
            if u and not u.endswith(".swf") and "youtube.com" not in u.lower():
                url = u
                break
        if not url and instances:
            url = instances[0].get("url") or instances[0].get("video_id") or f"wlasl:{g}"
        seen.add(g)
        candidates.append({
            "word": g,
            "source": "WLASL",
            "clip_id_or_url": url,
            "license_or_terms": "C-UDA 1.0; academic non-commercial (github.com/dxli94/WLASL)",
            "clip_available": "yes" if url.startswith("http") else "unknown",
            "confidence": "high" if url.startswith("http") else "low",
            "signbank_id_gloss": "",
            "signbank_cve_id": "",
        })

    candidates.sort(key=lambda r: (
        0 if r["source"] == "ASL Signbank" else 1,
        0 if r["confidence"] == "high" else 1,
        0 if " " not in r["word"] else 1,
        r["word"],
    ))
    selected = candidates[:1000]

    fields = ["word", "source", "clip_id_or_url", "license_or_terms",
              "clip_available", "confidence", "signbank_id_gloss", "signbank_cve_id"]
    with OUT.open("w", newline="", encoding="ascii", errors="replace") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in selected:
            w.writerow({k: str(row.get(k, "")).encode("ascii", "replace").decode("ascii")
                        for k in fields})

    by_src = {}
    by_conf = {}
    for r in selected:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
        by_conf[r["confidence"]] = by_conf.get(r["confidence"], 0) + 1
    print(f"wrote {len(selected)} / pool {len(candidates)}")
    print("by source", by_src)
    print("by confidence", by_conf)
    print("sample", [r["word"] for r in selected[:25]])

    # sample-check a few URLs
    import urllib.request, ssl
    ctx = ssl.create_default_context()
    ok = 0
    for r in selected[:5]:
        try:
            resp = urllib.request.urlopen(r["clip_id_or_url"], context=ctx, timeout=15)
            ok += 1 if resp.status == 200 else 0
            print("check", resp.status, r["word"], r["clip_id_or_url"])
        except Exception as e:
            print("check FAIL", r["word"], e)


if __name__ == "__main__":
    main()
