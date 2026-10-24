from __future__ import annotations

import csv
import gzip
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from nltk.corpus import wordnet as wn

ROOT = Path(__file__).resolve().parents[2]
ECV = ROOT / "datasets" / "signbank" / "asl.ecv"
PACK = ROOT / "gloss_to_asl" / "assets" / "signs.json.gz"
ASLLEX = ROOT / "datasets" / "asllex" / "signdata.csv"
OUT_DIR = Path(__file__).resolve().parent
REJECTS = OUT_DIR / "rejects.csv"
SYN_OUT = OUT_DIR / "proposed_synonyms.csv"
SAME_OUT = OUT_DIR / "proposed_same_sign.csv"
INFLECT_OUT = OUT_DIR / "proposed_inflections.csv"

SUFFIX_RE = re.compile(
    r"(IX|NOIX|ARC|STR|FLEX|WIG|NEUT|SYMM?|ASYM|ONEX|ROT|CK|BB|BD|BK|PA|"
    r"PRO|SUP|EL|FULL|REP|TWIST|MO|TH|NS)$"
)
NON_ALPHA = re.compile(r"[^A-Z' ]+")
STOP = {
    "A", "AN", "THE", "TO", "OF", "AND", "OR", "BE", "IS", "ARE", "AM", "WAS",
    "WERE", "BEEN", "BEING", "DO", "DOES", "DID", "HAVE", "HAS", "HAD", "NOT",
    "NO", "YES", "IN", "ON", "AT", "FOR", "WITH", "FROM", "BY", "AS", "IT",
    "ITS", "THIS", "THAT", "THESE", "THOSE", "IF", "SO", "BUT", "THAN", "THEN",
    "VERY", "REALLY", "JUST", "ALSO", "TOO", "MORE", "MOST", "SOME", "ANY",
    "OWN", "GET", "MAKE", "TAKE", "GO", "COME", "SEE", "LOOK", "PUT", "SET",
}
BLOCK = {
    "FUCK", "FUCKING", "SHIT", "DAMN", "HELL", "ASS", "SEX", "SCREW", "BONK",
    "EFF", "HUMP", "BANG", "JAZZ", "LAY", "SLEEP_WITH", "BARF", "PUKE",
    "UPCHUCK", "REGORGE", "DISGORGE", "HONK", "CHUCK", "RETCH", "SPEW",
}
BLOCK_PAIRS = {
    ("FALL", "AUTUMN"),
    ("REAR", "BACK"),
    ("MEDIUM", "AVERAGE"),
    ("FIELD", "AREA"),
    ("BRANCH", "ARM"),
    ("WEAPON", "ARM"),
    ("SLEEVE", "ARM"),
    ("BIT", "ACT"),
    ("BIT", "BITE"),
    ("PUKE", "SKUNK"),
    ("BARF", "SKUNK"),
    ("CAST", "SICK"),
    ("CHUCK", "SICK"),
}


def clean_word(t: str) -> str | None:
    t = NON_ALPHA.sub("", t.upper()).strip()
    t = re.sub(r"\s+", " ", t)
    if not t or len(t) < 2 or t in STOP or t in BLOCK:
        return None
    if any(len(p) == 1 for p in t.split()):
        return None
    if not re.fullmatch(r"[A-Z][A-Z' ]*", t):
        return None
    return t


def base_gloss(g: str) -> str:
    g = g.upper().replace("_", "-")
    parts = []
    for p in g.split("-"):
        p2 = SUFFIX_RE.sub("", p)
        p2 = re.sub(r"[0-9]+$", "", p2)
        if p2:
            parts.append(p2)
    return " ".join(parts)


def all_synsets(w: str):
    return set(wn.synsets(w.lower().replace(" ", "_")))


def primary_synsets(w: str, n: int = 1):
    return wn.synsets(w.lower().replace(" ", "_"))[:n]


def share_any(a: str, b: str) -> bool:
    sa, sb = all_synsets(a), all_synsets(b)
    return bool(sa and sb and (sa & sb))


def share_primary(a: str, b: str) -> bool:
    sa, sb = set(primary_synsets(a)), set(primary_synsets(b))
    return bool(sa and sb and (sa & sb))


BAD_SYNSET_PREFIXES = ("rotter.", "pot.n.09", "shutout.", "fuck.", "sleep_together.")


def lemmas_of_primary(sign: str) -> set[str]:
    out = set()
    for ss in primary_synsets(sign, 2):
        if ss.name().startswith(BAD_SYNSET_PREFIXES):
            continue
        related = [ss] + [r for r in ss.similar_tos()
                          if not r.name().startswith(BAD_SYNSET_PREFIXES)]
        if sign == "SICK":
            related = [s for s in wn.synsets("sick") if s.pos() in ("a", "s")][:2]
        for rel in related:
            for lem in rel.lemmas():
                name = lem.name().upper().replace("_", " ")
                cw = clean_word(name)
                if cw and " " not in cw:
                    out.add(cw)
    return out


def load_pack():
    pack = json.load(gzip.open(PACK, "rt", encoding="utf-8"))
    signs = {k.upper() for k in pack["signs"]}
    alias = {k.upper(): tuple(v) for k, v in pack["alias"].items()}
    nouns = {n.upper() for n in pack.get("nouns", [])}
    return signs, alias, nouns


def load_signbank():
    root = ET.parse(ECV).getroot()
    rows = []
    for el in root.iter("CV_ENTRY_ML"):
        gloss = ""
        desc = ""
        for c in el:
            if c.tag == "CVE_VALUE":
                gloss = (c.text or "").strip()
                desc = c.get("DESCRIPTION") or ""
        if gloss:
            rows.append((gloss, desc))
    return rows


def load_asllex_entries():
    entries = set()
    with open(ASLLEX, encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            for key in ("EntryID", "LemmaID"):
                v = (row.get(key) or "").strip().upper()
                if v:
                    entries.add(v.replace("-", " "))
    return entries


def main() -> None:
    from gloss_to_asl.aliases import inflections, NOT_INFLECTED, VERB_TOO, SAME_SIGN

    signs, alias, nouns = load_pack()
    asllex = load_asllex_entries()
    sb = load_signbank()

    rejects = []
    synonyms = []
    same_groups = defaultdict(set)
    mapped_to = {k: " ".join(v) for k, v in alias.items()}

    for group in SAME_SIGN:
        base = next((g for g in group if g in signs), None)
        if not base:
            continue
        for g in group:
            same_groups[base].add(g)

    def accept(word: str, sign: str, match: str, source: str) -> bool:
        if word == sign:
            return False
        if " " in word:
            rejects.append((word, sign, "multiword_skipped", "synonym"))
            return False
        if word in BLOCK:
            rejects.append((word, sign, "blocked_word", "synonym"))
            return False
        if (word, sign) in BLOCK_PAIRS:
            rejects.append((word, sign, "blocked_pair_ambiguous", "synonym"))
            return False
        if word in signs:
            rejects.append((word, sign, "already_own_sign", "synonym"))
            return False
        if word in mapped_to and mapped_to[word] != sign:
            rejects.append((word, sign, f"clash_maps_to_{mapped_to[word]}", "synonym"))
            return False
        if word in mapped_to and mapped_to[word] == sign:
            rejects.append((word, sign, "already_aliased", "synonym"))
            return False
        sign_ss = set(primary_synsets(sign, 2))
        word_primary = primary_synsets(word, 1)
        if not sign_ss or not word_primary or word_primary[0] not in sign_ss:
            if match.startswith("signbank") and share_any(sign, word):
                rejects.append((word, sign, "te_but_word_primary_sense_mismatch", "synonym"))
            else:
                rejects.append((word, sign, "word_primary_not_in_sign_top2", "synonym"))
            return False
        synonyms.append({
            "word": word,
            "sign": sign,
            "match": match,
            "asllex": "yes" if (word in asllex or sign in asllex) else "n/a",
            "source": source,
        })
        mapped_to[word] = sign
        same_groups[sign].add(sign)
        same_groups[sign].add(word)
        return True

    for gloss, desc in sb:
        te = [clean_word(x) for x in re.split(r"[,;/]", desc)]
        te = [x for x in te if x]
        bg = clean_word(base_gloss(gloss))
        sign = None
        for cand in ([bg] if bg else []) + te:
            if cand in signs:
                sign = cand
                break
        if not sign:
            continue
        for w in te:
            accept(w, sign, "signbank_te_shared_synset",
                   "ASL Signbank translation equivalent")

    for sign in sorted(signs):
        if " " in sign:
            continue
        for w in sorted(lemmas_of_primary(sign)):
            accept(w, sign, "wordnet_primary_lemma",
                   "WordNet primary-synset lemma of sign")

    syn_unique = []
    seen_w = set()
    for row in synonyms:
        if row["word"] in seen_w:
            continue
        seen_w.add(row["word"])
        syn_unique.append(row)

    syn_unique.sort(key=lambda r: (
        0 if r["match"].startswith("signbank") else 1,
        r["sign"], r["word"],
    ))
    selected_syn = syn_unique[:2500]

    kept_words = {r["word"] for r in selected_syn}
    for sign, group in list(same_groups.items()):
        same_groups[sign] = {g for g in group if g == sign or g in kept_words or g in alias}
        if sign in signs:
            same_groups[sign].add(sign)

    with SYN_OUT.open("w", newline="", encoding="ascii", errors="replace") as fh:
        w = csv.DictWriter(fh, fieldnames=["word", "sign", "match", "asllex", "source"])
        w.writeheader()
        for row in selected_syn:
            w.writerow(row)

    same_rows = []
    for sign, group in sorted(same_groups.items()):
        members = tuple(sorted(group, key=lambda x: (0 if x == sign else 1, x)))
        if len(members) < 2:
            continue
        same_rows.append({"base": sign, "members": "|".join(members), "count": len(members)})
    with SAME_OUT.open("w", newline="", encoding="ascii", errors="replace") as fh:
        w = csv.DictWriter(fh, fieldnames=["base", "members", "count"])
        w.writeheader()
        for row in same_rows:
            w.writerow(row)

    skip = set(signs) | set(alias) | set(mapped_to) | set(NOT_INFLECTED) | kept_words
    inflection_rows = []
    bases_for_infl = sorted(set(signs) | kept_words)
    for base in bases_for_infl:
        if " " in base or not base.isalpha():
            continue
        if base in signs:
            dest = base
            target_last = base
        else:
            dest = mapped_to.get(base)
            if not dest:
                continue
            target_last = dest
        noun_only = target_last in nouns and target_last not in VERB_TOO
        for form in inflections(base):
            if form in skip:
                continue
            tag = next((t for t in ("ING", "ED", "LY") if form.endswith(t)), "S")
            if noun_only and tag != "S":
                rejects.append((form, base, "noun_only_blocks_verb_form", "inflection"))
                continue
            if form in signs:
                rejects.append((form, base, "form_has_own_sign", "inflection"))
                continue
            inflection_rows.append({
                "form": form,
                "base": base,
                "signed_as": dest,
                "tag": tag,
            })
            skip.add(form)

    with INFLECT_OUT.open("w", newline="", encoding="ascii", errors="replace") as fh:
        w = csv.DictWriter(fh, fieldnames=["form", "base", "signed_as", "tag"])
        w.writeheader()
        for row in inflection_rows:
            w.writerow(row)

    with REJECTS.open("w", newline="", encoding="ascii", errors="replace") as fh:
        w = csv.writer(fh)
        w.writerow(["word", "intended_sign", "reason", "kind"])
        for row in rejects:
            w.writerow(row)

    same_member_count = sum(r["count"] for r in same_rows)
    same_extra = same_member_count - sum(1 for r in same_rows)
    print("synonyms selected", len(selected_syn), "of", len(syn_unique))
    print("same_sign groups", len(same_rows), "members", same_member_count)
    print("new inflections", len(inflection_rows))
    print("target3 count (same members + inflections)", same_member_count + len(inflection_rows))
    print("rejects", len(rejects))
    from collections import Counter
    print("syn match kinds", Counter(r["match"] for r in selected_syn))
    print("syn sample", selected_syn[:15])


if __name__ == "__main__":
    main()
