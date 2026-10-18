from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import psycopg

from export import database_url
from gloss_to_asl import aliases as A
from gloss_to_asl.library import load_pack


def words_report(out: Path, since: str | None, top: int) -> None:
    lib = load_pack()
    signs, alias, drop = set(lib["signs"]), lib["alias"], set(lib["drop"])
    known = signs | set(alias)
    maxw = max(len(k.split()) for k in known)
    sql = "SELECT text, words FROM sentences"
    params: list = []
    if since:
        sql += " WHERE created_at >= %s::timestamptz"
        params.append(since)
    with psycopg.connect(database_url(), connect_timeout=10) as db:
        rows = db.execute(sql + " ORDER BY id", params).fetchall()

    count: Counter = Counter()
    kind: dict = {}
    example: dict = {}
    for text, words in rows:
        toks = [(w["word"], w.get("lemma") or w["word"], w.get("pos", "")) for w in words]
        i = 0
        while i < len(toks):
            hit = 0
            for n in range(min(maxw, len(toks) - i), 1, -1):
                if " ".join(t[0] for t in toks[i:i + n]) in known:
                    hit = n
                    break
            if hit:
                i += hit
                continue
            word, lemma, pos = toks[i]
            i += 1
            if word in known or lemma in known or word in drop or word.isdigit():
                continue
            count[word] += 1
            kind[word] = "name" if pos == "PROPN" else "word"
            example.setdefault(word, text)

    out = Path(out)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["word", "times", "type", "example sentence", "sign it as", "notes"])
        for word, n in count.most_common(top):
            w.writerow([word.lower(), n, kind[word], example[word], "", ""])
    names = sum(1 for k in kind.values() if k == "name")
    print(f"{len(rows)} sentences, {len(count)} fingerspelled words ({names} look like names). Wrote {out}")


def alias_sheet(out: Path) -> None:
    lib = load_pack()
    signs, alias = set(lib["signs"]), lib["alias"]
    rows = []
    for group in A.SAME_SIGN:
        base = next((g for g in group if g in signs), None)
        for g in group:
            if g not in signs and base:
                rows.append((g, base, "same sign"))
    for k, v in A.ALIASES.items():
        if k in signs:
            continue
        rows.append((k, " ".join(alias[k]) if k in alias else "(no sign in the pack)",
                     "two signs" if len(v) > 1 else "same sign"))
    for k, v in A.PAST.items():
        if k in alias:
            rows.append((k, " ".join(alias[k]) + " (past)", "past tense"))
    for base in A.ADJECTIVES:
        if base in signs:
            rows.append((base.lower() + " + er / est", base + " MORE / MOST", "comparative"))
    out = Path(out)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["english word", "signed as", "kind", "correct? (yes / no)", "better sign or notes"])
        for k, v, kind in sorted(rows, key=lambda r: (r[2], r[0])):
            w.writerow([k.lower(), v, kind, "", ""])
    print(f"Wrote {len(rows)} aliases to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reports for improving the sign list")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("words", help="most common words that had no sign, from the saved sentences")
    p.add_argument("-o", "--out", default="missing_words.csv")
    p.add_argument("--since")
    p.add_argument("--top", type=int, default=200)
    p = sub.add_parser("aliases", help="a review sheet of every word we map to another sign")
    p.add_argument("-o", "--out", default="alias_review.csv")
    args = ap.parse_args()
    if args.cmd == "words":
        words_report(args.out, args.since, args.top)
    else:
        alias_sheet(args.out)


if __name__ == "__main__":
    main()
