from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYN = Path(__file__).resolve().parent / "proposed_synonyms.csv"
SAME = Path(__file__).resolve().parent / "proposed_same_sign.csv"
OUT = ROOT / "gloss_to_asl" / "assets" / "vocab_expansion.json"


def main() -> None:
    syn = list(csv.DictReader(SYN.open(encoding="ascii")))
    same = list(csv.DictReader(SAME.open(encoding="ascii")))

    aliases = {r["word"]: [r["sign"]] for r in syn}
    groups = []
    for row in same:
        members = row["members"].split("|")
        if len(members) >= 2:
            groups.append(members)

    by_sign = defaultdict(set)
    for w, dest in aliases.items():
        by_sign[dest[0]].add(w)
        by_sign[dest[0]].add(dest[0])
    for members in groups:
        base = members[0]
        for m in members:
            by_sign[base].add(m)

    merged = []
    seen = set()
    for base, members in sorted(by_sign.items()):
        key = tuple(sorted(members))
        if key in seen or len(members) < 2:
            continue
        seen.add(key)
        ordered = [base] + sorted(m for m in members if m != base)
        merged.append(ordered)

    data = {
        "same_sign": merged,
        "aliases": {k: v for k, v in sorted(aliases.items())},
        "counts": {
            "synonyms": len(aliases),
            "same_sign_groups": len(merged),
            "same_sign_members": sum(len(g) for g in merged),
        },
    }
    OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="ascii")
    print("wrote", OUT)
    print(data["counts"])


if __name__ == "__main__":
    main()
