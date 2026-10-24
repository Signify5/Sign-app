from __future__ import annotations

import gzip
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "gloss_to_asl" / "assets" / "signs.json.gz"
BACKUP = ROOT / "datasets" / "expansion" / "signs.json.gz.bak"


def main() -> None:
    from gloss_to_asl.aliases import DROP, resolve_aliases

    if not BACKUP.exists():
        shutil.copy2(PACK, BACKUP)
        print("backed up to", BACKUP)

    with gzip.open(PACK, "rt", encoding="utf-8") as fh:
        lib = json.load(fh)
    before = len(lib.get("alias", {}))
    lib["alias"], notes, lib["tags"] = resolve_aliases(lib["signs"], lib.get("nouns", []))
    lib["drop"] = DROP
    with gzip.open(PACK, "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump(lib, fh, separators=(",", ":"))
    print(f"alias {before} -> {len(lib['alias'])}, tags {len(lib['tags'])}, notes {len(notes)}")
    for n in notes:
        print(" note", n)


if __name__ == "__main__":
    main()
