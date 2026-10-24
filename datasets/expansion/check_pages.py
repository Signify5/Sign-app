from __future__ import annotations

import csv
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "new_words_manifest.csv"
OUT = HERE / "page_check.csv"
VIDEO = re.compile(r"['\"](/dictionary/protected_media/glossvideo/[^'\"]+\.mp4)['\"]")


def check(row):
    req = urllib.request.Request(row["clip_id_or_url"], headers={"User-Agent": "Mozilla/5.0 (signify research)"})
    for attempt in range(3):
        try:
            text = urllib.request.urlopen(req, timeout=30).read().decode("utf8", "ignore")
            m = VIDEO.search(text)
            return row["word"], row["signbank_cve_id"], ("https://aslsignbank.com" + m.group(1)) if m else "", "ok"
        except Exception as exc:
            err = type(exc).__name__
    return row["word"], row["signbank_cve_id"], "", err


def main():
    rows = list(csv.DictReader(MANIFEST.open(newline="", encoding="ascii")))
    with ThreadPoolExecutor(6) as pool:
        res = list(pool.map(check, rows))
    with OUT.open("w", newline="", encoding="ascii") as fh:
        w = csv.writer(fh)
        w.writerow(["word", "signbank_cve_id", "video_url", "status"])
        w.writerows(res)
    print("pages", len(res), "with video", sum(1 for r in res if r[2]), "errors", sum(1 for r in res if r[3] != "ok"))


if __name__ == "__main__":
    main()
