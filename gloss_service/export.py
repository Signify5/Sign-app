from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import psycopg


def database_url() -> str:
    url = os.environ.get("DATABASE_URL")
    env = Path(__file__).with_name(".env.local")
    if not url and env.exists():
        m = re.search(r'^DATABASE_URL="?([^"\r\n]+)', env.read_text(encoding="utf-8"), re.M)
        url = m.group(1) if m else None
    if not url:
        sys.exit("No DATABASE_URL. Run `vercel env pull .env.local` in gloss_service first.")
    return url


def main() -> None:
    ap = argparse.ArgumentParser(description="Export the logged sentences")
    ap.add_argument("-o", "--out", default="sentences.csv", help="output file (.csv or .jsonl)")
    ap.add_argument("--source", choices=["voice", "typed"], help="only this source")
    ap.add_argument("--since", help="only sentences on or after this date, e.g. 2026-10-01")
    args = ap.parse_args()

    where, params = [], []
    if args.source:
        where.append("source = %s")
        params.append(args.source)
    if args.since:
        where.append("created_at >= %s::timestamptz")
        params.append(args.since)
    sql = "SELECT id, created_at, source, text, words FROM sentences"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id"

    with psycopg.connect(database_url(), connect_timeout=10) as db:
        rows = db.execute(sql, params).fetchall()

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as fh:
        if out.suffix == ".jsonl":
            for i, t, s, text, words in rows:
                fh.write(json.dumps({"id": i, "created_at": t.isoformat(), "source": s, "text": text,
                                     "words": words}, ensure_ascii=False) + "\n")
        else:
            w = csv.writer(fh)
            w.writerow(["id", "created_at", "source", "text", "words"])
            for i, t, s, text, words in rows:
                w.writerow([i, t.isoformat(), s, text, json.dumps(words, ensure_ascii=False)])
    print(f"wrote {len(rows)} sentences to {out}")


if __name__ == "__main__":
    main()
