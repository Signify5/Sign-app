from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gloss_to_asl.clips import CLIPS_DIR, INDEX_FILENAME

EXPANSION = ROOT / "datasets" / "expansion"
VIDEOS = ROOT / "datasets" / "signbank_videos"
USER_AGENT = "Mozilla/5.0 (signify research)"


def _one(job: tuple) -> tuple:
    os.chdir(ROOT)
    from gloss_to_asl.animate import extract_motion

    word, cid, url, motion_path = job
    video = VIDEOS / f"{cid}.mp4"
    try:
        if not video.is_file() or video.stat().st_size == 0:
            video.parent.mkdir(parents=True, exist_ok=True)
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            data = urllib.request.urlopen(req, timeout=60).read()
            if not data:
                raise RuntimeError("empty response")
            video.write_bytes(data)
        m = extract_motion(video)
        Path(motion_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(motion_path)
        return word, cid, m.quality, None
    except Exception as exc:
        return word, cid, None, f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Download the approved Signbank clips, track them and add them to clips/index.json")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    checked = {r["signbank_cve_id"]: r for r in csv.DictReader((EXPANSION / "page_check.csv").open(newline="", encoding="ascii"))}
    clips_dir = ROOT / CLIPS_DIR
    path = clips_dir / INDEX_FILENAME
    data = json.loads(path.read_text(encoding="utf-8"))
    have = {c["clip_id"] for c in data["clips"]}
    jobs = []
    for r in csv.DictReader((EXPANSION / "new_words_manifest.csv").open(newline="", encoding="ascii")):
        cid = r["signbank_cve_id"]
        row = checked.get(cid)
        if not row or not row["video_url"] or f"signbank-{cid}" in have:
            continue
        gloss = r["word"].lower()
        jobs.append((gloss, cid, row["video_url"], str(clips_dir / "motion" / gloss / f"signbank-{cid}.npz")))
    if args.limit:
        jobs = jobs[:args.limit]
    added, failed = 0, []
    with ProcessPoolExecutor(args.workers) as pool:
        for n, fut in enumerate(as_completed([pool.submit(_one, j) for j in jobs]), 1):
            word, cid, q, err = fut.result()
            if err:
                failed.append((word, cid, err))
            else:
                data["clips"].append({"gloss": word, "path": f"motion/{word}/signbank-{cid}.npz",
                                      "clip_id": f"signbank-{cid}", "signer_id": None, "quality": q, "faults": 0})
                added += 1
            if n % 50 == 0:
                print(f"{n}/{len(jobs)}", flush=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"added {added} of {len(jobs)}; failed {len(failed)}")
    out = EXPANSION / "download_failed.csv"
    with out.open("w", newline="", encoding="ascii") as fh:
        w = csv.writer(fh)
        w.writerow(["word", "signbank_cve_id", "error"])
        for row in failed:
            w.writerow([row[0], row[1], row[2].encode("ascii", "replace").decode("ascii")])


if __name__ == "__main__":
    main()
