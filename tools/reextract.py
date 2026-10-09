"""Re-track every indexed clip from its kept video (after a change to animate.py) and refresh the index.

    python tools/reextract.py [--workers 12]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from clips import CLIPS_DIR, INDEX_FILENAME, WLASL_DIR, Clip, ClipIndex  # noqa: E402
from setup_clips import video_ranges  # noqa: E402


def _one(job: tuple) -> tuple:
    import os
    os.chdir(ROOT)
    from animate import extract_motion

    motion_path, video, start, end = job
    try:
        m = extract_motion(video, start, end)
        m.save(motion_path)
        return motion_path, m.quality, None
    except Exception as exc:
        return motion_path, None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    clips_dir = ROOT / CLIPS_DIR
    data = json.loads((clips_dir / INDEX_FILENAME).read_text(encoding="utf-8"))
    ranges = video_ranges(ROOT / WLASL_DIR)
    jobs = []
    for c in data["clips"]:
        vid = c["clip_id"].split("-", 1)[1]
        video = ROOT / WLASL_DIR / c["gloss"] / f"{vid}.mp4"
        if video.is_file():
            jobs.append((str(clips_dir / c["path"]), str(video), *ranges[vid]))
    quality, failed = {}, []
    with ProcessPoolExecutor(args.workers) as pool:
        for n, fut in enumerate(as_completed([pool.submit(_one, j) for j in jobs]), 1):
            path, q, err = fut.result()
            if err:
                failed.append((path, err))
            else:
                quality[path] = q
            if n % 100 == 0:
                print(f"{n}/{len(jobs)}", flush=True)
    for c in data["clips"]:
        q = quality.get(str(clips_dir / c["path"]))
        if q is not None:
            c["quality"] = q
    clips = [Clip(c["gloss"], str(clips_dir / c["path"]), c["clip_id"], c.get("signer_id"), c["quality"])
             for c in data["clips"]]
    letters = {k: Clip(k, str(clips_dir / v["path"]), v["clip_id"], v.get("signer_id"), v["quality"])
               for k, v in data.get("letters", {}).items()}
    ClipIndex.write(clips_dir / INDEX_FILENAME, clips, letters)
    print(f"re-tracked {len(quality)} of {len(jobs)}; failed {len(failed)}")
    for path, err in failed:
        print("failed:", path, err)


if __name__ == "__main__":
    main()
