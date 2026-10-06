from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests
import urllib3

from animate import ensure_models, extract_motion
from clips import CLIPS_DIR, INDEX_FILENAME, WLASL_DIR, Clip, ClipIndex

WLASL_JSON_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
WLASL_JSON_NAME = "WLASL_v0.3.json"
LETTERS_MANIFEST = Path("letters_manifest.json")
REPORT_NAME = "setup_report.json"

# YouTube needs a video extractor (yt-dlp); these hosts are skipped and reported.
SKIPPED_HOSTS = {"www.youtube.com", "youtube.com", "youtu.be", "m.youtube.com"}
USER_AGENT = "Mozilla/5.0 (signify setup_clips)"

INSECURE_HOSTS: set = set()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    verify = urlparse(url).netloc.lower() not in INSECURE_HOSTS
    with requests.get(
        url, stream=True, timeout=30, headers={"User-Agent": USER_AGENT}, verify=verify
    ) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as fh:
            for block in resp.iter_content(chunk_size=1 << 18):
                fh.write(block)
    if tmp.stat().st_size == 0:
        tmp.unlink()
        raise RuntimeError("empty response")
    tmp.replace(dest)


def _fetch_one(task: dict, clips_dir: Path) -> dict:
    """Download the source video (kept), then extract its hand motion to a .npz."""
    video = Path(task["video"])
    motion_path = clips_dir / task["motion_path"]
    try:
        if not video.is_file() or video.stat().st_size == 0:
            _download(task["url"], video)
        from motion import Motion

        motion = Motion.load(motion_path) if motion_path.is_file() else None
        if motion is None or not motion.world:  # older 2D-only files are re-extracted with 3D landmarks
            motion = extract_motion(video, task["start_ms"], task["end_ms"])
            motion.save(motion_path)
    except Exception as exc:
        motion_path.unlink(missing_ok=True)
        return {"task": task, "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}"}
    return {"task": task, "quality": motion.quality}


def _fetch_gloss(candidates: List[dict], clips_dir: Path, max_per_gloss: int) -> List[dict]:
    """Try a gloss's signings in order until `max_per_gloss` succeed; dead links fall through."""
    results, good = [], 0
    for task in candidates:
        if good >= max_per_gloss:
            break
        res = _fetch_one(task, clips_dir)
        results.append(res)
        good += "error" not in res
    return results


def video_ranges(wlasl_dir: Path = WLASL_DIR) -> dict:
    """WLASL video id -> (start_ms, end_ms) of the sign in its video, as select_candidates cuts it."""
    out = {}
    for entry in json.loads((wlasl_dir / WLASL_JSON_NAME).read_text(encoding="utf-8")):
        for inst in entry["instances"]:
            fps = inst.get("fps") or 25
            start, end = inst.get("frame_start", 1), inst.get("frame_end", -1)
            out[inst["video_id"]] = (int(max(start - 1, 0) / fps * 1000),
                                     None if end is None or end < 0 else int(end / fps * 1000))
    return out


def select_candidates(wlasl: list, max_glosses: Optional[int],
                      wlasl_dir: Path = WLASL_DIR) -> Tuple[List[List[dict]], List[dict]]:
    """(per-gloss candidate task lists, skipped-up-front) for the first `max_glosses` glosses."""
    per_gloss, skipped = [], []
    for entry in wlasl[:max_glosses]:
        gloss = entry["gloss"]
        candidates = []
        for inst in entry["instances"]:
            parsed = urlparse(inst["url"])
            if parsed.netloc.lower() in SKIPPED_HOSTS:
                skipped.append({"gloss": gloss, "url": inst["url"], "reason": "youtube host"})
                continue
            if not parsed.path.lower().endswith((".mp4", ".webm", ".mov")):
                skipped.append({"gloss": gloss, "url": inst["url"], "reason": "not a video file"})
                continue
            fps = inst.get("fps") or 25
            frame_start, frame_end = inst.get("frame_start", 1), inst.get("frame_end", -1)
            candidates.append({
                "gloss": gloss,
                "url": inst["url"],
                "video": str(wlasl_dir / gloss / f"{inst['video_id']}.mp4"),
                "motion_path": f"motion/{gloss}/{inst['video_id']}.npz",
                "clip_id": f"wlasl-{inst['video_id']}",
                "signer_id": inst.get("signer_id"),
                "start_ms": int(max(frame_start - 1, 0) / fps * 1000),
                "end_ms": None if frame_end is None or frame_end < 0 else int(frame_end / fps * 1000),
            })
        if candidates:
            per_gloss.append(candidates)
    return per_gloss, skipped


def fetch_letters(clips_dir: Path, manifest: Path) -> Tuple[Dict[str, Clip], List[dict]]:
    """Letter clips: downloads from the manifest (letter -> url), plus any clips/letters/X.mp4 dropped in."""
    failures: List[dict] = []
    letters_dir = clips_dir / "letters"
    if manifest.is_file():
        for letter, url in json.loads(manifest.read_text(encoding="utf-8")).items():
            dest = letters_dir / f"{letter.upper()}.mp4"
            if dest.is_file():
                continue
            try:
                _download(url, dest)
            except Exception as exc:
                failures.append({"letter": letter, "url": url, "error": f"{type(exc).__name__}: {exc}"})

    from motion import Motion

    letters: Dict[str, Clip] = {}
    for video in sorted(letters_dir.glob("*.mp4")) if letters_dir.is_dir() else []:
        letter = video.stem.upper()
        motion_path = clips_dir / "motion" / "letters" / f"{letter}.npz"
        try:
            motion = Motion.load(motion_path) if motion_path.is_file() else None
            if motion is None or not motion.world:
                motion = extract_motion(video)
                motion.save(motion_path)
        except Exception as exc:
            failures.append({"letter": letter, "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}"})
            continue
        letters[letter] = Clip(letter, str(motion_path), f"letter-{letter}", None, motion.quality)
    return letters, failures


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Download WLASL clips, extract hand motion, build clips/index.json")
    ap.add_argument("--clips-dir", type=Path, default=CLIPS_DIR)
    ap.add_argument("--wlasl-dir", type=Path, default=WLASL_DIR, help="where the WLASL videos and list are kept")
    ap.add_argument("--max-glosses", type=int, default=300,
                    help="first N WLASL glosses (file is ordered by frequency; 0 = all)")
    ap.add_argument("--max-per-gloss", type=int, default=3, help="signings kept per gloss")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--insecure-host", action="append", default=[], metavar="HOST",
                    help="skip TLS verification for this host only (e.g. aslsignbank.haskins.yale.edu, "
                         "whose server sends an incomplete certificate chain)")
    ap.add_argument("--letters-manifest", type=Path, default=LETTERS_MANIFEST,
                    help="optional JSON {letter: url} of fingerspelling clips (e.g. from ASL Signbank)")
    args = ap.parse_args(argv)

    INSECURE_HOSTS.update(h.lower() for h in args.insecure_host)
    if INSECURE_HOSTS:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    clips_dir: Path = args.clips_dir
    clips_dir.mkdir(parents=True, exist_ok=True)
    ensure_models()

    meta_path = args.wlasl_dir / WLASL_JSON_NAME
    if not meta_path.is_file():
        print(f"[wlasl] downloading metadata {WLASL_JSON_URL} ...")
        _download(WLASL_JSON_URL, meta_path)
    wlasl = json.loads(meta_path.read_text(encoding="utf-8"))

    per_gloss, skipped = select_candidates(wlasl, args.max_glosses or None, args.wlasl_dir)
    print(f"[wlasl] {len(per_gloss)} glosses with candidate clips ({len(skipped)} youtube/non-video skipped)")

    clips: List[Clip] = []
    failed: List[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        work = lambda cands: _fetch_gloss(cands, clips_dir, args.max_per_gloss)
        for n, results in enumerate(pool.map(work, per_gloss), 1):
            for res in results:
                task = res["task"]
                if "error" in res:
                    failed.append({"gloss": task["gloss"], "url": task["url"], "error": res["error"]})
                else:
                    clips.append(Clip(
                        gloss=task["gloss"], path=str(clips_dir / task["motion_path"]),
                        clip_id=task["clip_id"], signer_id=task["signer_id"], quality=res["quality"],
                    ))
            if n % 25 == 0 or n == len(per_gloss):
                print(f"[wlasl] {n}/{len(per_gloss)} glosses done, {len(clips)} clips, {len(failed)} failed links")

    letters, letter_failures = fetch_letters(clips_dir, args.letters_manifest)
    ClipIndex.write(clips_dir / INDEX_FILENAME, clips, letters)

    report = {
        "clips_indexed": len(clips),
        "glosses_with_clips": len({c.gloss for c in clips}),
        "letters_indexed": sorted(letters),
        "failed": failed,
        "skipped": skipped,
        "letter_failures": letter_failures,
    }
    (clips_dir / REPORT_NAME).write_text(json.dumps(report, indent=1), encoding="utf-8")

    print()
    print(f"Indexed {len(clips)} clips for {report['glosses_with_clips']} glosses "
          f"({len(failed)} dead/failed links, details in {clips_dir / REPORT_NAME}).")
    if letters:
        print(f"Letter clips: {''.join(sorted(letters))}")
    else:
        print("No dedicated letter clips. Fingerspelling falls back to one-letter WLASL glosses; "
              f"add clips/letters/A.mp4 ... or a {LETTERS_MANIFEST} manifest for full coverage.")
    if not clips:
        sys.exit(1)


if __name__ == "__main__":
    main()
