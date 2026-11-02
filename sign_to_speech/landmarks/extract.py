"""Ensure per-clip landmark sequences and run feature normalization.

Prefers existing clips/motion/*.npz and clips/pose/*.npy from the dataset
manifest. Only runs MediaPipe when --allow-video is set and a clip has no
motion cache. Does not touch datasets/ downloads; reads video paths from the
manifest when extracting.

Examples:
  # Smoke a handful of cached clips (no MediaPipe):
  venv/Scripts/python.exe -m sign_to_speech.landmarks.extract --limit 8

  # Parallel full pass (default 20 workers); resumable, parent writes report:
  venv/Scripts/python.exe -m sign_to_speech.landmarks.extract --allow-video --write-cache

  # Later, after download + manifest rebuild, full pass with video fallback:
  venv/Scripts/python.exe -m sign_to_speech.landmarks.extract --allow-video
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gloss_to_asl.motion import HANDS, Motion  # noqa: E402
from sign_to_speech.landmarks.normalize import (  # noqa: E402
    FEAT_DIM,
    NORM_VERSION,
    SEQ_LEN,
    NormalizeError,
    feature_layout,
    normalize_sequence,
)

DEFAULT_LEXICAL = ROOT / "sign_to_speech" / "dataset" / "lexical_manifest.json"
DEFAULT_LETTERS = ROOT / "sign_to_speech" / "dataset" / "letters_manifest.json"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
FEATURES_DIR = Path(__file__).resolve().parent / "features"
REPORT_PATH = Path(__file__).resolve().parent / "extract_report.json"

# Per-worker MediaPipe landmarker handles (set in _worker_init).
_WORKER_HAND = None
_WORKER_POSE = None
_WORKER_ALLOW_VIDEO = False


def _load_manifest(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("clips", []))


def _resolve(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    p = ROOT / path_str
    return p if p.is_file() else None


def _path_for_report(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT).as_posix())
    except ValueError:
        return str(resolved.as_posix())


def _limit_native_threads() -> None:
    """Keep each worker single-threaded so N processes do not oversubscribe cores."""
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
    ):
        os.environ[key] = "1"
    try:
        import cv2

        cv2.setNumThreads(1)
    except Exception:  # noqa: BLE001
        pass
    try:
        import torch

        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            torch.set_num_interop_threads(1)
    except Exception:  # noqa: BLE001
        pass


def _mp_base_options(model_asset_path: str):
    """CPU delegate; pass num_threads=1 when the installed MediaPipe API supports it."""
    from mediapipe.tasks.python.core import base_options as bo

    kwargs: Dict[str, Any] = {
        "model_asset_path": model_asset_path,
        "delegate": bo.BaseOptions.Delegate.CPU,
    }
    try:
        return bo.BaseOptions(**kwargs, num_threads=1)
    except TypeError:
        return bo.BaseOptions(**kwargs)


def _create_worker_landmarkers() -> None:
    """Build one Hand + one Pose landmarker for this worker process."""
    global _WORKER_HAND, _WORKER_POSE
    from mediapipe.tasks.python import vision

    from gloss_to_asl.animate import (
        HAND_MODEL_PATH,
        HAND_MODEL_URL,
        POSE_MODEL_PATH,
        POSE_MODEL_URL,
        _ensure,
        ensure_models,
    )

    ensure_models()
    hand_path = str(_ensure(HAND_MODEL_PATH, HAND_MODEL_URL))
    pose_path = str(_ensure(POSE_MODEL_PATH, POSE_MODEL_URL))

    if _WORKER_HAND is not None:
        try:
            _WORKER_HAND.close()
        except Exception:  # noqa: BLE001
            pass
        _WORKER_HAND = None
    if _WORKER_POSE is not None:
        try:
            _WORKER_POSE.close()
        except Exception:  # noqa: BLE001
            pass
        _WORKER_POSE = None

    _WORKER_HAND = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=_mp_base_options(hand_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    )
    _WORKER_POSE = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=_mp_base_options(pose_path),
            running_mode=vision.RunningMode.VIDEO,
        )
    )


def _worker_init(allow_video: bool) -> None:
    """ProcessPoolExecutor initializer: thread caps + optional MediaPipe instances."""
    global _WORKER_ALLOW_VIDEO
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    _limit_native_threads()
    _WORKER_ALLOW_VIDEO = bool(allow_video)
    if allow_video:
        _create_worker_landmarkers()


def load_cached_landmarks(clip: dict) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[np.ndarray], float, str]:
    """Load hands/world/pose from clips/ cache. Raises FileNotFoundError if motion missing."""
    motion_path = _resolve(clip.get("motion_path"))
    if motion_path is None:
        raise FileNotFoundError("no_motion_cache")
    motion = Motion.load(motion_path)
    hands = {h: motion.hands[h].astype(np.float32) for h in HANDS if h in motion.hands}
    world = {h: motion.world[h].astype(np.float32) for h in HANDS if h in motion.world}
    pose = None
    pose_path = _resolve(clip.get("pose_path"))
    if pose_path is not None:
        pose = np.load(pose_path).astype(np.float32)
        # Expect (T, 2, 2); if lengths differ, truncate to min.
        t = min(motion.frames, pose.shape[0])
        hands = {h: a[:t] for h, a in hands.items()}
        world = {h: a[:t] for h, a in world.items()}
        pose = pose[:t]
    return hands, world, pose, float(motion.fps), str(motion_path.relative_to(ROOT).as_posix())


def extract_from_video(video_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[np.ndarray], float]:
    """Run MediaPipe Hands+Pose via gloss_to_asl.animate (same models as the site).

    Pool workers created with --allow-video already hold one Hand+Pose landmarker
    pair from `_worker_init` (CPU delegate, single-thread env). Tracking itself
    still goes through animate so outputs stay identical to the serial path;
    animate opens short-lived VIDEO sessions per clip (timestamp reset).
    """
    from gloss_to_asl.animate import arm_track, extract_motion

    # Touch worker handles so an uninitialized --allow-video worker fails loudly.
    if _WORKER_ALLOW_VIDEO and (_WORKER_HAND is None or _WORKER_POSE is None):
        raise RuntimeError("MediaPipe worker landmarkers were not created in initializer")

    motion = extract_motion(video_path)
    hands = {h: motion.hands[h].astype(np.float32) for h in HANDS if h in motion.hands}
    world = {h: motion.world[h].astype(np.float32) for h in HANDS if h in motion.world}
    pose = None
    try:
        pose = arm_track(video_path).astype(np.float32)
        t = min(motion.frames, pose.shape[0])
        hands = {h: a[:t] for h, a in hands.items()}
        world = {h: a[:t] for h, a in world.items()}
        pose = pose[:t]
    except Exception:
        pose = None
    return hands, world, pose, float(motion.fps)


def save_landmark_cache(
    out_path: Path,
    hands: Dict[str, np.ndarray],
    world: Dict[str, np.ndarray],
    pose: Optional[np.ndarray],
    fps: float,
    clip_id: str,
    source: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "fps": np.float32(fps),
        "clip_id": np.asarray(clip_id),
        "source": np.asarray(source),
        "norm_space": np.asarray("shoulder_v1"),
        "norm_version": np.asarray(NORM_VERSION),
    }
    for h in HANDS:
        if h in hands:
            payload[h] = hands[h].astype(np.float32)
        if h in world:
            payload[f"{h}_world"] = world[h].astype(np.float32)
    if pose is not None:
        payload["pose_wrists"] = pose.astype(np.float32)
    np.savez_compressed(out_path, **payload)


def features_path_for(features_dir: Path, clip_id: str) -> Path:
    return features_dir / f"{clip_id}.npy"


def features_file_valid(path: Path) -> bool:
    """True if path is a loadable (SEQ_LEN, FEAT_DIM) float feature array."""
    if not path.is_file() or path.stat().st_size < 128:
        return False
    try:
        arr = np.load(path, mmap_mode="r")
        return (
            isinstance(arr, np.ndarray)
            and arr.shape == (SEQ_LEN, FEAT_DIM)
            and arr.dtype == np.float32
        )
    except Exception:  # noqa: BLE001
        return False


def skipped_feature_row(clip: dict, features_dir: Path) -> dict:
    clip_id = clip["clip_id"]
    feat_path = features_path_for(features_dir, clip_id)
    shape = None
    try:
        arr = np.load(feat_path, mmap_mode="r")
        shape = list(arr.shape)
    except Exception:  # noqa: BLE001
        shape = [SEQ_LEN, FEAT_DIM]
    return {
        "clip_id": clip_id,
        "gloss": clip.get("gloss"),
        "split": clip.get("split"),
        "ok": True,
        "source": "existing_features",
        "error": None,
        "frames_in": None,
        "feat_shape": shape,
        "features_path": _path_for_report(feat_path),
        "skipped": True,
    }


def process_clip(
    clip: dict,
    *,
    allow_video: bool,
    write_cache: bool,
    write_features: bool,
    cache_dir: Path,
    features_dir: Path,
) -> dict:
    clip_id = clip["clip_id"]
    row: Dict[str, Any] = {
        "clip_id": clip_id,
        "gloss": clip.get("gloss"),
        "split": clip.get("split"),
        "ok": False,
        "source": None,
        "error": None,
        "frames_in": None,
        "feat_shape": None,
    }
    hands = world = pose = None
    fps = 30.0
    try:
        from_video = False
        try:
            hands, world, pose, fps, src = load_cached_landmarks(clip)
            row["source"] = f"clips_cache:{src}"
        except FileNotFoundError:
            if not allow_video:
                row["error"] = "no_motion_cache"
                return row
            video = _resolve(clip.get("video_path"))
            if video is None:
                row["error"] = "no_motion_cache_and_no_video"
                return row
            hands, world, pose, fps = extract_from_video(video)
            row["source"] = f"mediapipe:{video.relative_to(ROOT).as_posix()}"
            from_video = True

        if not hands or all(np.isnan(np.asarray(hands[h])[:, 0, 0]).all() for h in hands):
            row["error"] = "no_hand_detected"
            return row

        row["frames_in"] = int(min(a.shape[0] for a in hands.values()))

        # Only persist a new npz when we ran MediaPipe, or when explicitly forced.
        if write_cache and from_video:
            cache_path = cache_dir / f"{clip_id}.npz"
            save_landmark_cache(cache_path, hands, world, pose, fps, clip_id, row["source"])
            row["cache_path"] = _path_for_report(cache_path)
        elif write_cache and not from_video:
            row["cache_path"] = clip.get("motion_path")

        features = normalize_sequence(hands, pose_wrists=pose)
        row["feat_shape"] = list(features.shape)
        if write_features:
            features_dir.mkdir(parents=True, exist_ok=True)
            feat_path = features_dir / f"{clip_id}.npy"
            np.save(feat_path, features.astype(np.float32))
            row["features_path"] = _path_for_report(feat_path)
        row["ok"] = True
        return row
    except NormalizeError as exc:
        row["error"] = str(exc)
        return row
    except Exception as exc:  # noqa: BLE001 - report any extraction failure
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row


def _process_clip_job(
    clip: dict,
    allow_video: bool,
    write_cache: bool,
    write_features: bool,
    cache_dir: str,
    features_dir: str,
) -> dict:
    """Picklable worker entrypoint."""
    return process_clip(
        clip,
        allow_video=allow_video,
        write_cache=write_cache,
        write_features=write_features,
        cache_dir=Path(cache_dir),
        features_dir=Path(features_dir),
    )


def _format_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _print_progress(done: int, total: int, started: float) -> None:
    elapsed = max(time.time() - started, 1e-6)
    rate = done / elapsed
    remaining = max(total - done, 0)
    eta = remaining / rate if rate > 0 else float("inf")
    print(
        f"progress: {done}/{total} ({rate:.2f} clips/s, ETA {_format_eta(eta)})",
        flush=True,
    )


def run(
    clips: Sequence[dict],
    *,
    allow_video: bool,
    write_cache: bool,
    write_features: bool,
    cache_dir: Path,
    features_dir: Path,
    workers: int = 20,
    force: bool = False,
) -> dict:
    started = time.time()
    total = len(clips)
    rows: List[Optional[dict]] = [None] * total
    todo: List[Tuple[int, dict]] = []

    for i, clip in enumerate(clips):
        feat_path = features_path_for(features_dir, clip["clip_id"])
        if (
            write_features
            and not force
            and features_file_valid(feat_path)
        ):
            rows[i] = skipped_feature_row(clip, features_dir)
        else:
            todo.append((i, clip))

    done = sum(1 for r in rows if r is not None)
    if done and (done % 100 == 0 or not todo):
        _print_progress(done, total, started)

    n_workers = max(int(workers), 1)

    if not todo:
        pass
    elif n_workers == 1:
        _worker_init(allow_video)
        for n, (idx, clip) in enumerate(todo, 1):
            row = process_clip(
                clip,
                allow_video=allow_video,
                write_cache=write_cache,
                write_features=write_features,
                cache_dir=cache_dir,
                features_dir=features_dir,
            )
            rows[idx] = row
            status = "ok" if row["ok"] else f"FAIL:{row['error']}"
            print(f"[{done + n}/{total}] {clip['clip_id']} {status}", flush=True)
            if (done + n) % 100 == 0 or (done + n) == total:
                _print_progress(done + n, total, started)
    else:
        cache_s = str(cache_dir)
        feat_s = str(features_dir)
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(allow_video,),
        ) as pool:
            futures = {
                pool.submit(
                    _process_clip_job,
                    clip,
                    allow_video,
                    write_cache,
                    write_features,
                    cache_s,
                    feat_s,
                ): (idx, clip)
                for idx, clip in todo
            }
            finished_todo = 0
            for fut in as_completed(futures):
                idx, clip = futures[fut]
                row = fut.result()
                rows[idx] = row
                finished_todo += 1
                finished = done + finished_todo
                status = "ok" if row["ok"] else f"FAIL:{row['error']}"
                print(f"[{finished}/{total}] {clip['clip_id']} {status}", flush=True)
                if finished % 100 == 0 or finished == total:
                    _print_progress(finished, total, started)

    out_rows: List[dict] = [r for r in rows if r is not None]
    if len(out_rows) != total:
        missing = total - len(out_rows)
        raise RuntimeError(f"internal error: {missing} clips missing results")

    ok = [r for r in out_rows if r["ok"]]
    failed = [r for r in out_rows if not r["ok"]]
    err_counts: Dict[str, int] = {}
    for r in failed:
        err_counts[r["error"] or "unknown"] = err_counts.get(r["error"] or "unknown", 0) + 1

    report = {
        "norm_version": NORM_VERSION,
        "seq_len": SEQ_LEN,
        "feat_dim": FEAT_DIM,
        "layout": feature_layout(),
        "num_clips": len(out_rows),
        "num_ok": len(ok),
        "num_failed": len(failed),
        "num_skipped": sum(1 for r in out_rows if r.get("skipped")),
        "workers": n_workers,
        "error_counts": err_counts,
        "elapsed_s": round(time.time() - started, 3),
        "clips": out_rows,
    }
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Extract / reuse landmarks and normalize features")
    ap.add_argument("--manifest", type=Path, nargs="*", default=None,
                    help="Manifest JSON(s). Default: lexical + letters manifests")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N clips (for smoke tests)")
    ap.add_argument("--allow-video", action="store_true",
                    help="Run MediaPipe when clips/motion cache is missing")
    ap.add_argument("--write-cache", action="store_true",
                    help="When MediaPipe runs, write npz under landmarks/cache/")
    ap.add_argument("--write-features", action="store_true", default=True,
                    help="Write (SEQ_LEN, FEAT_DIM) npy under landmarks/features (default on)")
    ap.add_argument("--no-write-features", action="store_false", dest="write_features")
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    ap.add_argument("--report", type=Path, default=REPORT_PATH)
    ap.add_argument("--only-missing", action="store_true",
                    help="Only clips whose motion_path is missing")
    ap.add_argument("--seed", type=int, default=0,
                    help="When --limit is set, shuffle with this seed before cutting")
    ap.add_argument("--workers", type=int, default=20,
                    help="Worker processes for clip extraction (default 20)")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even when a valid feature npy already exists")
    args = ap.parse_args(argv)

    paths = list(args.manifest) if args.manifest else [DEFAULT_LEXICAL, DEFAULT_LETTERS]
    clips: List[dict] = []
    for p in paths:
        clips.extend(_load_manifest(p if p.is_absolute() else ROOT / p))

    if args.only_missing:
        clips = [c for c in clips if _resolve(c.get("motion_path")) is None]

    if args.limit is not None:
        rng = np.random.default_rng(args.seed)
        # Prefer cached motion when not extracting from video, so --limit smokes
        # the normalizer instead of a pile of no_motion_cache failures.
        if not args.allow_video:
            cached = [c for c in clips if _resolve(c.get("motion_path")) is not None]
            rest = [c for c in clips if _resolve(c.get("motion_path")) is None]
            order_c = np.arange(len(cached))
            rng.shuffle(order_c)
            order_r = np.arange(len(rest))
            rng.shuffle(order_r)
            clips = [cached[i] for i in order_c] + [rest[i] for i in order_r]
        else:
            order = np.arange(len(clips))
            rng.shuffle(order)
            clips = [clips[i] for i in order]
        clips = clips[: args.limit]

    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else ROOT / args.cache_dir
    features_dir = args.features_dir if args.features_dir.is_absolute() else ROOT / args.features_dir
    report_path = args.report if args.report.is_absolute() else ROOT / args.report

    report = run(
        clips,
        allow_video=args.allow_video,
        write_cache=args.write_cache,
        write_features=args.write_features,
        cache_dir=cache_dir,
        features_dir=features_dir,
        workers=args.workers,
        force=args.force,
    )
    # Parent alone writes the extract report (no worker races).
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"done: ok={report['num_ok']} failed={report['num_failed']} "
        f"skipped={report.get('num_skipped', 0)} workers={report.get('workers')} "
        f"errors={report['error_counts']} report={report_path}",
        flush=True,
    )
    return 0 if report["num_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
