from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from motion import FPS, HANDS, Motion

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
HAND_MODEL_PATH = Path("models") / "hand_landmarker.task"
POSE_MODEL_PATH = Path("models") / "pose_landmarker_lite.task"

MAX_GAP_FRAMES = 8
SMOOTHING = 0.55  # weight on the new landmark; lower = smoother
MIN_SHOULDER_PX = 20.0
POSE_SAMPLES = 8
# Pose landmark of each arm's wrist, by the hand label used in motion files ("Right" is the signer's right).
POSE_WRIST = {"Left": 15, "Right": 16}
REST_Y = 1.15     # body units: a wrist below this is resting
ARM_RAISE = 0.3   # wrist rise above where the arm rests in the clip that counts as raised


class AnimateError(RuntimeError):
    pass


def _ensure(path: Path, url: str) -> Path:
    if path.is_file() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path


def ensure_models() -> None:
    _ensure(HAND_MODEL_PATH, HAND_MODEL_URL)
    _ensure(POSE_MODEL_PATH, POSE_MODEL_URL)


def _read_frames(path: Path, start_ms: int, end_ms: Optional[int]) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise AnimateError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    first = int(round(start_ms / 1000 * fps))
    last = None if end_ms is None else int(round(end_ms / 1000 * fps))
    frames, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok or (last is not None and i > last):
            break
        if i >= first:
            frames.append(frame)
        i += 1
    cap.release()
    if not frames:
        raise AnimateError(f"no frames in {path} for {start_ms}-{end_ms} ms")
    return frames, fps


def _mp_image(frame_bgr: np.ndarray):
    import mediapipe as mp

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))


def body_frame(frames: List[np.ndarray]) -> Tuple[np.ndarray, float]:
    """(mid-shoulder pixel position, shoulder width in px), the median over sampled frames."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_ensure(POSE_MODEL_PATH, POSE_MODEL_URL))),
        running_mode=vision.RunningMode.IMAGE,
    ))
    h, w = frames[0].shape[:2]
    mids, widths = [], []
    try:
        for i in np.linspace(0, len(frames) - 1, min(POSE_SAMPLES, len(frames))).astype(int):
            result = landmarker.detect(_mp_image(frames[i]))
            if not result.pose_landmarks:
                continue
            lm = result.pose_landmarks[0]
            left, right = np.array([lm[11].x * w, lm[11].y * h]), np.array([lm[12].x * w, lm[12].y * h])
            width = float(np.linalg.norm(left - right))
            if width >= MIN_SHOULDER_PX:
                mids.append((left + right) / 2)
                widths.append(width)
    finally:
        landmarker.close()
    if not widths:
        raise AnimateError("no body/shoulders found to normalize against")
    return np.median(mids, axis=0).astype(np.float32), float(np.median(widths))


def track_hands(frames: List[np.ndarray], fps: float) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
    """Per frame, {handedness: 21x5 landmarks}: pixel x, y, then MediaPipe 3D world x, y, z (metres); and the
    pose tracker's arm wrists, (T, 2, 2) pixels in HANDS order, NaN where no body is found."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    h, w = frames[0].shape[:2]
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_ensure(HAND_MODEL_PATH, HAND_MODEL_URL))),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    ))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_ensure(POSE_MODEL_PATH, POSE_MODEL_URL))),
        running_mode=vision.RunningMode.VIDEO,
    ))
    tracks: List[Dict[str, np.ndarray]] = []
    last: Dict[str, Tuple[int, np.ndarray]] = {}  # label -> (frame, wrist px) of its latest detection
    arms = np.full((len(frames), len(HANDS), 2), np.nan, np.float32)
    try:
        for n, frame in enumerate(frames):
            image, ms = _mp_image(frame), int(n / fps * 1000)
            result = landmarker.detect_for_video(image, ms)
            body = pose.detect_for_video(image, ms)
            arm = {}
            if body.pose_landmarks:
                lm = body.pose_landmarks[0]
                arm = {label: np.array([lm[i].x * w, lm[i].y * h]) for label, i in POSE_WRIST.items()}
                arms[n] = [arm[label] for label in HANDS]
            dets = [(np.array([[p.x * w, p.y * h, q.x, q.y, q.z] for p, q in zip(hand, world)], np.float32),
                     label[0].category_name)
                    for hand, world, label in zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness)]
            found = assign_hands(dets, arm, {k: v for k, (i, v) in last.items() if n - i <= MAX_GAP_FRAMES}, w)
            for label, pts in found.items():
                last[label] = (n, pts[0, :2])
            tracks.append(found)
    finally:
        landmarker.close()
        pose.close()
    return tracks, arms


def assign_hands(dets: List[Tuple[np.ndarray, str]], arm: Dict[str, np.ndarray],
                 last: Dict[str, np.ndarray], width: float) -> Dict[str, np.ndarray]:
    """Which detected hand is which: the one nearest each arm's wrist (from the pose), else nearest where
    that hand just was, else the hand tracker's own guess. Two detections never share a label.

    The tracker's handedness label alone flips between frames and can give both hands one label, which
    dropped a hand or swapped them mid-sign.
    """
    def cost(pts: np.ndarray, label: str) -> float:
        ref = arm.get(label, last.get(label))
        if ref is None:  # nothing to go on but the tracker's guess, which loses to any nearby evidence
            return width / 2 if label == dets_label[id(pts)] else width
        return float(np.linalg.norm(pts[0, :2] - ref))

    dets_label = {id(p): lab for p, lab in dets}
    if not dets:
        return {}
    if len(dets) == 1:
        pts = dets[0][0]
        return {min(HANDS, key=lambda lab: cost(pts, lab)): pts}
    (a, _), (b, _) = dets[:2]
    straight = cost(a, HANDS[0]) + cost(b, HANDS[1])
    crossed = cost(a, HANDS[1]) + cost(b, HANDS[0])
    return {HANDS[0]: a, HANDS[1]: b} if straight <= crossed else {HANDS[0]: b, HANDS[1]: a}


def clean_tracks(tracks: List[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
    """Interpolate short detection gaps per hand, then low-pass the motion."""
    n = len(tracks)
    out: List[Dict[str, np.ndarray]] = [dict() for _ in range(n)]
    for label in {k for t in tracks for k in t}:
        seen = [i for i in range(n) if label in tracks[i]]
        series: Dict[int, np.ndarray] = {i: tracks[i][label] for i in seen}
        for a, b in zip(seen, seen[1:]):
            if 1 < b - a <= MAX_GAP_FRAMES + 1:
                for i in range(a + 1, b):
                    t = (i - a) / (b - a)
                    series[i] = (1 - t) * tracks[a][label] + t * tracks[b][label]
        smoothed: Optional[np.ndarray] = None
        for i in range(n):
            if i not in series:
                smoothed = None
                continue
            smoothed = series[i] if smoothed is None else SMOOTHING * series[i] + (1 - SMOOTHING) * smoothed
            out[i][label] = smoothed
    return out


def _resample(arr: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    """Resample (T, 21, k) to dst_fps; a frame is NaN unless both neighbours are tracked."""
    n = len(arr)
    new_n = max(int(round(n * dst_fps / src_fps)), 1)
    pos = np.linspace(0, n - 1, new_n)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = (pos - lo)[:, None, None].astype(np.float32)
    return arr[lo] * (1 - frac) + arr[hi] * frac


def arm_track(in_path: Union[Path, str], start_ms: int = 0, end_ms: Optional[int] = None) -> np.ndarray:
    """(T, 2, 2) wrist positions of the two arms from the pose tracker (in HANDS order), in body units at FPS,
    NaN where no body is found. Frame for frame with extract_motion's hands, so a hand the hand tracker lost
    while its arm is up can be told from a hand resting out of view."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    frames, fps = _read_frames(Path(in_path), start_ms, end_ms)
    origin, unit = body_frame(frames)
    h, w = frames[0].shape[:2]
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(_ensure(POSE_MODEL_PATH, POSE_MODEL_URL))),
        running_mode=vision.RunningMode.VIDEO,
    ))
    out = np.full((len(frames), len(HANDS), 2), np.nan, np.float32)
    try:
        for n, frame in enumerate(frames):
            body = pose.detect_for_video(_mp_image(frame), int(n / fps * 1000))
            if body.pose_landmarks:
                lm = body.pose_landmarks[0]
                for k, label in enumerate(HANDS):
                    p = lm[POSE_WRIST[label]]
                    out[n, k] = (np.array([p.x * w, p.y * h]) - origin) / unit
    finally:
        pose.close()
    return _resample(out, fps, FPS)


def arm_raised(arms: np.ndarray) -> np.ndarray:
    """(T, 2) bool: each arm's wrist well above where that arm rests in the clip (some signers rest their hands
    at the waist, so a fixed height is not enough). `arms` is (T, 2, 2) in body units; no body is not raised."""
    out = np.zeros(arms.shape[:2], bool)
    for k in range(arms.shape[1]):
        y = arms[:, k, 1]
        if not np.isnan(y).all():
            with np.errstate(invalid="ignore"):
                out[:, k] = y < min(REST_Y, float(np.nanpercentile(y, 95)) - ARM_RAISE)
    return out


def fill_lost_hands(hands: Dict[str, np.ndarray], world: Dict[str, np.ndarray], arms: np.ndarray) -> int:
    """Fill frames where a hand's arm is raised but the hand tracker lost the hand (usually while the hands
    touch or cross, as in SCHOOL or STOP), in place. The hand keeps the shape it had either side of the gap
    (blended across it) and its wrist follows the arm's wrist from the pose tracker. Returns the frames filled.
    A hand never seen in the clip is left out."""
    filled = 0
    raised = arm_raised(arms)
    for k, label in enumerate(HANDS):
        xy, w3 = hands[label], world[label]
        seen = ~np.isnan(xy[:, 0, 0])
        if not seen.any():
            continue
        lost = raised[:, k] & ~seen
        t = 0
        while t < len(lost):
            if not lost[t]:
                t += 1
                continue
            a = t
            while t < len(lost) and lost[t]:
                t += 1
            anchors = [i for i in (a - 1, t) if 0 <= i < len(seen) and seen[i]]
            if not anchors:
                continue
            for f in range(a, t):
                wgt = 0.0 if len(anchors) == 1 else (f - anchors[0]) / (anchors[1] - anchors[0])
                i, j = anchors[0], anchors[-1]
                shape = (1 - wgt) * (xy[i] - xy[i, 0]) + wgt * (xy[j] - xy[j, 0])
                wrist = (1 - wgt) * xy[i, 0] + wgt * xy[j, 0]
                offset = (1 - wgt) * (xy[i, 0] - arms[i, k]) + wgt * (xy[j, 0] - arms[j, k])
                if not np.isnan(arms[f, k]).any() and not np.isnan(offset).any():
                    wrist = arms[f, k] + offset
                xy[f] = shape + wrist
                w3[f] = (1 - wgt) * w3[i] + wgt * w3[j]
                filled += 1
    return filled


def extract_motion(
    in_path: Union[Path, str],
    start_ms: int = 0,
    end_ms: Optional[int] = None,
) -> Motion:
    """Track both hands in a sign video as body-normalized motion data (no video is kept).

    Raises AnimateError if there is no body to normalize against or no hand is ever found.
    """
    frames, fps = _read_frames(Path(in_path), start_ms, end_ms)
    origin, unit = body_frame(frames)
    raw, arms_px = track_hands(frames, fps)
    tracks = clean_tracks(raw)
    if not any(tracks):
        raise AnimateError("no hands detected")

    n = len(frames)
    hands, world = {}, {}
    for label in HANDS:
        arr = np.full((n, 21, 2), np.nan, np.float32)
        arr3 = np.full((n, 21, 3), np.nan, np.float32)
        for i, t in enumerate(tracks):
            if label in t:
                arr[i] = (t[label][:, :2] - origin) / unit
                arr3[i] = t[label][:, 2:]
        hands[label] = _resample(arr, fps, FPS)
        world[label] = _resample(arr3, fps, FPS)
    fill_lost_hands(hands, world, _resample((arms_px - origin) / unit, fps, FPS))
    return Motion(hands, FPS, world)
