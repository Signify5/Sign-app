"""Feature normalization for Sign -> Speech landmark sequences.

Contract (must match the browser reimplementation; see NORMALIZATION.md):
  Input landmarks are already in the shoulder frame used by gloss_to_asl/animate.py:
    (xy_px - shoulder_midpoint) / shoulder_width
  Missing landmarks are NaN. Hands keys are anatomical "Left" / "Right".

  Output is a fixed-length float32 array of shape (SEQ_LEN, FEAT_DIM).
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np

# Bump when the layout or math changes. Browser and golden tests pin this.
NORM_VERSION = "sts_landmarks_v1"

HANDS = ("Left", "Right")
SEQ_LEN = 64
# Per hand: presence(1) + wrist_xy(2) + rel_xy for landmarks 1..20 (40) = 43
# Two hands = 86; pose wrists: presence+xy per side = 3*2 = 6; total 92
FEAT_PER_HAND = 43
FEAT_POSE = 6
FEAT_DIM = FEAT_PER_HAND * 2 + FEAT_POSE  # 92

# After trim, need at least this many frames with any hand present.
MIN_HAND_FRAMES = 3
# Presence channel after temporal resample: treat as present if >= this.
PRESENCE_THRESH = 0.5


class NormalizeError(ValueError):
    """Raised when a clip cannot be turned into a feature window."""


def shoulder_normalize(xy_px: np.ndarray, origin: np.ndarray, unit: float) -> np.ndarray:
    """Map image-pixel xy into the animate.py shoulder frame. unit is shoulder width px."""
    if unit <= 0:
        raise NormalizeError("shoulder width must be positive")
    return (xy_px.astype(np.float32) - np.asarray(origin, np.float32)) / float(unit)


def _hand_present(xy: np.ndarray) -> bool:
    return xy.ndim == 2 and xy.shape[0] >= 1 and not np.isnan(xy[0, 0])


def _pack_hand(xy: np.ndarray) -> np.ndarray:
    """Pack one hand into FEAT_PER_HAND floats. xy is (21, 2), NaN if missing."""
    out = np.zeros(FEAT_PER_HAND, np.float32)
    if not _hand_present(xy):
        return out
    wrist = xy[0].astype(np.float32)
    rel = (xy[1:21].astype(np.float32) - wrist)
    if np.isnan(rel).any() or np.isnan(wrist).any():
        return np.zeros(FEAT_PER_HAND, np.float32)
    out[0] = 1.0
    out[1:3] = wrist
    out[3:] = rel.reshape(-1)
    return out


def _pack_pose_wrist(xy: np.ndarray) -> np.ndarray:
    """Pack one pose wrist into 3 floats: presence, x, y."""
    out = np.zeros(3, np.float32)
    if xy is None or np.isnan(xy).any():
        return out
    out[0] = 1.0
    out[1:] = xy.astype(np.float32)
    return out


def frame_features(
    hands: Mapping[str, np.ndarray],
    pose_wrists: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a (FEAT_DIM,) vector for a single frame.

    hands[label] is (21, 2) shoulder-normalized; pose_wrists is (2, 2) Left/Right or None.
    """
    parts = [_pack_hand(np.asarray(hands.get(h, np.full((21, 2), np.nan, np.float32)), np.float32))
             for h in HANDS]
    if pose_wrists is None:
        pose = np.full((2, 2), np.nan, np.float32)
    else:
        pose = np.asarray(pose_wrists, np.float32)
        if pose.shape != (2, 2):
            raise NormalizeError(f"pose_wrists must be (2, 2), got {pose.shape}")
    parts.append(_pack_pose_wrist(pose[0]))
    parts.append(_pack_pose_wrist(pose[1]))
    return np.concatenate(parts).astype(np.float32)


def _sequence_length(hands: Mapping[str, np.ndarray]) -> int:
    lengths = [int(np.asarray(hands[h]).shape[0]) for h in HANDS if h in hands]
    if not lengths:
        raise NormalizeError("no hands in landmark sequence")
    return int(min(lengths))


def _any_hand_mask(hands: Mapping[str, np.ndarray], n: int) -> np.ndarray:
    mask = np.zeros(n, bool)
    for h in HANDS:
        if h not in hands:
            continue
        arr = np.asarray(hands[h], np.float32)
        if arr.shape[0] < n:
            raise NormalizeError(f"hand {h} shorter than sequence length")
        mask |= ~np.isnan(arr[:n, 0, 0])
    return mask


def trim_hand_span(hands: Mapping[str, np.ndarray], n: int) -> Tuple[int, int]:
    """Inclusive-exclusive [lo, hi) covering first..last frame with any hand."""
    present = _any_hand_mask(hands, n)
    idx = np.flatnonzero(present)
    if len(idx) == 0:
        raise NormalizeError("no_hand_detected")
    if len(idx) < MIN_HAND_FRAMES:
        raise NormalizeError("too_short")
    return int(idx[0]), int(idx[-1]) + 1


def _resample_linear(arr: np.ndarray, new_n: int) -> np.ndarray:
    """Linear resample along axis 0. arr is (T, D)."""
    n = arr.shape[0]
    if new_n < 1:
        raise NormalizeError("SEQ_LEN must be >= 1")
    if n == new_n:
        return arr.astype(np.float32, copy=True)
    if n == 1:
        return np.repeat(arr.astype(np.float32), new_n, axis=0)
    pos = np.linspace(0.0, n - 1.0, new_n, dtype=np.float64)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = (pos - lo).astype(np.float32)[:, None]
    a = arr[lo].astype(np.float32)
    b = arr[hi].astype(np.float32)
    return a * (1.0 - frac) + b * frac


def _binarize_presence(features: np.ndarray) -> np.ndarray:
    """Threshold presence channels in-place style; return new array."""
    out = features.astype(np.float32, copy=True)
    # Hand presence at offsets 0 and FEAT_PER_HAND
    for off in (0, FEAT_PER_HAND):
        present = out[:, off] >= PRESENCE_THRESH
        out[:, off] = present.astype(np.float32)
        # Zero the hand block when absent so interpolated ghosts do not leak.
        dead = ~present
        if dead.any():
            out[dead, off:off + FEAT_PER_HAND] = 0.0
    # Pose presence at 86 and 89
    pose_base = FEAT_PER_HAND * 2
    for off in (pose_base, pose_base + 3):
        present = out[:, off] >= PRESENCE_THRESH
        out[:, off] = present.astype(np.float32)
        dead = ~present
        if dead.any():
            out[dead, off:off + 3] = 0.0
    return out


def normalize_sequence(
    hands: Mapping[str, np.ndarray],
    pose_wrists: Optional[np.ndarray] = None,
    *,
    seq_len: int = SEQ_LEN,
) -> np.ndarray:
    """Convert a variable-length landmark clip into (seq_len, FEAT_DIM) features.

    hands: map "Left"/"Right" -> (T, 21, 2) float32, shoulder-normalized, NaN if missing.
    pose_wrists: optional (T, 2, 2) Left/Right pose wrists in the same frame, NaN if missing.
    """
    if seq_len != SEQ_LEN:
        # Allow override only for experiments; golden / shipped path pins SEQ_LEN.
        pass
    n = _sequence_length(hands)
    lo, hi = trim_hand_span(hands, n)
    if pose_wrists is not None:
        pw = np.asarray(pose_wrists, np.float32)
        if pw.shape[0] < hi:
            raise NormalizeError("pose_wrists shorter than hand sequence")
    else:
        pw = None

    rows = []
    for t in range(lo, hi):
        frame_hands = {h: np.asarray(hands[h][t], np.float32) for h in HANDS if h in hands}
        pose_t = None if pw is None else pw[t]
        rows.append(frame_features(frame_hands, pose_t))
    seq = np.stack(rows, axis=0)
    if seq.shape[0] < MIN_HAND_FRAMES:
        raise NormalizeError("too_short")
    resampled = _resample_linear(seq, seq_len)
    return _binarize_presence(resampled)


def normalize_motion_arrays(
    left: np.ndarray,
    right: np.ndarray,
    pose_wrists: Optional[np.ndarray] = None,
    *,
    seq_len: int = SEQ_LEN,
) -> np.ndarray:
    """Convenience wrapper taking explicit Left/Right arrays."""
    return normalize_sequence(
        {"Left": left, "Right": right},
        pose_wrists=pose_wrists,
        seq_len=seq_len,
    )


def feature_layout() -> Dict[str, object]:
    """Machine-readable layout description for exporters and the browser."""
    return {
        "norm_version": NORM_VERSION,
        "seq_len": SEQ_LEN,
        "feat_dim": FEAT_DIM,
        "hands_order": list(HANDS),
        "min_hand_frames": MIN_HAND_FRAMES,
        "presence_thresh": PRESENCE_THRESH,
        "per_frame": {
            "Left.presence": 0,
            "Left.wrist_xy": [1, 2],
            "Left.rel_xy_landmarks_1_to_20": [3, 42],
            "Right.presence": 43,
            "Right.wrist_xy": [44, 45],
            "Right.rel_xy_landmarks_1_to_20": [46, 85],
            "pose_Left.presence_xy": [86, 88],
            "pose_Right.presence_xy": [89, 91],
        },
        "input_space": "shoulder_v1",
        "input_space_note": (
            "(xy_px - median_shoulder_midpoint) / median_shoulder_width; "
            "shoulders = MediaPipe pose landmarks 11 (Left) and 12 (Right); "
            "same as gloss_to_asl.animate.extract_motion"
        ),
    }
