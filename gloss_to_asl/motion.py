from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

FPS = 30.0
HANDS = ("Left", "Right")  # MediaPipe labels; "Right" appears on the viewer's left
TRANSITION_S = 0.30

# Coordinates are body units: origin at the mid-shoulder point, 1.0 = shoulder width, +y is down.


class MotionError(RuntimeError):
    pass


@dataclass
class Motion:
    """Hand-joint motion of one sign: per hand a (T, 21, 2) array, NaN where the hand is absent.

    `world` optionally holds the same frames as (T, 21, 3) MediaPipe world landmarks (metres, hand-centred,
    camera axes: x right, y down, z away), which drive the 3D hand.
    """

    hands: Dict[str, np.ndarray]
    fps: float = FPS
    world: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def frames(self) -> int:
        return len(next(iter(self.hands.values()))) if self.hands else 0

    def present(self, hand: str) -> np.ndarray:
        arr = self.hands.get(hand)
        return np.zeros(self.frames, bool) if arr is None else ~np.isnan(arr[:, 0, 0])

    @property
    def quality(self) -> float:
        """Share of frames with at least one hand tracked."""
        if not self.frames:
            return 0.0
        any_hand = np.zeros(self.frames, bool)
        for hand in HANDS:
            any_hand |= self.present(hand)
        return float(any_hand.mean())

    def trimmed(self) -> "Motion":
        """Drop leading/trailing frames with no hand at all."""
        any_hand = np.zeros(self.frames, bool)
        for hand in HANDS:
            any_hand |= self.present(hand)
        idx = np.flatnonzero(any_hand)
        if not len(idx):
            raise MotionError("motion has no tracked hands")
        lo, hi = idx[0], idx[-1] + 1
        return Motion({h: a[lo:hi] for h, a in self.hands.items()}, self.fps,
                      {h: a[lo:hi] for h, a in self.world.items()})

    def save(self, path: Union[Path, str]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, fps=np.float32(self.fps), **self.hands,
                            **{f"{h}_world": a for h, a in self.world.items()})

    @classmethod
    def load(cls, path: Union[Path, str]) -> "Motion":
        try:
            with np.load(path) as data:
                hands = {h: data[h].astype(np.float32) for h in HANDS if h in data.files}
                world = {h: data[f"{h}_world"].astype(np.float32) for h in HANDS if f"{h}_world" in data.files}
                return cls(hands, float(data["fps"]), world)
        except (OSError, KeyError, ValueError) as exc:
            raise MotionError(f"could not read motion file {path}: {exc}") from exc


def _template(mirror: bool) -> np.ndarray:
    """A relaxed, open hand: wrist at the origin, fingers pointing up (-y)."""
    pts = np.array([
        (0, 0),
        (-.07, -.05), (-.12, -.10), (-.15, -.15), (-.17, -.20),
        (-.06, -.20), (-.07, -.27), (-.075, -.32), (-.078, -.36),
        (-.02, -.21), (-.02, -.29), (-.02, -.35), (-.02, -.39),
        (.02, -.20), (.03, -.27), (.035, -.32), (.037, -.36),
        (.06, -.18), (.075, -.23), (.083, -.27), (.088, -.30),
    ], np.float32)
    if mirror:
        pts[:, 0] *= -1
    return pts


REST: Dict[str, np.ndarray] = {
    "Left": _template(mirror=True) + np.array([0.62, 2.6], np.float32),
    "Right": _template(mirror=False) + np.array([-0.62, 2.6], np.float32),
}


def _smoothstep(t: float) -> float:
    return t * t * (3 - 2 * t)


def _retime(arr: np.ndarray, factor: float) -> np.ndarray:
    """Resample to 1/factor as many frames (factor > 1 = faster). NaN-aware: needs both neighbours."""
    n = len(arr)
    new_n = max(int(round(n / factor)), 1)
    if new_n == n:
        return arr
    pos = np.linspace(0, n - 1, new_n)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    frac = (pos - lo)[:, None, None].astype(np.float32)
    return arr[lo] * (1 - frac) + arr[hi] * frac


def _fill_gaps(arr: np.ndarray) -> Optional[np.ndarray]:
    """Hold the nearest tracked pose across NaN frames; None if the hand never appears."""
    ok = ~np.isnan(arr[:, 0, 0])
    if not ok.any():
        return None
    idx = np.where(ok, np.arange(len(arr)), -1)
    np.maximum.accumulate(idx, out=idx)
    first = np.flatnonzero(ok)[0]
    idx[idx < 0] = first
    return arr[idx]


@dataclass
class Timeline:
    hands: Dict[str, np.ndarray]  # (N, 21, 2), fully valid
    fps: float
    segments: List[Tuple[int, int, str]] = field(default_factory=list)  # (first, last, label)

    @property
    def frames(self) -> int:
        return len(self.hands["Left"])

    def pose(self, i: int) -> Dict[str, np.ndarray]:
        return {h: self.hands[h][i] for h in HANDS}

    def label_at(self, i: int) -> str:
        for first, last, label in self.segments:
            if first <= i <= last:
                return label
        return ""


def compose(
    items: Sequence[Tuple[Motion, str]],
    gap_s: float = 0.0,
    speed: float = 1.0,
    transition_s: float = TRANSITION_S,
    fps: float = FPS,
) -> Timeline:
    """Chain signs into one continuous animation.

    Hands glide from wherever the previous sign ended (or from rest) into the next sign's first pose,
    and return to rest at the end. A hand a sign does not use rests below the frame.
    """
    if not items:
        raise MotionError("nothing to compose")
    if speed <= 0:
        raise MotionError("speed must be positive")

    n_trans = max(int(round(transition_s * fps / speed)), 2)
    n_gap = int(round(gap_s * fps))
    out: Dict[str, List[np.ndarray]] = {h: [] for h in HANDS}
    cur = {h: REST[h].copy() for h in HANDS}
    segments: List[Tuple[int, int, str]] = []

    def emit(pose: Dict[str, np.ndarray]) -> None:
        for h in HANDS:
            out[h].append(pose[h])

    for motion, label in items:
        m = motion.trimmed()
        m = Motion({h: _retime(a, m.fps / fps * speed) for h, a in m.hands.items()}, fps)
        filled = {h: _fill_gaps(m.hands[h]) if h in m.hands else None for h in HANDS}

        seg_start = len(out["Left"])
        first = {h: filled[h][0] if filled[h] is not None else REST[h] for h in HANDS}
        for k in range(1, n_trans + 1):
            t = _smoothstep(k / n_trans)
            emit({h: (1 - t) * cur[h] + t * first[h] for h in HANDS})
        for i in range(m.frames):
            emit({h: filled[h][i] if filled[h] is not None else REST[h] for h in HANDS})
        cur = {h: out[h][-1] for h in HANDS}
        segments.append((seg_start, len(out["Left"]) - 1, label))
        for _ in range(n_gap):
            emit(cur)

    for k in range(1, n_trans + 1):
        t = _smoothstep(k / n_trans)
        emit({h: (1 - t) * cur[h] + t * REST[h] for h in HANDS})

    return Timeline({h: np.stack(v).astype(np.float32) for h, v in out.items()}, fps, segments)
