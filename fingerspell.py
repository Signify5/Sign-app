"""ASL manual alphabet as authored hand poses for the 3D rig (canonical right hand).

Each letter is a handshape (per-finger flex and spread, a thumb tip target solved with a small IK)
plus a palm orientation. J and Z also trace a short path. Coordinates for thumb targets are in the
palm frame of the rig, in centimetres: x toward the thumb side, y along the fingers, z out of the palm.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import minimize

from hand3d import Pose, Rig, Tracks, min_jerk, q_axis, q_from_matrix, q_mul, q_rotate
from motion import FPS

FINGERS = ("index", "middle", "ring", "pinky")
SPELL_POS = np.array([-0.12, 0.03, 0.26])  # beside the chin, a little in front of the shoulder
HOLD_S = 0.28  # each letter is held this long
CHANGE_S = 0.12  # handshape change between letters
PATH_S = 0.5  # J and Z take longer to draw
DOUBLE_SHIFT_M = 0.025  # a doubled letter slides toward the signer's right

# (mcp, pip, dip) flex in degrees
STRAIGHT = (0, 0, 0)
FIST = (90, 85, 25)  # bend mostly at the knuckle so the curled fingers keep their length
CURVED = (18, 38, 26)  # C
ROUND = (42, 62, 38)  # O
BENT = (25, 95, 72)  # E: tips folded down onto the thumb
HOOK = (12, 95, 62)  # X
FOLDED = (78, 88, 40)  # M, N: fingers draped over the thumb
TUCK = (60, 80, 45)  # D's companions curl less than a fist

# The bind pose fans the fingers out; these turns (degrees about the palm normal, + toward the pinky)
# bring them parallel. `spread` in a Shape is added on top.
PARALLEL = {"index": 5.0, "middle": -6.0, "ring": -15.0, "pinky": -19.0}
# Fingers curled at the knuckle pack together toward the middle finger (scaled by the knuckle bend).
CURL_GATHER = {"index": 3.0, "middle": 0.0, "ring": -3.0, "pinky": -6.0}

ThumbTarget = Union[Tuple[float, float, float], str]  # palm-frame cm, or "tip:<finger>" / "pip:<finger>"


@dataclass
class Shape:
    fingers: Dict[str, Tuple[float, float, float]]
    thumb: ThumbTarget
    spread: Dict[str, float] = field(default_factory=dict)  # degrees on top of PARALLEL, + toward the pinky
    fingers_dir: Tuple[float, float, float] = (0.08, 1.0, 0.0)  # scene axes: x signer's left, y up, z to viewer
    palm_dir: Tuple[float, float, float] = (0.15, 0.0, 1.0)
    path: Optional[str] = None  # "J" or "Z"


def _f(**kw) -> Dict[str, Tuple[float, float, float]]:
    out = {f: FIST for f in FINGERS}
    out.update(kw)
    return out


SIDE = dict(fingers_dir=(0.1, 1.0, 0.1), palm_dir=(1.0, 0.0, 0.45))  # palm toward the signer's left: C, O
POINT_LEFT = dict(fingers_dir=(1.0, 0.05, 0.1), palm_dir=(0.0, 0.0, -1.0))  # G, H: fingers sideways, palm in
DOWN_FWD = dict(fingers_dir=(0.25, -0.75, 0.6), palm_dir=(0.0, -0.6, -0.8))  # P
DOWN = dict(fingers_dir=(0.15, -1.0, 0.15), palm_dir=(0.0, 0.0, -1.0))  # Q
ON_FIST = (-0.6, 8.9, 5.0)  # thumb across the front of the curled index and middle (S, I, X, Z)
BESIDE = (5.8, 9.0, 1.0)  # thumb up along the side of the index (A)
OUT = (9.5, 5.5, 0.8)  # thumb stuck out sideways (L, Y)

ALPHABET: Dict[str, Shape] = {
    "A": Shape(_f(), BESIDE),
    "B": Shape(_f(index=STRAIGHT, middle=STRAIGHT, ring=STRAIGHT, pinky=STRAIGHT), (-1.2, 5.8, 2.4),
               spread=dict(index=1, middle=1, ring=2, pinky=3)),
    "C": Shape(_f(index=CURVED, middle=CURVED, ring=CURVED, pinky=CURVED), (2.0, 6.5, 6.0), **SIDE),
    "D": Shape(_f(index=STRAIGHT, middle=TUCK, ring=TUCK, pinky=TUCK), "tip:middle"),
    "E": Shape(_f(index=BENT, middle=BENT, ring=BENT, pinky=BENT), (-1.5, 7.3, 2.0)),
    "F": Shape(_f(index=(38, 62, 38), middle=STRAIGHT, ring=STRAIGHT, pinky=STRAIGHT), "tip:index",
               spread=dict(middle=-2, ring=4, pinky=8)),
    "G": Shape(_f(index=STRAIGHT), (3.6, 10.5, 2.4), **POINT_LEFT),
    "H": Shape(_f(index=STRAIGHT, middle=STRAIGHT), "pip:ring", spread=dict(index=1), **POINT_LEFT),
    "I": Shape(_f(pinky=STRAIGHT), ON_FIST),
    "J": Shape(_f(pinky=STRAIGHT), ON_FIST, path="J"),
    "K": Shape(_f(index=STRAIGHT, middle=(40, 0, 0)), "pip:middle", spread=dict(index=-5, middle=3)),
    "L": Shape(_f(index=STRAIGHT), OUT),
    "M": Shape(_f(index=FOLDED, middle=FOLDED, ring=FOLDED), (-4.4, 7.4, 2.0)),
    "N": Shape(_f(index=FOLDED, middle=FOLDED), (-2.0, 7.6, 2.0)),
    "O": Shape(_f(index=ROUND, middle=ROUND, ring=ROUND, pinky=ROUND), "tip:index", **SIDE),
    "P": Shape(_f(index=STRAIGHT, middle=(40, 0, 0)), "pip:middle", spread=dict(index=-5, middle=3), **DOWN_FWD),
    "Q": Shape(_f(index=STRAIGHT), (3.6, 10.5, 2.4), **DOWN),
    "R": Shape(_f(index=STRAIGHT, middle=(10, 0, 0)), "pip:ring", spread=dict(index=9, middle=-10)),
    "S": Shape(_f(), ON_FIST),
    "T": Shape(_f(index=(80, 95, 50)), (1.4, 9.8, 2.8)),
    "U": Shape(_f(index=STRAIGHT, middle=STRAIGHT), "pip:ring", spread=dict(index=1)),
    "V": Shape(_f(index=STRAIGHT, middle=STRAIGHT), "pip:ring", spread=dict(index=-11, middle=11)),
    "W": Shape(_f(index=STRAIGHT, middle=STRAIGHT, ring=STRAIGHT), "tip:pinky", spread=dict(index=-10, ring=10)),
    "X": Shape(_f(index=HOOK), ON_FIST),
    "Y": Shape(_f(pinky=STRAIGHT), OUT, spread=dict(pinky=22)),
    "Z": Shape(_f(index=STRAIGHT), ON_FIST, path="Z"),
}


# ---- forward kinematics on the rig ----

class _Kin:
    def __init__(self, rig: Rig) -> None:
        self.rig = rig
        self.F = rig.rest_frame  # columns: thumb side, fingers, palm normal (rig bind space)
        self.x, self.y, self.z = self.F.T
        ji = {n: i for i, n in enumerate(rig.names)}
        self.ji = ji
        self.wrist = rig.joints[ji["wrist"]]

    def finger_quats(self, finger: str, flex: Tuple[float, float, float], spread: float) -> List[Tuple[int, np.ndarray]]:
        bones = [b for b, _ in self.rig.chains[finger]][1:]  # mcp, pip, dip (the metacarpal stays put)
        out = []
        for k, (bone, deg) in enumerate(zip(bones, flex)):
            q = q_axis(self.x, np.radians(deg))
            if k == 0:
                q = q_mul(q_axis(self.z, np.radians(spread)), q)
            out.append((bone, q))
        return out

    def chain_points(self, finger: str, q: np.ndarray) -> List[np.ndarray]:
        """Joint positions (bind space, metres) down a chain for local rotations q, ending at the tip."""
        chain = self.rig.chains[finger]
        g = np.array([0.0, 0, 0, 1])
        p = self.rig.joints[chain[0][0]].copy()
        pts = [p.copy()]
        for bone, d in chain:
            g = q_mul(g, q[bone])
            p = p + q_rotate(g, d)
            pts.append(p.copy())
        return pts

    def to_palm(self, p: np.ndarray) -> np.ndarray:
        return self.F.T @ (p - self.wrist) * 100

    def from_palm(self, c: np.ndarray) -> np.ndarray:
        return self.wrist + self.F @ (np.asarray(c, float) / 100)

    def thumb_quats(self, params: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        a, b, c, e = np.radians(params)
        (cmc, _), (mcp, d1), (ip, d2) = self.rig.chains["thumb"]
        pad = -0.75 * self.x + 0.66 * self.z  # the thumb pad faces across the palm
        return [(cmc, q_mul(q_axis(self.z, a), q_axis(self.y, b))),
                (mcp, q_axis(np.cross(d1, pad), c)),
                (ip, q_axis(np.cross(d2, pad), e))]


def _solve_thumb(kin: _Kin, q: np.ndarray, target: np.ndarray) -> None:
    """Fit the thumb's four angles so its tip lands on `target` (bind space), writing them into q."""
    def cost(params: np.ndarray) -> float:
        qq = q.copy()
        for b, r in kin.thumb_quats(params):
            qq[b] = r
        pts = kin.chain_points("thumb", qq)
        err = np.sum((pts[-1] - target) ** 2) * 1e4  # cm^2
        # keep the thumb out of the palm and in a natural range
        z = min(kin.to_palm(p)[2] for p in pts[1:])
        return err + 0.0002 * np.sum(params ** 2) + 50 * max(0.0, 0.3 - z) ** 2
    best = None
    for start in ([0, 0, 10, 10], [-30, -30, 30, 30], [30, -40, 20, 20], [-50, -10, 40, 40]):
        r = minimize(cost, np.array(start, float), method="L-BFGS-B",
                     bounds=[(-75, 60), (-80, 40), (-15, 75), (-15, 85)])
        if best is None or r.fun < best.fun:
            best = r
    for b, r in kin.thumb_quats(best.x):
        q[b] = r


def _orient(fingers: Tuple[float, float, float], palm: Tuple[float, float, float]) -> np.ndarray:
    f = np.asarray(fingers, float)
    f /= np.linalg.norm(f)
    p = np.asarray(palm, float)
    p -= f * (p @ f)
    p /= np.linalg.norm(p)
    return np.stack([np.cross(f, p), f, p], axis=1)


_CACHE: Dict[Tuple[int, str], Pose] = {}


def letter_pose(rig: Rig, letter: str) -> Pose:
    """Static pose of one letter, held beside the chin."""
    key = (id(rig), letter.upper())
    if key not in _CACHE:
        _CACHE[key] = _letter_pose(rig, letter)
    return _CACHE[key]


def _letter_pose(rig: Rig, letter: str) -> Pose:
    shape = ALPHABET[letter.upper()]
    kin = _Kin(rig)
    q = np.tile([0.0, 0.0, 0.0, 1.0], (len(rig.names), 1))
    for f in FINGERS:
        gather = CURL_GATHER[f] * min(shape.fingers[f][0] / 90.0, 1.0)
        for b, r in kin.finger_quats(f, shape.fingers[f], PARALLEL[f] + gather + shape.spread.get(f, 0.0)):
            q[b] = r
    t = shape.thumb
    if isinstance(t, str):
        kind, finger = t.split(":")
        pts = kin.chain_points(finger, q)
        target = pts[-1] if kind == "tip" else (pts[2] + pts[3]) / 2  # pip: middle of the proximal phalanx
        # touch the pad, not the centre of the bone
        target = target + kin.F @ np.array([0.0, 0.0, 0.006])
    else:
        target = kin.from_palm(t)
    _solve_thumb(kin, q, target)
    q[0] = q_from_matrix(_orient(shape.fingers_dir, shape.palm_dir) @ rig.rest_frame.T)
    return Pose(SPELL_POS.copy(), q)


def letter_frames(rig: Rig, letter: str, n: int) -> List[Pose]:
    """`n` frames of a letter: held still, or tracing J / Z."""
    base = letter_pose(rig, letter)
    path = ALPHABET[letter.upper()].path
    if path is None:
        return [base] * n
    out = []
    for k in range(n):
        t = k / max(n - 1, 1)
        pos = base.pos.copy()
        quats = base.quats.copy()
        if path == "Z":  # index draws Z as the signer would write it (the signer's right is -x)
            pts = np.array([[0, 0], [-0.06, 0], [0.0, -0.06], [-0.06, -0.06]])
            s = min(t * 3, 2.999)
            i = int(s)
            pos[:2] += pts[i] + (pts[i + 1] - pts[i]) * (s - i)
        else:  # J: the pinky draws down and hooks toward the signer's left as the palm turns in
            pos[0] += 0.035 * t * t
            pos[1] -= 0.06 * np.sin(t * np.pi / 2)
            turn = q_axis(np.array([0.0, 1.0, 0.0]), np.radians(100) * t * t)
            quats[0] = q_mul(turn, quats[0])
        out.append(Pose(pos, quats))
    return out


def spell_track(rig: Rig, word: str, speed: float = 1.0, fps: float = FPS) -> Optional[Tracks]:
    """The dominant (right) hand spells `word` letter by letter; the other hand stays at rest.

    Characters outside A-Z are skipped. Returns None if nothing is left to spell.
    """
    letters = [c for c in word.upper() if "A" <= c <= "Z"]
    if not letters:
        return None
    hold = max(int(round(HOLD_S * fps / speed)), 1)
    change = max(int(round(CHANGE_S * fps / speed)), 2)
    draw = max(int(round(PATH_S * fps / speed)), 3)
    out: List[Pose] = []
    prev = None
    for i, c in enumerate(letters):
        frames = letter_frames(rig, c, draw if ALPHABET[c].path else hold)
        if i and c == letters[i - 1] and ALPHABET[c].path is None and not (i > 1 and c == letters[i - 2]):
            shift = np.array([-DOUBLE_SHIFT_M, 0.0, 0.0])
            frames = [Pose(p.pos + shift, p.quats) for p in frames]
        if prev is not None:
            out.extend(prev.lerp(frames[0], min_jerk(k / change)) for k in range(1, change))
        out.extend(frames)
        prev = frames[-1]
    return {"Right": out, "Left": None}
