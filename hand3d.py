"""3D hand playback: retarget tracked MediaPipe hand landmarks onto the rigged MakeHuman hand.

Each frame becomes, per hand, a wrist position plus one rotation per rig bone; `write_player` bakes a
playlist into a self-contained three.js page (`python hand3d.py HELLO WORLD -o hello.html`).

Scene frame: metres, origin at mid-shoulder, x toward the viewer's right, y up, z toward the viewer.
The signer's left hand is solved and drawn as a mirrored right hand ("canonical" space, x flipped).
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from motion import FPS, HANDS, TRANSITION_S, Motion, MotionError

ASSETS = Path(__file__).parent / "assets" / "hand"
RIG_PATH = ASSETS / "makehuman_hand.json"
TEMPLATE_PATH = ASSETS / "player_tpl.html"

BODY_UNIT_M = 0.36  # shoulder width in metres (motion files are in shoulder widths)
LIFT_Y_M = -0.26  # below this the signer is still raising the hands into the video (or lowering them after)
HAND_DEPTH_M = 0.22  # hands sit this far in front of the shoulder plane (depth is not tracked)
MIRROR = {"Left": True, "Right": False}  # MediaPipe "Right" tracks the signer's right hand (right-handed 3D landmarks)

# MediaPipe landmark ids: 0 wrist; thumb 1-4; index 5-8; middle 9-12; ring 13-16; pinky 17-20.
FINGER_LM = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
CHAIN_LM: Dict[str, List[int]] = {f: [0, b, b + 1, b + 2, b + 3] for f, b in FINGER_LM.items()}
CHAIN_LM["thumb"] = [1, 2, 3, 4]


# ---- quaternions as (x, y, z, w), matching three.js ----

def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def q_conj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def q_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return q_mul(q_mul(q, np.array([*v, 0.0])), q_conj(q))[:3]


def q_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation taking direction a onto direction b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = float(a @ b)
    if d < -0.999999:
        axis = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0, 1.0, 0])
        return np.array([*(axis / np.linalg.norm(axis)), 0.0])
    q = np.array([*np.cross(a, b), 1.0 + d])
    return q / np.linalg.norm(q)


def q_from_matrix(m: np.ndarray) -> np.ndarray:
    t = np.trace(m)
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        q = [(m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s, 0.25 / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def q_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    d = float(a @ b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + t * (b - a)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - t) * th) * a + np.sin(t * th) * b) / np.sin(th)


def palm_frame(wrist: np.ndarray, index_mcp: np.ndarray, middle_mcp: np.ndarray, pinky_mcp: np.ndarray) -> np.ndarray:
    """Columns: thumb side (x), fingers (y), palm normal (z) of a right hand."""
    y = middle_mcp - wrist
    y /= np.linalg.norm(y)
    x = index_mcp - pinky_mcp
    x -= y * (x @ y)
    x /= np.linalg.norm(x)
    return np.stack([x, y, np.cross(x, y)], axis=1)


# ---- rig ----

@dataclass
class Rig:
    data: dict
    names: List[str] = field(init=False)
    parents: List[int] = field(init=False)
    joints: np.ndarray = field(init=False)
    rest_frame: np.ndarray = field(init=False)
    chains: Dict[str, List[Tuple[int, np.ndarray]]] = field(init=False)  # finger -> [(bone, rest dir)]

    def __post_init__(self) -> None:
        d = self.data
        self.names, self.parents = d["names"], d["parents"]
        self.joints = np.array(d["joints"], float)
        ji = {n: i for i, n in enumerate(self.names)}
        j = lambda n: self.joints[ji[n]]
        self.rest_frame = palm_frame(j("wrist"), j("index-mcp"), j("middle-mcp"), j("pinky-mcp"))
        self.chains = {}
        for f in ("index", "middle", "ring", "pinky", "thumb"):
            names = [f"{f}-{p}" for p in (("cmc", "mcp", "ip") if f == "thumb" else ("mc", "mcp", "pip", "dip"))]
            pts = [j(n) for n in names] + [np.array(d["tips"][f], float)]
            self.chains[f] = [(ji[n], pts[k + 1] - pts[k]) for k, n in enumerate(names)]

    @classmethod
    def load(cls, path: Path = RIG_PATH) -> "Rig":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def to_scene(world: np.ndarray) -> np.ndarray:
    """MediaPipe world axes (x right, y down, z away) -> scene axes (x right, y up, z toward viewer)."""
    return world * np.array([1.0, -1.0, -1.0])


def retarget(rig: Rig, lm: np.ndarray) -> np.ndarray:
    """(21, 3) right-hand landmarks in scene axes -> (20, 4) bone quaternions (bone 0 = wrist, in scene)."""
    frame = palm_frame(lm[0], lm[5], lm[9], lm[17])
    g = frame @ rig.rest_frame.T
    q = np.tile([0.0, 0.0, 0.0, 1.0], (len(rig.names), 1))
    q[0] = q_from_matrix(g)
    for finger, chain in rig.chains.items():
        ids = CHAIN_LM[finger]
        parent = np.array([0.0, 0, 0, 1])  # global rotation of the parent bone, in the hand's own frame
        for k, (bone, rest_dir) in enumerate(chain):
            seg = g.T @ (lm[ids[k + 1]] - lm[ids[k]])
            if np.linalg.norm(seg) < 1e-6:
                continue
            swing = q_between(q_rotate(parent, rest_dir), seg)
            local = q_mul(q_mul(q_conj(parent), swing), parent)
            q[bone] = local
            parent = q_mul(parent, local)
    return q


# ---- per-frame hand poses and the timeline ----

@dataclass
class Pose:
    pos: np.ndarray  # (3,) wrist, canonical space
    quats: np.ndarray  # (20, 4)

    def lerp(self, other: "Pose", t: float) -> "Pose":
        return Pose((1 - t) * self.pos + t * other.pos,
                    np.stack([q_slerp(a, b, t) for a, b in zip(self.quats, other.quats)]))


def q_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return np.array([*(axis * np.sin(angle / 2)), np.cos(angle / 2)])


# Relaxed hand: (mcp, pip, dip) flex in degrees, curling a little more toward the pinky,
# and how far each finger turns in toward the middle finger at the knuckle.
RELAX_FLEX = {"index": (14, 22, 12), "middle": (20, 30, 15), "ring": (26, 36, 18), "pinky": (32, 40, 20)}
RELAX_GATHER = {"index": 3, "middle": -3, "ring": -9, "pinky": -11}  # + turns toward the pinky; closes the bind fan


def rest_pose(rig: Rig) -> Pose:
    """Hands relaxed low in front of the signer: fingers loosely curled, palm angled down (canonical right hand).

    Local rotations are in the rig's bind frame (as `retarget` writes them), so each joint flexes
    about the palm frame's thumb-side axis and gathers about the palm normal.
    """
    x, y, z = rig.rest_frame.T  # thumb side, fingers, palm normal
    q = np.tile([0.0, 0.0, 0.0, 1.0], (len(rig.names), 1))
    for finger, flex in RELAX_FLEX.items():
        bones = [b for b, _ in rig.chains[finger]][1:]  # skip the metacarpal
        for k, (bone, deg) in enumerate(zip(bones, flex)):
            q[bone] = q_axis(x, np.radians(deg))
            if k == 0:
                q[bone] = q_mul(q_axis(z, np.radians(RELAX_GATHER[finger])), q[bone])
    (cmc, t0), (mcp, t1), (ip, t2) = rig.chains["thumb"]
    q[cmc] = q_axis(y, np.radians(-22))  # bring the thumb round in front of the palm
    q[mcp] = q_axis(np.cross(t1, z), np.radians(12))
    q[ip] = q_axis(np.cross(t2, z), np.radians(18))
    fingers = np.array([0.45, -0.25, 0.6])
    fingers /= np.linalg.norm(fingers)
    palm = np.array([0.5, -1.0, 0.0])
    palm -= fingers * (palm @ fingers)
    palm /= np.linalg.norm(palm)
    g = np.stack([np.cross(fingers, palm), fingers, palm], axis=1)
    q[0] = q_from_matrix(g @ rig.rest_frame.T)
    return Pose(np.array([-0.15, -0.19, 0.18]), q)


def _hold_gaps(ok: np.ndarray) -> Optional[np.ndarray]:
    """Index of the nearest earlier tracked frame (or the first tracked one) for every frame."""
    if not ok.any():
        return None
    idx = np.where(ok, np.arange(len(ok)), -1)
    np.maximum.accumulate(idx, out=idx)
    idx[idx < 0] = np.flatnonzero(ok)[0]
    return idx


# Share of a curled finger's bend taken by the knuckle, middle and end joints in a real fist. The tracker
# under-bends the knuckle (about 50 degrees) and folds the end joints instead, which makes fists look stubby.
FIST_SHARE = np.array([90.0, 85.0, 25.0]) / 200.0
CURL_FROM, CURL_TO = 90.0, 170.0  # total finger bend (degrees) where the correction starts and is complete


def natural_curl(rig: Rig, q: np.ndarray) -> np.ndarray:
    """Redistribute each curled finger's flex toward real fist proportions, keeping the total bend."""
    x = rig.rest_frame[:, 0]
    q = q.copy()
    for finger in ("index", "middle", "ring", "pinky"):
        bones = [b for b, _ in rig.chains[finger]][1:]  # mcp, pip, dip
        flex = np.array([2 * np.arctan2(q[b, :3] @ x, q[b, 3]) for b in bones])
        total = np.degrees(flex.sum())
        w = np.clip((total - CURL_FROM) / (CURL_TO - CURL_FROM), 0.0, 1.0)
        w = w * w * (3 - 2 * w)
        if w == 0:
            continue
        delta = w * (flex.sum() * FIST_SHARE - flex)
        for b, d in zip(bones, delta):
            q[b] = q_mul(q_axis(x, d), q[b])
    return q


def hand_poses(rig: Rig, motion: Motion, hand: str) -> Optional[List[Pose]]:
    """Canonical poses for every frame of one hand, or None if the sign never uses it."""
    flat, world = motion.hands.get(hand), motion.world.get(hand)
    if flat is None or world is None:
        return None
    idx = _hold_gaps(~np.isnan(flat[:, 0, 0]) & ~np.isnan(world[:, 0, 0]))
    if idx is None:
        return None
    flip = np.array([-1.0 if MIRROR[hand] else 1.0, 1.0, 1.0])
    cache: Dict[int, Pose] = {}
    poses = []
    for i in idx:
        if i not in cache:
            lm = to_scene(world[i].astype(float)) * flip
            x, y = flat[i, 0]
            pos = np.array([x * BODY_UNIT_M, -y * BODY_UNIT_M, HAND_DEPTH_M]) * flip
            cache[i] = Pose(pos, natural_curl(rig, retarget(rig, lm)))
        poses.append(cache[i])
    return smooth_track(poses)


def smooth_track(poses: List[Pose], sigma: float = 1.2) -> List[Pose]:
    """Gaussian filter over frames (sigma in frames) to take out tracker jitter.

    Positions are averaged directly; quaternions are flipped onto the centre frame's hemisphere,
    averaged and renormalised, which is accurate for the small angles between neighbouring frames.
    """
    if len(poses) < 3 or sigma <= 0:
        return poses
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    pos = np.stack([p.pos for p in poses])
    quats = np.stack([p.quats for p in poses])
    n = len(poses)
    out = []
    for i in range(n):
        j = np.clip(np.arange(i - r, i + r + 1), 0, n - 1)
        w = k / k.sum()
        q = quats[j] * np.where(np.sum(quats[j] * quats[i], -1, keepdims=True) < 0, -1.0, 1.0)
        q = np.tensordot(w, q, 1)
        out.append(Pose(w @ pos[j], q / np.linalg.norm(q, axis=-1, keepdims=True)))
    return out


def min_jerk(t: float) -> float:
    """Minimum-jerk profile: zero velocity and acceleration at both ends, like a reaching arm."""
    return t * t * t * (10 - 15 * t + 6 * t * t)


def glide_frames(a: Pose, b: Pose, base: int) -> int:
    """Transition length grows with how far the wrist travels (0.25 m is a typical move)."""
    d = float(np.linalg.norm(b.pos - a.pos))
    return max(int(round(base * np.clip(0.5 + 2.0 * d, 0.6, 1.6))), 2)


@dataclass
class Timeline3D:
    frames: np.ndarray  # (N, 2, 3 + 4 * bones): per HANDS entry, wrist position then bone quaternions
    fps: float
    segments: List[Tuple[int, int, str]]


def active_span(tracks: Dict[str, Optional[List[Pose]]]) -> Tuple[int, int]:
    """First and last frame where some hand is up in signing space; the lift in and drop out are left to the glide."""
    ys = np.max([[p.pos[1] for p in t] for t in tracks.values() if t is not None], axis=0)
    up = np.flatnonzero(ys > LIFT_Y_M)
    return (int(up[0]), int(up[-1])) if len(up) else (0, len(ys) - 1)


Tracks = Dict[str, Optional[List[Pose]]]  # per HANDS entry, None for a hand the sign does not use


def sign_track(rig: Rig, motion: Motion, speed: float = 1.0, fps: float = FPS) -> Optional[Tracks]:
    """One clip resampled to `fps` (sped up by `speed`) and cut to the part where the hands are up."""
    m = motion.trimmed()
    step = m.fps / fps * speed
    n = max(int(round(m.frames / step)), 1)
    pick = np.minimum(np.round(np.arange(n) * step).astype(int), m.frames - 1)
    tracks = {h: hand_poses(rig, m, h) for h in HANDS}
    if all(t is None for t in tracks.values()):
        return None
    lo, hi = active_span(tracks)
    pick = pick[(pick >= lo) & (pick <= hi)]
    if not len(pick):
        pick = np.array([lo])
    return {h: None if t is None else [t[i] for i in pick] for h, t in tracks.items()}


def compose3d(rig: Rig, items: Sequence[Tuple[Tracks, str]], gap_s: float = 0.0, speed: float = 1.0,
              transition_s: float = TRANSITION_S, fps: float = FPS) -> Timeline3D:
    """Chain signs: each hand glides (slerp) from where it was into the next sign, and back to rest at the end.

    Items are (tracks, label) from `sign_track` or `fingerspell.spell_track`; a hand a sign leaves out rests.
    """
    rest = rest_pose(rig)
    n_trans = max(int(round(transition_s * fps / speed)), 2)
    out: List[List[Pose]] = []
    cur = {h: rest for h in HANDS}
    segments = []

    def glide(target: Dict[str, Pose]) -> None:
        n = max(glide_frames(cur[h], target[h], n_trans) for h in HANDS)
        for k in range(1, n + 1):
            t = min_jerk(k / n)
            out.append([cur[h].lerp(target[h], t) for h in HANDS])

    for tracks, label in items:
        if tracks is None or all(t is None for t in tracks.values()):
            continue
        n = max(len(t) for t in tracks.values() if t is not None)
        start = len(out)
        glide({h: tracks[h][0] if tracks[h] else rest for h in HANDS})
        for i in range(n):
            out.append([tracks[h][min(i, len(tracks[h]) - 1)] if tracks[h] else rest for h in HANDS])
        cur = dict(zip(HANDS, out[-1]))
        segments.append((start, len(out) - 1, label))
        out.extend([list(out[-1])] * int(round(gap_s * fps)))
    if not out:
        raise MotionError("nothing to compose (no 3D hand data; re-run setup_clips.py)")
    glide({h: rest for h in HANDS})
    arr = np.array([[np.concatenate([p.pos, p.quats.ravel()]) for p in row] for row in out], np.float32)
    return Timeline3D(arr, fps, segments)


# ---- player page ----

def _blob(arrays: List[Tuple[str, np.ndarray]]) -> Tuple[List[list], str]:
    raw, layout, off = b"", [], 0
    for name, a in arrays:
        pad = (-off) % 2
        raw += b"\0" * pad
        off += pad
        layout.append([name, a.dtype.str, off, int(a.size)])
        raw += a.tobytes()
        off += a.nbytes
    return layout, base64.b64encode(gzip.compress(raw, 9)).decode()


def pack_rig(rig: Rig) -> dict:
    d = dict(rig.data)
    layout, blob = _blob([
        ("verts", np.round(np.array(d.pop("verts")) * 1e5).astype("<i2")),
        ("faces", np.array(d.pop("faces")).astype("<u2")),
        ("skinIdx", np.array(d.pop("skinIdx")).astype("u1")),
        ("skinW", np.round(np.array(d.pop("skinW")) * 255).astype("u1")),
    ])
    d.update(layout=layout, blob=blob)
    return d


def pack_timeline(tl: Timeline3D) -> dict:
    f = tl.frames.copy()
    f[:, :, :3] *= 1e4  # 0.1 mm
    f[:, :, 3:] *= 32767
    layout, blob = _blob([("frames", np.round(f).astype("<i2"))])
    return {"n": len(f), "stride": f.shape[2], "fps": tl.fps, "segments": tl.segments,
            "mirror": [MIRROR[h] for h in HANDS], "layout": layout, "blob": blob}


def write_player(tl: Timeline3D, out: Path, title: str, rig: Optional[Rig] = None) -> Path:
    rig = rig or Rig.load()
    html = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = (html.replace("/*TITLE*/", title)
                .replace("/*RIG*/", json.dumps(pack_rig(rig), separators=(",", ":")))
                .replace("/*TIMELINE*/", json.dumps(pack_timeline(tl), separators=(",", ":")))
                .replace("/*LIBRARY*/", "null"))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main(argv: Optional[List[str]] = None) -> None:
    from clips import ClipIndex
    from fingerspell import spell_track
    from playlist import FINGERSPELL, build_playlist

    ap = argparse.ArgumentParser(description="Bake a gloss list into a 3D hand player page")
    ap.add_argument("glosses", nargs="+")
    ap.add_argument("-o", "--out", type=Path, default=Path("exports") / "signs3d.html")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--gap", type=float, default=0.1, help="pause between signs, seconds")
    ap.add_argument("--open", action="store_true", help="open the page in the browser")
    args = ap.parse_args(argv)

    rig = Rig.load()
    playlist = build_playlist(args.glosses, ClipIndex.load(), on_status=lambda e: print(f"[{e.state}] {e.detail}"),
                              authored_letters=True)
    items = []
    for item in playlist.items:  # words without a clip are spelled with the authored alphabet, not letter clips
        tracks = (spell_track(rig, item.gloss, speed=args.speed) if item.kind == FINGERSPELL
                  else sign_track(rig, Motion.load(item.clips[0].path), speed=args.speed))
        items.append((tracks, item.gloss))
    tl = compose3d(rig, items, gap_s=args.gap, speed=args.speed)
    path = write_player(tl, args.out, " ".join(g.upper() for g in args.glosses), rig)
    print(f"wrote {path} ({len(tl.frames)} frames, {path.stat().st_size // 1024} KB)")
    if args.open:
        webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    main()
