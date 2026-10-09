"""Check each sign's tracked motion against how ASL forms the sign.

Three kinds of check:
- Tracking faults that make any sign look wrong (serious, "bad"): too few frames, no hand raised, a hand that
  jumps across the body between frames (the tracker swapped the hands), and a hand the tracker lost while
  the pose tracker shows its arm raised (common when the hands touch or cross), which plays as a missing hand.
- ASL form, for glosses in ASL-LEX 2.0 (Caselli et al. 2017, Sehyr et al. 2021; CC BY 4.0), which codes
  every sign by sign type (one hand or two), major location (head, body, other hand, neutral space) and the
  selected fingers of the handshape. The motion should agree: a one-handed sign moves one hand, a two-handed
  sign two; a head sign reaches the head; an index-finger handshape extends the index and not the rest.
  ASL-LEX records one way of signing a word and WLASL signers often use another (SOON, GRAB, EXPERT), so a
  mismatch is a warning: it picks the closest of a word's signings but never drops the word.
  Signs not in ASL-LEX get Battison's symmetry condition instead: when both hands move, they share a
  handshape.
- Agreement: with three or more signings of a gloss, one whose strong hand moves unlike the others (which
  agree with each other) is flagged; it is more likely mislabelled or mistracked.

`python signcheck.py --apply` stores each clip's faults in clips/index.json, so the best-formed signing of a
gloss is the one that plays, and writes clips/sign_check.json. library.py leaves out a gloss whose best
signing still has a serious fault; the word is spelled instead. The lost-hand check needs the source videos
and caches arm positions in clips/pose/.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from animate import REST_Y, arm_raised
from motion import HANDS, Motion

ASLLEX_URL = "https://osf.io/download/9nygd/"  # ASL-LEX 2.0 signdata.csv, https://osf.io/zpha4/
ASLLEX_PATH = Path("datasets") / "asllex" / "signdata.csv"
REPORT_PATH = Path("clips") / "sign_check.json"
POSE_DIR = Path("clips") / "pose"

TIPS, PIPS, MCPS = (8, 12, 16, 20), (6, 10, 14, 18), (5, 9, 13, 17)
FINGERS = "imrp"  # index, middle, ring, pinky (ASL-LEX selected-finger letters)

# Thresholds in body units (1.0 = shoulder width, origin mid-shoulder, +y down).
HEAD_Y = -0.45      # a fingertip above this is at the face or head
MOVE_MIN = 0.25     # wrist path length that counts as a moving hand
USED_MIN = 0.35     # share of frames a hand must be up to count as used
CONSENSUS = 0.35   # mean wrist distance between two signings that counts as moving differently
JUMP = 0.6          # wrist move in one frame (at 30 fps) that only a hand swap explains
EXTENDED = 0.8      # tip-to-knuckle distance over finger length for a straight finger
MIN_FRAMES = 8
LOW_TRACKING = 0.25  # share of frames with a raised hand below which tracking is poor
LOST = 0.5          # share of an arm's raised frames with no tracked hand that plays as a missing hand
# Highest fingertip a head sign must reach, by ASL-LEX minor location (calibrated on 1377 WLASL signs: about
# the 90th percentile of the signs coded at each place).
HEAD_TOP = {"Forehead": -0.45, "Eye": -0.45, "HeadAway": -0.3, "Other": -0.35, "CheekNose": -0.3,
            "Mouth": -0.15, "Chin": -0.15, "UnderChin": -0.1}


@dataclass
class Form:
    """What the motion shows."""
    frames: int
    used: Dict[str, float]          # share of frames each hand is tracked and raised
    path: Dict[str, float]          # wrist path length while raised
    top: Dict[str, float]           # highest fingertip (min y)
    extended: Dict[str, np.ndarray]  # per hand, share of raised frames each finger (imrp) is straight
    jumps: int
    dominant: str
    hands_used: int


@dataclass
class Check:
    gloss: str
    issues: List[str] = field(default_factory=list)  # each "severity: text", severity bad or warn
    reference: Optional[str] = None                    # the ASL-LEX entry the sign matched best

    @property
    def bad(self) -> int:
        return sum(i.startswith("bad") for i in self.issues)

    @property
    def warn(self) -> int:
        return sum(i.startswith("warn") for i in self.issues)


def _straightness(world: np.ndarray) -> np.ndarray:
    """(T, 4): per finger, knuckle-to-tip distance over the summed bone lengths (1 = straight)."""
    out = np.full((len(world), 4), np.nan, np.float32)
    for f, (m, t) in enumerate(zip(MCPS, TIPS)):
        bones = sum(np.linalg.norm(world[:, j + 1] - world[:, j], axis=1) for j in range(m, t))
        out[:, f] = np.linalg.norm(world[:, t] - world[:, m], axis=1) / np.maximum(bones, 1e-6)
    return out


def form(motion: Motion) -> Form:
    used, path, top, ext = {}, {}, {}, {}
    jumps = 0
    for h in HANDS:
        xy = motion.hands.get(h)
        if xy is None:
            used[h], path[h], top[h], ext[h] = 0.0, 0.0, np.inf, np.zeros(4)
            continue
        wrist = xy[:, 0]
        up = ~np.isnan(wrist[:, 0]) & (wrist[:, 1] < REST_Y)
        used[h] = float(up.mean()) if len(up) else 0.0
        step = np.linalg.norm(np.diff(wrist, axis=0), axis=1)
        both = up[1:] & up[:-1]
        path[h] = float(np.nansum(step[both]))
        jumps += int(np.sum(step[both] > JUMP))
        tips = xy[up][:, TIPS, 1]
        top[h] = float(np.nanmin(tips)) if tips.size else np.inf
        w = motion.world.get(h)
        if w is not None and up.any():
            s = _straightness(w[up])
            ext[h] = np.nanmean(s > EXTENDED, axis=0)
        else:
            ext[h] = np.zeros(4)
    score = {h: path[h] + 2 * used[h] for h in HANDS}
    dominant = max(HANDS, key=score.get)
    weak = next(h for h in HANDS if h != dominant)
    # the weak hand counts when it is up for a good part of the time the strong hand is (videos lead in and
    # out at rest, so absolute shares vary)
    two = used[weak] >= max(USED_MIN * used[dominant], 0.1)
    return Form(motion.frames, used, path, top, ext, jumps, dominant, 2 if two else int(used[dominant] > 0.05))


def trajectory(motion: Motion, f: Form, n: int = 24) -> Optional[np.ndarray]:
    """The strong hand's wrist path while raised, resampled to n points, mirrored for a left-handed signer."""
    xy = motion.hands.get(f.dominant)
    if xy is None:
        return None
    w = xy[:, 0]
    up = ~np.isnan(w[:, 0]) & (w[:, 1] < REST_Y)
    if up.sum() < 2:
        return None
    w = w[up] * (np.array([-1.0, 1.0]) if f.dominant == "Left" else 1.0)
    t = np.linspace(0, len(w) - 1, n)
    return np.stack([np.interp(t, np.arange(len(w)), w[:, k]) for k in range(2)], 1)


def odd_ones(paths: Sequence[Optional[np.ndarray]]) -> List[bool]:
    """With three or more signings, flag one that moves unlike the others while those agree with each other."""
    out = [False] * len(paths)
    ok = [i for i, p in enumerate(paths) if p is not None]
    if len(ok) < 3:
        return out
    d = {(i, j): float(np.linalg.norm(paths[i] - paths[j], axis=1).mean()) for i in ok for j in ok if i != j}
    for i in ok:
        mine = np.median([d[i, j] for j in ok if j != i])
        rest = [d[j, k] for j in ok for k in ok if j < k and i not in (j, k)]
        out[i] = mine > CONSENSUS and np.median(rest) < 0.6 * mine
    return out


# Selected fingers expected straight, by ASL-LEX flexion; closed and bent handshapes check only that the
# unselected fingers are not straight.
def _finger_issue(entry: dict, ext: np.ndarray) -> Optional[str]:
    sel = entry["SelectedFingers.2.0"]
    flex = entry["Flexion.2.0"]
    if not sel or sel == "t" or flex in ("NA", "") or entry["FlexionChange.2.0"] == "1":
        return None
    picked = [FINGERS.index(c) for c in sel if c in FINGERS]
    others = [k for k in range(4) if k not in picked]
    if "m" in sel and "i" not in sel:  # the 8 handshapes: the other fingers stay spread and straight
        others = []
    if flex == "FullyOpen" and picked and ext[picked].mean() < 0.25:
        return f"selected fingers ({sel}) are not straight"
    # unselected fingers are folded in every ASL handshape; straight ones mean a different handshape
    if others and flex in ("FullyOpen", "Bent", "Flat") and ext[others].mean() > 0.75:
        return f"fingers outside {sel} are straight"
    if flex == "FullyClosed" and ext.mean() > 0.75:
        return "an open hand where ASL closes it"
    return None


def lost_hands(motion: Motion, arms: np.ndarray) -> Dict[str, float]:
    """Per hand, the share of the frames its arm is raised (pose tracker) in which the hand is not tracked."""
    out = {}
    n = min(len(arms), motion.frames)
    raised = arm_raised(arms[:n])
    for k, h in enumerate(HANDS):
        xy = motion.hands.get(h)
        tracked = np.zeros(n, bool) if xy is None else ~np.isnan(xy[:n, 0, 0])
        if raised[:, k].mean() > 0.2:
            out[h] = float((raised[:, k] & ~tracked).sum() / raised[:, k].sum())
    return out


def check(gloss: str, motion: Motion, entries: Sequence[dict] = (), f: Optional[Form] = None,
          arms: Optional[np.ndarray] = None) -> Check:
    """`arms`: the pose tracker's wrist positions for the same frames (animate.arm_track), if known."""
    c = Check(gloss)
    f = f or form(motion)
    if f.frames < MIN_FRAMES:
        c.issues.append(f"bad: only {f.frames} frames")
    if f.hands_used == 0:
        c.issues.append("bad: no hand is raised")
        return c
    if f.jumps:
        c.issues.append(f"bad: a hand jumps across the frame {f.jumps} time(s) (tracker swapped hands)")
    for h, share in (lost_hands(motion, arms) if arms is not None else {}).items():
        if share > LOST:
            c.issues.append(f"bad: the tracker loses the signer's {h.lower()} hand for {share:.0%} of the time "
                            "that arm is up")
    if max(f.used.values()) < LOW_TRACKING:
        c.issues.append(f"warn: hands tracked in only {max(f.used.values()):.0%} of frames")
    if not entries:
        weak = next(h for h in HANDS if h != f.dominant)
        if f.hands_used == 2 and f.path[weak] > MOVE_MIN and f.path[f.dominant] > MOVE_MIN:
            if np.abs(f.extended[weak] - f.extended[f.dominant]).max() > 0.8:
                c.issues.append("warn: both hands move but their handshapes differ (symmetry condition)")
        return c
    # the closest ASL-LEX variant decides; a sign only has to match one way of signing the word
    best: Optional[List[str]] = None
    for e in entries:
        found = _against(e, f)
        if best is None or sum(i.startswith("bad") for i in found) * 10 + len(found) < \
                sum(i.startswith("bad") for i in best) * 10 + len(best):
            best, c.reference = found, e["EntryID"]
    c.issues += best or []
    return c


def _against(e: dict, f: Form) -> List[str]:
    out = []
    kind = e["SignType.2.0"]
    if kind == "OneHanded" and f.hands_used == 2:
        weak = next(h for h in HANDS if h != f.dominant)
        if f.path[weak] > 2 * MOVE_MIN:
            out.append("warn: ASL-LEX signs it with one hand; both hands move here")
    if kind.startswith(("Symmetrical", "Asymmetrical")) and f.hands_used < 2:
        out.append("warn: ASL-LEX signs it with two hands; one hand here")
    loc, minor = e["MajorLocation.2.0"], e["MinorLocation.2.0"]
    top = min(f.top.values())  # either hand: the strong one reaches the place, the weak one may be a base
    if loc == "Head" and top > HEAD_TOP.get(minor, -0.3):
        out.append("warn: ASL-LEX signs it at the head; the hands stay below it here")
    if loc == "Neutral" and minor == "Neutral" and top < HEAD_Y - 0.5:
        out.append("warn: ASL-LEX signs it in front of the body; the hand goes above the head here")
    issue = _finger_issue(e, f.extended[f.dominant])
    if issue:
        out.append("warn: handshape: " + issue)
    return out


def load_asllex(path: Path = ASLLEX_PATH) -> Dict[str, List[dict]]:
    """Gloss -> ASL-LEX entries (FATHER, and variants like BAT_1, BAT_2 under BAT). Empty if absent."""
    if not Path(path).is_file():
        return {}
    out: Dict[str, List[dict]] = {}
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(lex_key(row["EntryID"]), []).append(row)
    return out


def lex_key(entry_id: str) -> str:
    """The gloss an ASL-LEX entry signs: bat_1 -> BAT, back_up -> BACK UP."""
    key = entry_id.rsplit("_", 1)[0] if entry_id[-2:-1] == "_" else entry_id
    return key.upper().replace("_", " ")


def pose_path(clip) -> Path:
    return POSE_DIR / clip.gloss.lower() / f"{clip.clip_id}.npy"


def _arm_job(job: tuple) -> Optional[str]:
    from animate import arm_track

    out, video, start, end = job
    try:
        arms = arm_track(video, start, end)
    except Exception as exc:
        return f"{video}: {type(exc).__name__}: {exc}"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arms)
    return None


def cache_arms(index, clips_dir: Path, workers: int) -> None:
    """Track the arms (pose) of every clip whose source video is kept and has no cached arms yet."""
    from concurrent.futures import ProcessPoolExecutor

    from clips import WLASL_DIR
    from setup_clips import video_ranges

    ranges = video_ranges()
    jobs = []
    for gloss in index.glosses:
        for clip in index.lookup(gloss):
            vid = clip.clip_id.split("-", 1)[1]
            video = WLASL_DIR / clip.gloss.lower() / f"{vid}.mp4"
            if not pose_path(clip).is_file() and video.is_file() and vid in ranges:
                jobs.append((str(pose_path(clip)), str(video), *ranges[vid]))
    if not jobs:
        return
    print(f"tracking the arms in {len(jobs)} videos ...", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for err in pool.map(_arm_job, jobs, chunksize=4):
            if err:
                print("arm tracking failed:", err)


def check_index(index, lex: Dict[str, List[dict]]) -> Dict[str, List[tuple]]:
    """Every clip of every gloss: gloss -> [(clip, Check)], with signings that disagree with the rest flagged.
    Arm positions cached by cache_arms add the lost-hand check."""
    out = {}
    for gloss in index.glosses:
        rows = []
        for clip in index.lookup(gloss):
            try:
                m = Motion.load(clip.path)
            except Exception as exc:  # unreadable file: the worst score
                c = Check(gloss, [f"bad: unreadable motion ({exc})"])
                rows.append((clip, c, None))
                continue
            f = form(m)
            arms = np.load(pose_path(clip)) if pose_path(clip).is_file() else None
            rows.append((clip, check(gloss, m, lex.get(gloss, ()), f, arms), trajectory(m, f)))
        for (clip, c, _), odd in zip(rows, odd_ones([r[2] for r in rows])):
            if odd:
                c.issues.append(f"warn: moves unlike the other {len(rows) - 1} signings of {gloss}")
        out[gloss] = [(clip, c) for clip, c, _ in rows]
    return out


def faults(c: Check) -> int:
    return 10 * c.bad + c.warn


def main(argv: Optional[List[str]] = None) -> None:
    import argparse

    from clips import CLIPS_DIR, INDEX_FILENAME, ClipIndex

    ap = argparse.ArgumentParser(description="Check every sign's motion against ASL form")
    ap.add_argument("--apply", action="store_true",
                    help="store each clip's faults in the index, so the best-formed signing of a gloss plays")
    ap.add_argument("--workers", type=int, default=8, help="processes for tracking the arms in new videos")
    args = ap.parse_args(argv)
    index = ClipIndex.load()
    cache_arms(index, CLIPS_DIR, args.workers)
    lex = load_asllex()
    if not lex:
        print(f"no ASL-LEX at {ASLLEX_PATH} (download {ASLLEX_URL}); only tracking faults are checked")
    checked = check_index(index, lex)
    if args.apply:
        path = CLIPS_DIR / INDEX_FILENAME
        data = json.loads(path.read_text(encoding="utf-8"))
        score = {clip.clip_id: faults(c) for rows in checked.values() for clip, c in rows}
        for entry in data["clips"]:
            entry["faults"] = score.get(entry["clip_id"], 0)
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")
        index = ClipIndex.load()
    report = {}
    for gloss, rows in checked.items():
        best = index.best(gloss)
        clip, c = next((clip, c) for clip, c in rows if clip.clip_id == best.clip_id)
        report[gloss] = {"clip": clip.clip_id, "signings": len(rows), "reference": c.reference, "issues": c.issues,
                         "others": {o.clip_id: oc.issues for o, oc in rows if o.clip_id != clip.clip_id}}
    REPORT_PATH.write_text(json.dumps(report, indent=1), encoding="utf-8")
    bad = sorted(g for g, r in report.items() if any(i.startswith("bad") for i in r["issues"]))
    warn = sorted(g for g, r in report.items() if r["issues"] and g not in bad)
    print(f"{len(report)} signs ({sum(r['signings'] for r in report.values())} signings), "
          f"{sum(1 for r in report.values() if r['reference'])} checked against ASL-LEX: "
          f"{len(bad)} still bad, {len(warn)} with warnings -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
