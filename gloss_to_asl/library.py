"""Every sign, the alphabet and a few phrases in one 3D player page.

`python -m gloss_to_asl.library` bakes the best clip of every gloss in the index (tracked motion, cut to the part
where the hands are up, at LIB_FPS) and the authored A-Z handshapes into one self-contained page.
The page chains signs in the browser, so any typed sentence plays; words without a sign are spelled.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from gloss_to_asl.aliases import DROP, resolve_aliases
from gloss_to_asl.clips import ClipIndex
from gloss_to_asl.fingerspell import ALPHABET, CHANGE_S, DOUBLE_SHIFT_M, HOLD_S, PATH_S, letter_frames, letter_pose
from gloss_to_asl.hand3d import HANDS, MIRROR, TEMPLATE_PATH, Pose, Rig, Tracks, _blob, pack_rig, rest_pose, sign_track
from gloss_to_asl.motion import FPS, TRANSITION_S, Motion, MotionError

LIB_FPS = 15.0  # stored rate; the page upsamples to FPS when it chains signs
GAP_S = 0.1
POS_SCALE = 1e4  # 0.1 mm
FAILED_FAULTS = 10  # signcheck.py scores a serious problem 10, a warning 1
GLOSS_JS = Path(__file__).resolve().parent / "assets" / "hand" / "gloss.js"  # English -> ASL sign order
NOUNS_PATH = GLOSS_JS.with_name("nouns.txt")  # glosses that are mainly nouns (tools/build_nouns.py)

# Shown as one-click examples; every word must be in the vocabulary unless listed in SPELLED.
PRESETS = [
    "HELLO NICE MEET YOU",
    "THANK YOU",
    "MY NAME VEDANT",
    "WHAT YOUR NAME",
    "I WANT LEARN SIGN LANGUAGE",
    "WHERE BATHROOM",
    "HOW YOU",
    "SEE YOU LATER",
    "GOOD MORNING",
    "PLEASE HELP ME",
    "I DON'T WANT",
    "I LOVE YOU",
]
SPELLED = {"VEDANT"}


def agent_track(rig: Rig, person: Tracks) -> Tracks:
    """The agent ending (TEACH + AGENT = TEACHER): the PERSON sign's movement, both hands down the sides, with
    flat hands instead of P hands. Fingers straight and together as in B, thumb relaxed beside the palm."""
    q = letter_pose(rig, "B").quats.copy()
    thumb = [b for b, _ in rig.chains["thumb"]]
    q[thumb] = rest_pose(rig).quats[thumb]
    return {h: None if t is None else [Pose(p.pos, np.concatenate([p.quats[:1], q[1:]])) for p in t]
            for h, t in person.items()}


def vocabulary(index: ClipIndex, keep_failed: bool = False) -> List[str]:
    """Glosses that get a sign. One-letter glosses are left to the authored alphabet, except the pronoun I.
    A gloss whose best signing fails a serious sign check (signcheck.py: wrong number of hands, wrong place)
    is left out, so the word is spelled instead of signed wrongly."""
    return [g for g in index.glosses if (len(g) > 1 or g == "I")
            and (keep_failed or index.best(g).faults < FAILED_FAULTS)]


def failed_glosses(index: ClipIndex) -> List[str]:
    return [g for g in index.glosses if index.best(g).faults >= FAILED_FAULTS]


def _canon(q: np.ndarray) -> np.ndarray:
    return q * np.where(q[..., 3:4] < 0, -1.0, 1.0)


def _delta(a: np.ndarray) -> np.ndarray:
    """Frame-to-frame differences (wrapping in the array's integer type), which compress far better."""
    return np.diff(a, axis=0, prepend=np.zeros((1, a.shape[1]), a.dtype)).astype(a.dtype)


def pack_library(rig: Rig, index: ClipIndex, glosses: Sequence[str],
                 progress: Optional[Callable[[int, int], None]] = None) -> dict:
    """Signs as quantised frames plus the letters, rest pose and timing the page needs to chain them.

    A sign's hands are stored one after the other, each as n rows of differences from the previous row.
    Per hand-frame: `head` int16 (wrist position in 0.1 mm, wrist quaternion * 32767) and `bones` int8
    (x, y, z of each finger-bone quaternion * 127 with w >= 0; w is rebuilt in the page).
    """
    head: List[np.ndarray] = []
    bones: List[np.ndarray] = []
    meta: Dict[str, List[int]] = {}
    k = 0
    made: Dict[str, Tracks] = {}  # PERSON, which AGENT is built from

    def track(gloss: str) -> Optional[Tracks]:
        if gloss == "AGENT":
            return agent_track(rig, made["PERSON"]) if "PERSON" in made else None
        clip = index.best(gloss)
        try:
            return sign_track(rig, Motion.load(clip.path), fps=LIB_FPS) if clip else None
        except MotionError:
            return None

    for i, gloss in enumerate(list(glosses) + ["AGENT"]):
        if progress:
            progress(i, len(glosses))
        tracks = track(gloss)
        if tracks is None:
            continue
        if gloss == "PERSON":
            made[gloss] = tracks
        n = max(len(t) for t in tracks.values() if t is not None)
        mask = 0
        for bit, h in enumerate(HANDS):
            if tracks[h] is None:
                continue
            mask |= 1 << bit
            q = _canon(np.stack([p.quats for p in tracks[h]]))
            pos = np.stack([p.pos for p in tracks[h]])
            head.append(_delta(np.round(np.concatenate([pos * POS_SCALE, q[:, 0] * 32767], 1)).astype("<i2")))
            bones.append(_delta(np.round(q[:, 1:, :3].reshape(n, -1) * 127).astype("i1")))
        meta[gloss] = [k, n, mask]
        k += n * bin(mask).count("1")
    layout, blob = _blob([("head", np.concatenate(head).ravel()), ("bones", np.concatenate(bones).ravel())])
    flat = lambda p: [round(float(v), 5) for v in np.concatenate([p.pos, p.quats.ravel()])]
    draw = int(round(PATH_S * FPS))
    nouns = [g for g in NOUNS_PATH.read_text(encoding="utf-8").split() if g in meta]
    alias, _, tags = resolve_aliases(meta, nouns)
    return {
        "fps": LIB_FPS, "outFps": FPS, "posScale": POS_SCALE, "nBones": len(rig.names),
        "hands": list(HANDS), "mirror": [MIRROR[h] for h in HANDS],
        "transition": TRANSITION_S, "gap": GAP_S,
        "hold": HOLD_S, "change": CHANGE_S, "doubleShift": DOUBLE_SHIFT_M,
        "rest": flat(rest_pose(rig)),
        "letters": {c: [flat(p) for p in letter_frames(rig, c, draw if ALPHABET[c].path else 1)] for c in ALPHABET},
        "signs": meta, "presets": PRESETS, "alias": alias, "drop": DROP,
        "tags": tags, "nouns": nouns,
        "layout": layout, "blob": blob,
    }


def write_library(lib: dict, out: Path, rig: Rig, title: str = "ASL signs and alphabet") -> Path:
    html = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = (html.replace("/*TITLE*/", title)
                .replace("/*RIG*/", json.dumps(pack_rig(rig), separators=(",", ":")))
                .replace("/*TIMELINE*/", "null")
                .replace("/*GLOSS*/", GLOSS_JS.read_text(encoding="utf-8"))
                .replace("/*LIBRARY*/", json.dumps(lib, separators=(",", ":"))))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def check_presets(glosses: Sequence[str]) -> None:
    """Fail the build if an example phrase would fall back to spelling a word that should be signed."""
    vocab = set(glosses)
    for phrase in PRESETS:
        words = phrase.split()
        i = 0
        while i < len(words):
            for n in range(min(4, len(words) - i), 0, -1):
                if " ".join(words[i:i + n]) in vocab:
                    i += n
                    break
            else:
                if words[i] not in SPELLED:
                    raise SystemExit(f"preset {phrase!r}: no sign for {words[i]!r}")
                i += 1


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Bake every sign and the alphabet into one 3D player page")
    ap.add_argument("-o", "--out", type=Path, default=Path("exports") / "library.html")
    ap.add_argument("--limit", type=int, help="only the first N glosses (for a quick test build)")
    ap.add_argument("--open", action="store_true", help="open the page in the browser")
    ap.add_argument("--keep-failed", action="store_true", help="also sign glosses that fail a serious sign check")
    args = ap.parse_args(argv)

    rig, index = Rig.load(), ClipIndex.load()
    glosses = vocabulary(index, args.keep_failed)
    if not args.keep_failed:
        failed = failed_glosses(index)
        print(f"left out {len(failed)} signs that failed a sign check (spelled instead):", " ".join(failed))
    check_presets(glosses)
    if args.limit:
        glosses = glosses[:args.limit]
    lib = pack_library(rig, index, glosses,
                       progress=lambda i, n: print(f"\r{i + 1}/{n}", end="", flush=True) if i % 50 == 0 else None)
    print(f"\r{len(lib['signs'])} of {len(glosses)} signs packed, {len(lib['alias'])} more words play them")
    for note in resolve_aliases(lib["signs"])[1]:
        print("alias left out:", note)
    path = write_library(lib, args.out, rig)
    print(f"wrote {path} ({path.stat().st_size // 1024} KB)")
    if args.open:
        webbrowser.open(path.resolve().as_uri())


if __name__ == "__main__":
    main()
