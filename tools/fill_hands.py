"""Fill hands the tracker lost while their arm was up, in every indexed motion file, from the arm positions
signcheck.py cached in clips/pose/ (new clips get this in animate.extract_motion). Then re-run signcheck.

    python tools/fill_hands.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from animate import fill_lost_hands  # noqa: E402
from clips import ClipIndex  # noqa: E402
from motion import HANDS, Motion  # noqa: E402
from signcheck import pose_path  # noqa: E402


def main() -> None:
    import os
    os.chdir(ROOT)
    index = ClipIndex.load()
    clips = files = frames = 0
    for gloss in index.glosses:
        for clip in index.lookup(gloss):
            clips += 1
            if not pose_path(clip).is_file():
                continue
            m = Motion.load(clip.path)
            if not all(h in m.hands and h in m.world for h in HANDS):
                continue
            arms = np.load(pose_path(clip))
            n = min(len(arms), m.frames)
            hands = {h: m.hands[h][:n] for h in HANDS}
            world = {h: m.world[h][:n] for h in HANDS}
            got = fill_lost_hands(hands, world, arms[:n])
            if got:
                Motion(hands, m.fps, world).save(clip.path)
                files += 1
                frames += got
    print(f"filled {frames} frames in {files} of {clips} clips")


if __name__ == "__main__":
    main()
