from __future__ import annotations

import numpy as np

from animate import MAX_GAP_FRAMES, clean_tracks
from render import BACKGROUND, SKIN, _draw_hand


def hand(x=100.0, y=100.0, size=60.0):
    """Plausible 21-point hand: wrist at the bottom, fingers fanning upward."""
    pts = np.zeros((21, 2), np.float32)
    pts[0] = (x, y + size)
    for f, (dx, base) in enumerate(((-0.45, 1), (-0.25, 5), (0.0, 9), (0.25, 13), (0.45, 17))):
        for j in range(4):
            idx = base + j
            pts[idx] = (x + dx * size * (0.6 + 0.25 * j), y + size * (0.55 - 0.3 * j))
    return pts


def test_short_gap_is_interpolated():
    a, b = hand(x=100), hand(x=200)
    tracks = [{"Left": a}, {}, {}, {"Left": b}]
    out = clean_tracks(tracks)
    assert all("Left" in t for t in out)
    xs = [t["Left"][0][0] for t in out]
    assert xs == sorted(xs)


def test_long_gap_is_not_invented():
    tracks = [{"Left": hand()}] + [{} for _ in range(MAX_GAP_FRAMES + 3)] + [{"Left": hand()}]
    out = clean_tracks(tracks)
    assert out[1] == {} and out[-2] == {}


def test_hands_tracked_independently():
    out = clean_tracks([{"Left": hand(x=50), "Right": hand(x=150)}] * 3)
    assert set(out[1]) == {"Left", "Right"}


def test_draw_hand_paints_skin_only_near_the_hand():
    canvas = np.empty((300, 300, 3), np.uint8)
    canvas[:] = BACKGROUND
    _draw_hand(canvas, hand(x=150, y=120, size=80))
    skin_pixels = np.all(canvas == SKIN, axis=-1)
    assert skin_pixels.sum() > 500
    assert not skin_pixels[:20].any() and not skin_pixels[:, :20].any()
