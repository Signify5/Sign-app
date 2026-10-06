from __future__ import annotations

import numpy as np
import pytest

from clips import Clip
from export import ExportError, build_timeline, export_playlist
from motion import FPS, REST, Motion, MotionError, compose
from playlist import Playlist, PlaylistItem


def motion(x: float, frames: int = 30, hand: str = "Right") -> Motion:
    arr = np.tile(REST[hand], (frames, 1, 1)).astype(np.float32)
    arr[:, :, 0] += x
    arr[:, :, 1] -= 2.0  # lift above rest so it differs from the rest pose
    return Motion({hand: arr}, FPS)


def test_compose_orders_signs_and_labels_them():
    tl = compose([(motion(0.0), "A"), (motion(0.5), "B")])
    assert [s[2] for s in tl.segments] == ["A", "B"]
    assert tl.segments[0][1] < tl.segments[1][0]
    assert tl.label_at(tl.segments[1][0]) == "B"


def test_starts_and_ends_near_rest():
    tl = compose([(motion(0.0), "A")])
    assert np.allclose(tl.pose(tl.frames - 1)["Right"], REST["Right"], atol=1e-4)
    assert not np.allclose(tl.pose(tl.segments[0][0] + 10)["Right"], REST["Right"], atol=0.1)


def test_gap_adds_hold_frames():
    base = compose([(motion(0.0), "A"), (motion(0.5), "B")], gap_s=0.0)
    gapped = compose([(motion(0.0), "A"), (motion(0.5), "B")], gap_s=0.5)
    assert gapped.frames - base.frames == 2 * int(round(0.5 * FPS))


def test_speed_shortens_the_animation():
    slow = compose([(motion(0.0, frames=60), "A")], speed=1.0)
    fast = compose([(motion(0.0, frames=60), "A")], speed=2.0)
    assert fast.frames < slow.frames


def test_transitions_are_continuous():
    tl = compose([(motion(0.0), "A"), (motion(0.8), "B")])
    steps = np.abs(np.diff(tl.hands["Right"], axis=0)).max(axis=(1, 2))
    assert steps.max() < 1.0


def test_compose_rejects_bad_input():
    with pytest.raises(MotionError):
        compose([])
    with pytest.raises(MotionError):
        compose([(motion(0.0), "A")], speed=0)


def test_motion_save_load_roundtrip(tmp_path):
    m = motion(0.2)
    m.hands["Right"][:3] = np.nan
    m.save(tmp_path / "m.npz")
    back = Motion.load(tmp_path / "m.npz")
    assert back.frames == m.frames and back.fps == FPS
    assert back.quality == pytest.approx(m.quality)
    assert back.trimmed().frames == m.frames - 3


def test_empty_motion_cannot_be_trimmed():
    with pytest.raises(MotionError):
        Motion({"Right": np.full((5, 21, 2), np.nan, np.float32)}).trimmed()


def playlist_for(tmp_path, n=2) -> Playlist:
    items = []
    for i in range(n):
        path = tmp_path / f"{i}.npz"
        motion(0.3 * i).save(path)
        clip = Clip(f"G{i}", str(path), f"c{i}")
        items.append(PlaylistItem(f"G{i}", "sign", [clip]))
    return Playlist(items)


def test_build_timeline_uses_playlist_order(tmp_path):
    tl = build_timeline(playlist_for(tmp_path), gap_s=0.1)
    assert [s[2] for s in tl.segments] == ["G0", "G1"]


def test_build_timeline_empty_playlist_errors():
    with pytest.raises(ExportError):
        build_timeline(Playlist())


def test_export_writes_mp4(tmp_path):
    out = export_playlist(playlist_for(tmp_path), tmp_path / "o.mp4", size=96)
    assert out.is_file() and out.stat().st_size > 0
