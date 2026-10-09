import numpy as np
import pytest

from fingerspell import ALPHABET, DOUBLE_SHIFT_M, _Kin, letter_frames, letter_pose, spell_track
from hand3d import Rig


@pytest.fixture(scope="module")
def rig():
    return Rig.load()


def test_every_letter_has_a_valid_pose(rig):
    assert sorted(ALPHABET) == [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    for c in ALPHABET:
        p = letter_pose(rig, c)
        assert p.quats.shape == (len(rig.names), 4)
        assert np.allclose(np.linalg.norm(p.quats, axis=1), 1.0)


@pytest.mark.parametrize("letter", ["F", "O", "D", "U"])
def test_thumb_reaches_the_finger_it_touches(rig, letter):
    kind, finger = ALPHABET[letter].thumb.split(":")
    kin, q = _Kin(rig), letter_pose(rig, letter).quats
    pts = kin.chain_points(finger, q)
    pad = (pts[-1] if kind == "tip" else (pts[2] + pts[3]) / 2) + kin.F @ np.array([0.0, 0.0, 0.006])
    assert np.linalg.norm(kin.chain_points("thumb", q)[-1] - pad) < 0.01  # within 1 cm


def test_j_and_z_move_and_static_letters_hold(rig):
    for c in "JZ":
        f = letter_frames(rig, c, 10)
        assert np.linalg.norm(f[-1].pos - f[0].pos) > 0.03
    a = letter_frames(rig, "A", 4)
    assert all(np.allclose(p.pos, a[0].pos) for p in a)


def test_spell_track_uses_the_right_hand_and_skips_non_letters(rig):
    tracks = spell_track(rig, "a-1b")
    assert tracks["Left"] is None
    ab = spell_track(rig, "AB")["Right"]
    assert len(tracks["Right"]) == len(ab)
    assert spell_track(rig, "123") is None


def test_doubled_letter_slides_sideways(rig):
    poses = spell_track(rig, "LL")["Right"]
    assert np.isclose(poses[-1].pos[0] - poses[0].pos[0], -DOUBLE_SHIFT_M)
