import numpy as np

import signcheck as sc
from animate import assign_hands
from motion import Motion

T = 40


def hand(x, y0, y1, straight=True):
    """A hand whose wrist moves from (x, y0) to (x, y1); fingers straight or curled."""
    xy = np.zeros((T, 21, 2), np.float32)
    xy[:, :, 0] = x
    xy[:, :, 1] = np.linspace(y0, y1, T)[:, None]
    xy[:, sc.TIPS, 1] -= 0.2
    world = np.zeros((T, 21, 3), np.float32)
    for m, tip in zip(sc.MCPS, sc.TIPS):
        for k, j in enumerate(range(m, tip + 1)):
            world[:, j, 1] = -0.02 * k if straight else -0.02 * (k % 2)
    return xy, world


def motion(right, left=None):
    hands, world = {}, {}
    for name, h in (("Right", right), ("Left", left)):
        if h is None:
            hands[name] = np.full((T, 21, 2), np.nan, np.float32)
            world[name] = np.full((T, 21, 3), np.nan, np.float32)
        else:
            hands[name], world[name] = h
    return Motion(hands, 30.0, world)


ONE = {"EntryID": "x", "SignType.2.0": "OneHanded", "MajorLocation.2.0": "Head", "MinorLocation.2.0": "Chin",
       "SelectedFingers.2.0": "imrp", "Flexion.2.0": "FullyOpen", "FlexionChange.2.0": "0"}


def test_one_handed_head_sign_passes():
    c = sc.check("X", motion(hand(-0.3, -0.2, -0.5)), [ONE])
    assert c.issues == [] and c.reference == "x"


def test_two_hands_where_asl_uses_one():
    c = sc.check("X", motion(hand(-0.3, -0.2, -0.9), hand(0.3, 0.6, 0.0)), [ONE])
    assert any("one hand" in i for i in c.issues)


def test_sign_below_the_face_where_asl_signs_at_head():
    c = sc.check("X", motion(hand(-0.3, 0.6, 0.4)), [ONE])
    assert any("head" in i for i in c.issues)


def test_closed_hand_where_asl_opens_it():
    c = sc.check("X", motion(hand(-0.3, -0.2, -0.5, straight=False)), [ONE])
    assert any("handshape" in i for i in c.issues)


def test_swapped_hands_are_caught():
    xy, w = hand(-0.3, 0.2, 0.2)
    xy[20:, :, 0] = 0.5  # the tracker hands the right hand's label to the left hand
    c = sc.check("X", motion((xy, w)))
    assert any("jumps" in i for i in c.issues)


def test_odd_signing_out():
    a = np.zeros((24, 2))
    paths = [a, a + 0.05, a + [1.0, 0.0]]
    assert sc.odd_ones(paths) == [False, False, True]
    assert sc.odd_ones(paths[:2]) == [False, False]


def test_assign_hands_by_arm():
    a = np.zeros((21, 5), np.float32)
    b = np.zeros((21, 5), np.float32)
    a[0, :2], b[0, :2] = (100, 200), (400, 200)
    arm = {"Right": np.array([110.0, 210.0]), "Left": np.array([390.0, 190.0])}
    # the tracker calls both hands "Left"; the arms decide
    out = assign_hands([(a, "Left"), (b, "Left")], arm, {}, 640)
    assert out["Right"] is a and out["Left"] is b
    # one hand, no pose: it stays the hand it was a moment ago
    out = assign_hands([(b, "Right")], {}, {"Left": np.array([395.0, 200.0])}, 640)
    assert list(out) == ["Left"]


def test_lost_hand_is_serious_but_a_one_handed_variant_is_not():
    TWO = dict(ONE, **{"SignType.2.0": "SymmetricalOrAlternating", "MajorLocation.2.0": "Neutral",
                       "MinorLocation.2.0": "Neutral"})
    m = motion(hand(-0.3, 0.2, 0.0))
    arms = np.full((T, 2, 2), 1.6, np.float32)  # both arms down (Left, Right)
    arms[5:35, 1, 1] = 0.3                        # the right arm is up and its hand is tracked
    c = sc.check("X", m, [TWO], arms=arms)
    assert c.bad == 0 and any("two hands" in i for i in c.issues)  # signed one-handed: a variant
    arms[5:35, 0, 1] = 0.3  # the left arm is up too, but the tracker never found its hand
    c = sc.check("X", m, [TWO], arms=arms)
    assert c.bad == 1 and any("loses the signer's left hand" in i for i in c.issues)


def test_a_lost_hand_is_filled_along_its_arm():
    from animate import fill_lost_hands
    m = motion(hand(-0.3, 0.2, 0.2), hand(0.3, 0.2, 0.2))
    hands, world = m.hands, m.world
    hands["Left"][10:30] = np.nan  # lost while touching the other hand
    world["Left"][10:30] = np.nan
    arms = np.full((T, 2, 2), 1.6, np.float32)  # arms rest outside frames 5 to 34
    arms[5:35, 0] = (0.3, 0.2)  # left arm up from frame 5 to 34, at the hand's wrist
    arms[5:35, 1] = (-0.3, 0.2)
    arms[20, 0] = (0.4, 0.2)   # the arm moves right at frame 20
    assert fill_lost_hands(hands, world, arms) == 20
    assert np.allclose(hands["Left"][20, 0], (0.4, 0.2)) and np.allclose(hands["Left"][15, 0], (0.3, 0.2))
    assert not np.isnan(world["Left"][10:30]).any()
