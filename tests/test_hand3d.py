import numpy as np

from hand3d import HANDS, Pose, Rig, active_span, compose3d, glide_frames, min_jerk, natural_curl, q_axis, rest_pose, smooth_track


def _pose(x, angle):
    q = np.tile([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)], (20, 1))
    return Pose(np.array([x, 0.0, 0.0]), q)


def test_min_jerk_endpoints_and_symmetry():
    assert min_jerk(0.0) == 0.0 and min_jerk(1.0) == 1.0
    assert abs(min_jerk(0.5) - 0.5) < 1e-12
    h = 1e-4  # flat start and finish
    assert min_jerk(h) / h < 1e-6 and (1 - min_jerk(1 - h)) / h < 1e-6


def test_smooth_track_removes_jitter_and_keeps_length():
    rng = np.random.default_rng(0)
    poses = [_pose(0.1 + 0.005 * rng.standard_normal(), 0.3 + 0.05 * rng.standard_normal()) for _ in range(60)]
    out = smooth_track(poses)
    assert len(out) == len(poses)
    raw = np.std([p.pos[0] for p in poses])
    assert np.std([p.pos[0] for p in out]) < raw * 0.7
    assert np.allclose(np.linalg.norm(out[10].quats, axis=1), 1.0)


def test_smooth_track_handles_quaternion_sign_flips():
    poses = [_pose(0.0, 0.4) for _ in range(10)]
    for p in poses[::2]:
        p.quats = -p.quats  # same rotation, opposite sign
    out = smooth_track(poses)
    ref = _pose(0.0, 0.4).quats[0]
    assert all(abs(abs(p.quats[0] @ ref) - 1) < 1e-9 for p in out)


def test_glide_frames_scale_with_distance():
    a = _pose(0.0, 0.0)
    near, far = glide_frames(a, _pose(0.02, 0.0), 10), glide_frames(a, _pose(0.5, 0.0), 10)
    assert 2 <= near < far <= 16


def test_active_span_skips_lift_in_and_out():
    up = [Pose(np.array([0.0, y, 0.0]), np.tile([0.0, 0, 0, 1], (20, 1))) for y in (-0.4, -0.3, -0.1, 0.0, -0.2, -0.35)]
    assert active_span({"Left": None, "Right": up}) == (2, 4)


def test_compose3d_glides_from_rest_and_back():
    rig = Rig.load()
    rest = rest_pose(rig)
    sign = [_pose(0.1 * i, 0.2) for i in range(5)]
    tl = compose3d(rig, [({"Left": None, "Right": sign}, "ONE"), (None, "SKIPPED"), ({"Left": sign, "Right": sign}, "TWO")])
    assert [s[2] for s in tl.segments] == ["ONE", "TWO"]
    right = HANDS.index("Right")
    assert np.allclose(tl.frames[-1, right, :3], rest.pos)
    s, e, _ = tl.segments[0]
    assert np.allclose(tl.frames[e, right, :3], sign[-1].pos)
    assert np.allclose(tl.frames[s:e + 1, HANDS.index("Left"), :3], rest.pos)  # the unused hand rests


def test_natural_curl_moves_bend_to_the_knuckle_and_keeps_the_total():
    rig = Rig.load()
    x = rig.rest_frame[:, 0]
    q = np.tile([0.0, 0, 0, 1], (len(rig.names), 1))
    bones = [b for b, _ in rig.chains["index"]][1:]
    for b, deg in zip(bones, (50, 75, 80)):  # a tracked fist: shallow knuckle, folded tip
        q[b] = q_axis(x, np.radians(deg))
    flex = lambda q: [np.degrees(2 * np.arctan2(q[b, :3] @ x, q[b, 3])) for b in bones]
    out = flex(natural_curl(rig, q))
    assert np.isclose(sum(out), 205) and out[0] > 85 and out[2] < 30
    q2 = np.tile([0.0, 0, 0, 1], (len(rig.names), 1))
    assert np.allclose(natural_curl(rig, q2), q2)  # an open hand is untouched
