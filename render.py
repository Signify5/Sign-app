from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

SIZE = 480
SUPERSAMPLE = 2
# Visible window in body units (origin mid-shoulder, 1.0 = shoulder width, +y down).
VIEW_X = (-1.1, 1.1)
VIEW_Y = (-1.0, 1.4)
HAND_SCALE = 1.5  # cosmetic: hands are drawn larger than life so finger shapes stay readable

BACKGROUND = (38, 34, 32)  # BGR
GUIDE = (47, 43, 41)
SKIN = (150, 188, 232)
SKIN_SHADE = (104, 134, 178)
OUTLINE = (62, 84, 124)

FINGERS = ((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
PALM = (0, 1, 5, 9, 13, 17)


def _to_px(pts: np.ndarray, big: int) -> np.ndarray:
    span = np.array([VIEW_X[1] - VIEW_X[0], VIEW_Y[1] - VIEW_Y[0]], np.float32)
    origin = np.array([VIEW_X[0], VIEW_Y[0]], np.float32)
    return (pts - origin) / span * big


def _draw_guide(canvas: np.ndarray, big: int) -> None:
    """Faint head-and-shoulders silhouette so the viewer can see where a sign happens."""
    def px(points):
        return _to_px(np.array(points, np.float32), big).astype(np.int32)

    torso = px([(-0.15, -0.35), (-0.5, 0.0), (-0.58, 1.5), (0.58, 1.5), (0.5, 0.0), (0.15, -0.35)])
    cv2.fillConvexPoly(canvas, torso, GUIDE, cv2.LINE_AA)
    cx, cy = _to_px(np.array([0.0, -0.59], np.float32), big)
    axes = (int(0.2 / (VIEW_X[1] - VIEW_X[0]) * big), int(0.29 / (VIEW_Y[1] - VIEW_Y[0]) * big))
    cv2.ellipse(canvas, (int(cx), int(cy)), axes, 0, 0, 360, GUIDE, -1, cv2.LINE_AA)


def _smooth(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _draw_hand(canvas: np.ndarray, p: np.ndarray) -> None:
    """Shaded hand: rounded volume from a distance-to-edge ramp, joint creases, soft cast shadow, fading forearm."""
    palm_w = max(float(np.linalg.norm(p[5] - p[17])), 10.0)
    centre = p[list(PALM)].mean(axis=0)
    wrist_dir = p[0] - centre
    norm = float(np.linalg.norm(wrist_dir))
    wrist_dir = wrist_dir / norm if norm else np.array([0.0, 1.0], np.float32)

    margin = int(palm_w * 3.2)
    lo = np.floor(p.min(axis=0) - margin).astype(int)
    hi = np.ceil(p.max(axis=0) + margin).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, [canvas.shape[1], canvas.shape[0]])
    if hi[0] - lo[0] < 4 or hi[1] - lo[1] < 4:
        return
    q = (p - lo).astype(np.float32)
    h, w = int(hi[1] - lo[1]), int(hi[0] - lo[0])

    def capsule(img, a, b, ra, rb, value=255):
        steps = max(int(np.linalg.norm(b - a) / max(min(ra, rb) * 0.5, 1)), 1) + 1
        for t in np.linspace(0, 1, steps):
            c = (1 - t) * a + t * b
            cv2.circle(img, (int(c[0]), int(c[1])), int(round((1 - t) * ra + t * rb)), value, -1, cv2.LINE_AA)

    hand = np.zeros((h, w), np.uint8)
    capsule(hand, q[0], q[0] + wrist_dir * palm_w * 0.5, palm_w * 0.40, palm_w * 0.38)
    poly = q[list(PALM)].astype(np.int32)
    cv2.fillConvexPoly(hand, poly, 255, cv2.LINE_AA)
    cv2.polylines(hand, [poly], True, 255, int(palm_w * 0.28), cv2.LINE_AA)
    for chain in FINGERS:
        base = palm_w * (0.15 if chain[0] != 1 else 0.19)
        for j in range(3):
            capsule(hand, q[chain[j]], q[chain[j + 1]], base * (1 - 0.10 * j), base * (1 - 0.10 * (j + 1)))

    forearm = np.zeros((h, w), np.uint8)  # fades out so the arm does not end in a hard stump
    start = q[0] + wrist_dir * palm_w * 0.4
    for t in np.linspace(1, 0, 48):
        c = start + wrist_dir * palm_w * 1.8 * t
        value = int(255 * (1 - t) ** 0.8)
        cv2.circle(forearm, (int(c[0]), int(c[1])), int(palm_w * (0.38 + 0.04 * t)), value, -1, cv2.LINE_AA)
    alpha = np.maximum(hand, forearm).astype(np.float32) / 255.0

    # Cast shadow on the background.
    shadow = cv2.GaussianBlur(alpha, (0, 0), palm_w * 0.12)
    dx, dy = int(palm_w * 0.10), int(palm_w * 0.14)
    shifted = np.zeros_like(shadow)
    shifted[dy:, dx:] = shadow[: h - dy, : w - dx]
    roi = canvas[lo[1]:hi[1], lo[0]:hi[0]].astype(np.float32)
    roi *= (1.0 - 0.45 * shifted)[..., None]

    # Volume: skin brightens from the silhouette edge inwards; plateau is exactly SKIN.
    inside = np.maximum(hand, (forearm > 127).astype(np.uint8) * 255) > 127
    dist = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 3)
    ramp = _smooth(dist / (palm_w * 0.17))[..., None]
    skin = np.array(SKIN, np.float32)
    edge = np.array(OUTLINE, np.float32) * 0.5 + skin * 0.5
    colour = edge + (skin - edge) * ramp

    # Finger joint creases and a subtle palm crease, drawn darker, only where there is skin.
    creases = np.zeros((h, w), np.float32)
    for chain in FINGERS:
        for j in (1, 2):
            a, b = q[chain[j]], q[chain[j + 1]]
            d = b - a
            n = float(np.linalg.norm(d))
            if n < 1e-3:
                continue
            perp = np.array([-d[1], d[0]], np.float32) / n
            r = palm_w * 0.11 * (1 - 0.1 * j)
            m = a + d * 0.0
            cv2.line(creases, tuple((m - perp * r).astype(int)), tuple((m + perp * r).astype(int)), 1.0, 1, cv2.LINE_AA)
    cv2.line(creases, tuple(q[5].astype(int)), tuple(q[17].astype(int)), 0.6, 1, cv2.LINE_AA)
    colour *= (1.0 - 0.22 * creases)[..., None]

    a3 = alpha[..., None]
    roi = roi * (1 - a3) + colour * a3
    canvas[lo[1]:hi[1], lo[0]:hi[0]] = np.clip(roi, 0, 255).astype(np.uint8)


def render_frame(pose: Dict[str, np.ndarray], size: int = SIZE, guide: bool = True, mirror: bool = False) -> np.ndarray:
    """One RGB frame (size x size, uint8) of the digital hands for a {hand: 21x2} pose."""
    big = size * SUPERSAMPLE
    canvas = np.empty((big, big, 3), np.uint8)
    canvas[:] = BACKGROUND
    if guide:
        _draw_guide(canvas, big)
    hands = []
    for pts in pose.values():
        pts = np.asarray(pts, np.float32)
        centre = pts[list(PALM)].mean(axis=0)
        pts = centre + (pts - centre) * HAND_SCALE
        if mirror:
            pts = pts * np.array([-1, 1], np.float32)
        hands.append(_to_px(pts, big))
    for px in sorted(hands, key=lambda a: -a[:, 1].mean()):
        _draw_hand(canvas, px)
    frame = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
