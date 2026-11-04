"""Golden test: landmark input -> expected features for sts_landmarks_v1."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from sign_to_speech.landmarks.normalize import (
    FEAT_DIM,
    NORM_VERSION,
    SEQ_LEN,
    NormalizeError,
    normalize_sequence,
)

GOLDEN = Path(__file__).resolve().parent / "golden"


def test_golden_vector() -> None:
    meta = json.loads((GOLDEN / "meta.json").read_text(encoding="utf-8"))
    assert meta["norm_version"] == NORM_VERSION
    with np.load(GOLDEN / "input.npz") as data:
        left = data["Left"].astype(np.float32)
        right = data["Right"].astype(np.float32)
        pose = data["pose_wrists"].astype(np.float32)
    expected = np.load(GOLDEN / "expected.npy").astype(np.float32)
    got = normalize_sequence({"Left": left, "Right": right}, pose_wrists=pose)
    assert got.shape == (SEQ_LEN, FEAT_DIM)
    assert expected.shape == (SEQ_LEN, FEAT_DIM)
    atol = float(meta.get("atol", 1e-5))
    max_abs = float(np.max(np.abs(got - expected)))
    assert max_abs <= atol, f"max_abs_diff={max_abs} > atol={atol}"


def test_no_hand_detected() -> None:
    T = 8
    left = np.full((T, 21, 2), np.nan, np.float32)
    right = np.full((T, 21, 2), np.nan, np.float32)
    try:
        normalize_sequence({"Left": left, "Right": right})
        raise AssertionError("expected NormalizeError")
    except NormalizeError as exc:
        assert str(exc) == "no_hand_detected"


def test_too_short() -> None:
    T = 8
    left = np.full((T, 21, 2), np.nan, np.float32)
    right = np.full((T, 21, 2), np.nan, np.float32)
    # Only two live frames (< MIN_HAND_FRAMES=3).
    for t in (2, 5):
        right[t, :, :] = 0.0
        right[t, 0] = [-0.3, 0.9]
    try:
        normalize_sequence({"Left": left, "Right": right})
        raise AssertionError("expected NormalizeError")
    except NormalizeError as exc:
        assert str(exc) == "too_short"


def test_missing_hand_zeroed() -> None:
    T = 10
    left = np.full((T, 21, 2), np.nan, np.float32)
    right = np.zeros((T, 21, 2), np.float32)
    for t in range(T):
        right[t, 0] = [-0.4, 0.8]
        right[t, 1:] = right[t, 0] + 0.01
    feats = normalize_sequence({"Left": left, "Right": right})
    # Left presence channel must stay 0 for the whole window.
    assert np.all(feats[:, 0] == 0.0)
    assert np.all(feats[:, 1:43] == 0.0)
    # Right should be present on most resampled frames.
    assert float(feats[:, 43].mean()) > 0.9


if __name__ == "__main__":
    test_golden_vector()
    test_no_hand_detected()
    test_too_short()
    test_missing_hand_zeroed()
    print("all golden checks passed")
    sys.exit(0)
