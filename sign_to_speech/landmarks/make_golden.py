"""Build the golden landmark -> feature vector (deterministic, no video I/O)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sign_to_speech.landmarks.normalize import (
    FEAT_DIM,
    NORM_VERSION,
    SEQ_LEN,
    feature_layout,
    normalize_sequence,
)

OUT = Path(__file__).resolve().parent / "golden"


def build_input(T: int = 16) -> dict:
    """Synthetic clip: right hand always present; left appears mid-clip; one gap."""
    left = np.full((T, 21, 2), np.nan, np.float32)
    right = np.full((T, 21, 2), np.nan, np.float32)
    pose = np.full((T, 2, 2), np.nan, np.float32)

    rng = np.random.default_rng(0)
    for t in range(T):
        # Right hand every frame except t==3 (tests missing-hand masking).
        if t != 3:
            wrist = np.array([-0.4 + 0.01 * t, 0.8 - 0.005 * t], np.float32)
            shape = rng.normal(0.0, 0.02, size=(20, 2)).astype(np.float32)
            # Thumb-side bias so the vector is not pure noise.
            shape[:, 0] += 0.05
            right[t, 0] = wrist
            right[t, 1:] = wrist + shape
            pose[t, 1] = wrist + np.array([0.02, 0.03], np.float32)

        # Left hand only on frames 6..12.
        if 6 <= t <= 12:
            wrist = np.array([0.35 - 0.008 * (t - 6), 0.85], np.float32)
            shape = rng.normal(0.0, 0.015, size=(20, 2)).astype(np.float32)
            shape[:, 0] -= 0.04
            left[t, 0] = wrist
            left[t, 1:] = wrist + shape
            pose[t, 0] = wrist + np.array([-0.02, 0.03], np.float32)

    return {"Left": left, "Right": right, "pose_wrists": pose}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = build_input(16)
    expected = normalize_sequence(
        {"Left": data["Left"], "Right": data["Right"]},
        pose_wrists=data["pose_wrists"],
    )
    assert expected.shape == (SEQ_LEN, FEAT_DIM)
    assert np.isfinite(expected).all()

    np.savez_compressed(
        OUT / "input.npz",
        Left=data["Left"],
        Right=data["Right"],
        pose_wrists=data["pose_wrists"],
    )
    np.save(OUT / "expected.npy", expected.astype(np.float32))
    meta = {
        "norm_version": NORM_VERSION,
        "seq_len": SEQ_LEN,
        "feat_dim": FEAT_DIM,
        "input_frames": 16,
        "layout": feature_layout(),
        "atol": 1e-5,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'input.npz'}, expected shape {expected.shape}")


if __name__ == "__main__":
    main()
