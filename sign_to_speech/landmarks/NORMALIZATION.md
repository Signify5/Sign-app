# Landmark feature normalization (`sts_landmarks_v1`)

This note is the contract between Python training and the browser recognizer.
Reimplement it identically in JavaScript. Check both sides against
`golden/input.npz` -> `golden/expected.npy`.

## Version

- `NORM_VERSION`: `sts_landmarks_v1`
- `SEQ_LEN`: 64
- `FEAT_DIM`: 92
- Hand order: `Left`, then `Right` (anatomical labels from MediaPipe handedness
  + pose-wrist assignment, same as `gloss_to_asl.animate.assign_hands`)

## Upstream MediaPipe (must match)

Same Tasks models as `gloss_to_asl/animate.py`:

- Hand Landmarker float16/1 (`models/hand_landmarker.task`)
- Pose Landmarker lite float16/1 (`models/pose_landmarker_lite.task`)
- 21 hand landmarks per hand; pose shoulders 11/12; pose wrists 15 (Left), 16 (Right)
- Sequences resampled to 30 fps before this normalizer (existing `clips/motion`
  cache already is 30 fps)

## Input space: shoulder frame (`shoulder_v1`)

Before `normalize_sequence`, convert image-pixel landmarks exactly as
`gloss_to_asl.animate.extract_motion` does:

1. Sample body frames; take MediaPipe pose landmarks 11 (left shoulder) and 12
   (right shoulder) in pixel xy.
2. `origin = median( (left + right) / 2 )` over samples with
   `||left - right|| >= 20` px.
3. `unit = median( ||left - right|| )` (shoulder width in pixels).
4. Every hand landmark: `(xy_px - origin) / unit`.
5. Pose wrists (15/16) use the same origin/unit.
6. Missing detections are NaN (not zeros) at this stage.

`normalize_sequence` does **not** recompute shoulders; it expects this frame.

Offline training reuses `clips/motion/{gloss}/{id}.npz` (keys `Left`, `Right`,
optional `*_world`) and `clips/pose/{gloss}/{clip_id}.npy` shaped `(T, 2, 2)`
(Left/Right pose wrists already in shoulder frame).

## Per-frame packing

For each hand (`Left`, then `Right`), emit 43 floats:

| Offset (within hand block) | Content |
| --- | --- |
| 0 | presence: `1.0` if wrist (landmark 0) is finite, else `0.0` |
| 1..2 | wrist xy in shoulder frame (zeros if absent) |
| 3..42 | landmarks 1..20 as `xy[i] - wrist` (40 floats; zeros if absent) |

If the wrist is missing, or any of landmarks 0..20 used above are NaN, the whole
hand block is zeros and presence is 0.

Then pose wrists (6 floats):

| Offset | Content |
| --- | --- |
| 86 | Left pose wrist presence |
| 87..88 | Left pose wrist xy (shoulder frame) |
| 89 | Right pose wrist presence |
| 90..91 | Right pose wrist xy |

World landmarks are **not** part of `sts_landmarks_v1` (kept in the raw cache
only). Handedness is encoded by the Left/Right channel order; mirror
augmentation is a training concern, not this normalizer.

## Clip windowing

1. Let `T = min(len(Left), len(Right))`.
2. Build a boolean mask: frame t is live if either hand wrist is finite.
3. If no live frames: fail `no_hand_detected`.
4. If fewer than `MIN_HAND_FRAMES` (3) live frames: fail `too_short`.
5. Trim to the inclusive span from the first live frame to the last live frame.
6. Pack each kept frame into a `(FEAT_DIM,)` vector as above (NaN already replaced
   by zeros + presence flags).
7. Linearly resample the `(T_trim, FEAT_DIM)` array to `SEQ_LEN` along time
   (same lerp as `animate._resample`: `pos = linspace(0, T_trim-1, SEQ_LEN)`).
8. Binarize presence channels (offsets 0, 43, 86, 89): value `>= 0.5` -> `1`,
   else `0`. When a presence becomes 0, zero that hand's 43-block or that pose
   wrist's 3-block so interpolated ghosts do not leak.

## Output

`float32` array shaped `(64, 92)`, finite, no NaNs.

## Golden vector

```
sign_to_speech/landmarks/golden/input.npz
  Left:  (T, 21, 2)
  Right: (T, 21, 2)
  pose_wrists: (T, 2, 2)
sign_to_speech/landmarks/golden/expected.npy  # (64, 92)
sign_to_speech/landmarks/golden/meta.json     # NORM_VERSION, T, etc.
```

Python: `venv/Scripts/python.exe -m sign_to_speech.landmarks.test_golden`
JS (later): load the same files and assert `max_abs_diff <= 1e-5`.
