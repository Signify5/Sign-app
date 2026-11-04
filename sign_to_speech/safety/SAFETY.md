# Sign -> Speech safety lexicon and decide contract

This note is the contract between Python (`decide.py`) and the browser
recognizer. Reimplement the decision logic identically in JavaScript. Check both
sides against `golden/decision_vectors.json`.

ASCII punctuation only.

## Status

- Section 4 training has **not** run. There are no held-out per-class
  accuracies yet.
- `enable_list.json` is an explicit **placeholder** with
  `status: "not_generated"` and an empty `classes` list. Do not invent an
  enable list. Generate it later with `build_enable_list.py`.
- `risky_pairs.json` is a **draft** (`status: "draft"`,
  `review_required: true`). A human must review it before it is treated as
  final.
- Abstain-rate vs accuracy on the held-out set is **skipped** until a trained
  model and held-out predictions exist.

## Files

| Path | Role |
| --- | --- |
| `enable_list.json` | UI-allowed class subset (placeholder until generated) |
| `risky_pairs.json` | Curated risky confusions + number group |
| `decide.py` | Emit / abstain decision |
| `build_enable_list.py` | Build enable list from per-class accuracies |
| `golden/decision_vectors.json` | Golden test vectors for Python and JS |
| `test_decide.py` | Python tests including golden vectors |

## Enable list (not yet generated)

### Placeholder

Current `enable_list.json`:

- `status`: `"not_generated"`
- `classes`: `[]`
- `threshold`: `null`
- `note` explains that Section 4 has not produced accuracies

Until generated, the UI must not treat any lexical class as enabled from this
file.

### Input format for `build_enable_list.py`

Per-class held-out accuracy JSON (version 1):

```json
{
  "version": 1,
  "split": "test",
  "metric": "top1_accuracy",
  "classes": [
    {"gloss": "water", "accuracy": 0.97, "n": 12, "correct": 11},
    {"gloss": "yes", "accuracy": 0.91, "n": 8, "correct": 7}
  ]
}
```

Rules:

- Top-level must be an object.
- `classes` must be a non-empty list.
- Each class requires `gloss` (non-empty string) and `accuracy` (number in
  `[0, 1]`).
- `n` and `correct` are optional non-negative ints.
- Duplicate `gloss` values are rejected.
- Top-level `version`, `split`, and `metric` are optional but recommended.
  Default `metric` when omitted: `top1_accuracy`.

### Generation rule

A class is **enabled** iff `accuracy >= threshold`.

Default threshold: `0.95` (M1 gate for the enabled set).

```text
venv/Scripts/python.exe -m sign_to_speech.safety.build_enable_list ^
  --accuracies path/to/per_class_accuracy.json ^
  --threshold 0.95 ^
  --out sign_to_speech/safety/enable_list.json
```

Output fields of note:

- `status`: `"generated"`
- `classes`: list of enabled gloss strings
- `excluded`: each excluded gloss with accuracy and reason string
- `threshold`, `source_accuracy_file`, `generated_at`

The enable list filters what the UI may emit. It does not replace the decide
function. Even an enabled class can abstain under `decide`.

## Risky pairs

### Schema

`risky_pairs.json` version 1:

- `pairs`: list of `{a, b, severity, category, note}`
  - Undirected: `(a, b)` means the same as `(b, a)`.
  - Only glosses that exist in
    `sign_to_speech/dataset/lexical_manifest.json` are included.
- `groups`: list of `{id, severity, category, note, members}`
  - Any two **distinct** members of the same group form a risky confusion.
  - Used for the numbers/quantity cluster so the file stays curated.
- `default_margins`: per-severity margin thresholds (see below).
- `missing_wanted`: risky concepts desired for the lexicon but **not** present
  as classes in the inventory (reported for review, not used at runtime).
- `status`: `"draft"` until human review signs off.

### Severity levels

| Severity | Default margin | Typical use |
| --- | --- | --- |
| `critical` | `0.25` | Negation, yes/no, life/death, person flips |
| `high` | `0.20` | Numbers group, time/deixis, gender/kinship, outcomes |
| `medium` | `0.15` | Milder opposites (size, speed, social greeting) |

## Decision logic (identical Python / JS)

### Defaults

- `confidence_threshold` = `0.55`
- `margin_threshold` = `null` (use per-severity margins from the lexicon)
- Fallback margin if severity unknown = `0.15`

When calling `decide`, you may pass an explicit `margin_threshold` to override
**all** severity margins with one global value. If `margin_threshold` is
`null`/`undefined`, use `lexicon.margin_for(severity)`.

### Input

`topk`: list of `{label, score}` (or `[label, score]`).

- `label` / `gloss`: string class name (inventory spelling, usually lowercase).
- `score`: number (calibrated probability or comparable confidence).

### Algorithm

Pseudocode (normative):

```
function decide(topk, lexicon, confidence_threshold=0.55, margin_threshold=null):
  scored = normalize_to_label_score_pairs(topk)
  # Stable sort by score descending (equal scores keep input order).
  scored.sort_stable_by_score_desc()

  if scored is empty:
    return abstain(reason="empty", label=null, guess=null, confidence=0,
                   margin=null)

  top1 = scored[0]
  top2 = scored[1] if len(scored) > 1 else null
  margin = (top1.score - top2.score) if top2 else null

  # 1) Global confidence gate (checked before risky margin).
  if top1.score < confidence_threshold:
    return abstain(reason="low_confidence", label=null, guess=top1.label,
                   confidence=top1.score, margin=margin)

  # 2) Risky confusion gate.
  if top2 is not null:
    hit = lexicon.lookup(top1.label, top2.label)
    if hit is not null:
      required = margin_threshold if margin_threshold is not null
                 else lexicon.margin_for(hit.severity)
      if margin < required:
        ordered = sort_lexicographic([top1.label, top2.label])
        return abstain(reason="risky_confusion", label=null,
                       guess=top1.label, confidence=top1.score,
                       margin=margin, severity=hit.severity,
                       risky_pair=ordered, category=hit.category)

  # 3) Emit.
  return emit(reason="ok", label=top1.label, guess=top1.label,
              confidence=top1.score, margin=margin)
```

### `lexicon.lookup(a, b)`

1. If `a == b`: not risky.
2. If frozenset `{a, b}` matches an explicit pair: return that hit
   (`source="pair"`). First pair declaration wins if duplicates exist.
3. Else if some group contains both `a` and `b`: return that hit
   (`source="group"`, include `group_id`). First matching group wins.
4. Else: not risky.

### Output fields

| Field | Meaning |
| --- | --- |
| `action` | `"emit"` or `"abstain"` |
| `label` | Emitted gloss if `action=="emit"`, else `null` |
| `guess` | Top-1 label for UI copy when abstaining; `null` if empty |
| `confidence` | Top-1 score (0 if empty) |
| `margin` | `top1.score - top2.score`, or `null` if fewer than 2 scores |
| `reason` | `"ok"` \| `"empty"` \| `"low_confidence"` \| `"risky_confusion"` |
| `severity` | Set on risky abstain; else `null` |
| `risky_pair` | Lexicographically sorted `[a, b]` on risky abstain; else `null` |
| `category` | Pair/group category on risky abstain; else `null` |

### Invariants

- Never set `action="emit"` when the top-2 form a risky pair and
  `margin < required`. That is the "never emit the dangerous swap" rule.
- On abstain, `label` is always `null`. UI may show
  `Not sure - guess: {guess}` using `guess`, which is informational only.
- Enable-list filtering is a separate UI gate: even `emit` should be dropped
  (or forced to abstain) if `label` is not in the generated enable list. That
  filter is not inside `decide` itself.

### Floating point

Compare margins and thresholds with ordinary IEEE-754 floats. Golden vectors
use simple decimals; do not add epsilon unless both Python and JS agree in a
future revision of this doc.

## Golden vectors

File: `golden/decision_vectors.json`

Each case has:

- `id`: string
- `topk`: `[[label, score], ...]`
- `confidence_threshold`: number
- `margin_threshold`: number or `null`
- `expected`: decision object as above

Python: `test_decide.py` loads the lexicon from `risky_pairs.json` and asserts
each vector. JS must do the same after porting `decide`.

## Evaluation deferred

Abstain-rate vs accuracy on the held-out signer split requires a trained
model and per-example top-k predictions. **Not run in Section 5.** Revisit in
Section 9 (eval / M1 gate report).

## How to verify (Python)

```text
venv/Scripts/python.exe -m sign_to_speech.safety.test_decide
venv/Scripts/python.exe -m pytest sign_to_speech/safety/test_decide.py -q
```
