"""Build enable_list.json from held-out per-class accuracies.

Section 4 has not produced accuracies yet. When they exist, run:

  venv/Scripts/python.exe -m sign_to_speech.safety.build_enable_list \\
    --accuracies path/to/per_class_accuracy.json \\
    --threshold 0.95 \\
    --out sign_to_speech/safety/enable_list.json

See SAFETY.md for the exact input schema.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_THRESHOLD = 0.95
ENABLE_LIST_VERSION = 1


def _die(msg: str, code: int = 2) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def load_per_class_accuracies(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and validate the per-class held-out accuracy file.

    Expected JSON (version 1)::

        {
          "version": 1,
          "split": "test",
          "metric": "top1_accuracy",
          "classes": [
            {"gloss": "water", "accuracy": 0.97, "n": 12, "correct": 11},
            ...
          ]
        }

    Required per class: gloss (str), accuracy (float in [0, 1]).
    Optional per class: n (int), correct (int), and any other metrics.
    Top-level split/metric are optional but recommended.
    """
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        _die(f"cannot read {path}: {exc}")
    except json.JSONDecodeError as exc:
        _die(f"invalid JSON in {path}: {exc}")

    if not isinstance(doc, dict):
        _die(f"{path}: top-level value must be an object")
    classes = doc.get("classes")
    if not isinstance(classes, list) or not classes:
        _die(f"{path}: 'classes' must be a non-empty list")

    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, row in enumerate(classes):
        if not isinstance(row, dict):
            _die(f"{path}: classes[{i}] must be an object")
        gloss = row.get("gloss")
        acc = row.get("accuracy")
        if not isinstance(gloss, str) or not gloss:
            _die(f"{path}: classes[{i}].gloss must be a non-empty string")
        if gloss in seen:
            _die(f"{path}: duplicate gloss {gloss!r}")
        seen.add(gloss)
        if not isinstance(acc, (int, float)) or isinstance(acc, bool):
            _die(f"{path}: classes[{i}].accuracy must be a number")
        acc_f = float(acc)
        if acc_f < 0.0 or acc_f > 1.0:
            _die(f"{path}: classes[{i}].accuracy must be in [0, 1], got {acc_f}")
        entry: dict[str, Any] = {
            "gloss": gloss,
            "accuracy": acc_f,
        }
        if "n" in row:
            n = row["n"]
            if not isinstance(n, int) or isinstance(n, bool) or n < 0:
                _die(f"{path}: classes[{i}].n must be a non-negative int")
            entry["n"] = n
        if "correct" in row:
            correct = row["correct"]
            if not isinstance(correct, int) or isinstance(correct, bool) or correct < 0:
                _die(f"{path}: classes[{i}].correct must be a non-negative int")
            entry["correct"] = correct
        out.append(entry)

    meta = {
        "version": doc.get("version"),
        "split": doc.get("split"),
        "metric": doc.get("metric", "top1_accuracy"),
    }
    return out, meta


def build_enable_list(
    classes: list[dict[str, Any]],
    threshold: float,
    *,
    source_accuracy_file: str | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    enabled: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for row in sorted(classes, key=lambda r: r["gloss"]):
        item = {
            "gloss": row["gloss"],
            "accuracy": row["accuracy"],
        }
        if "n" in row:
            item["n"] = row["n"]
        if "correct" in row:
            item["correct"] = row["correct"]
        if row["accuracy"] >= threshold:
            enabled.append(item)
        else:
            item["reason"] = (
                f"held-out accuracy {row['accuracy']:.6f} < threshold {threshold:.6f}"
            )
            excluded.append(item)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "version": ENABLE_LIST_VERSION,
        "status": "generated",
        "review_required": True,
        "note": (
            "Generated from held-out per-class accuracies. "
            "Only listed classes may be emitted by the UI. "
            "Human review still required before treating as final."
        ),
        "threshold": threshold,
        "source_accuracy_file": source_accuracy_file,
        "source_meta": meta or {},
        "generated_at": now,
        "classes": [e["gloss"] for e in enabled],
        "enabled_detail": enabled,
        "excluded": excluded,
        "stats": {
            "num_enabled": len(enabled),
            "num_excluded": len(excluded),
            "num_input_classes": len(classes),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build enable_list.json from per-class held-out accuracies."
    )
    parser.add_argument(
        "--accuracies",
        type=Path,
        required=True,
        help="Path to per-class accuracy JSON (see SAFETY.md).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Minimum held-out top-1 accuracy to enable (default {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "enable_list.json",
        help="Output enable_list.json path.",
    )
    args = parser.parse_args(argv)

    classes, meta = load_per_class_accuracies(args.accuracies)
    try:
        doc = build_enable_list(
            classes,
            args.threshold,
            source_accuracy_file=str(args.accuracies).replace("\\", "/"),
            meta=meta,
        )
    except ValueError as exc:
        _die(str(exc))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out} "
        f"(enabled={doc['stats']['num_enabled']} "
        f"excluded={doc['stats']['num_excluded']} "
        f"threshold={args.threshold})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
