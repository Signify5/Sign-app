"""Tests for decide() and enable-list builder helpers."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

from sign_to_speech.safety.build_enable_list import build_enable_list, load_per_class_accuracies
from sign_to_speech.safety.decide import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    Decision,
    RiskyLexicon,
    ScoredLabel,
    decide,
    decision_to_dict,
)

SAFETY = Path(__file__).resolve().parent
GOLDEN = SAFETY / "golden" / "decision_vectors.json"
RISKY = SAFETY / "risky_pairs.json"
ENABLE = SAFETY / "enable_list.json"


def _approx_equal(a, b, tol: float = 1e-9) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return False
        return abs(float(a) - float(b)) <= tol
    return a == b


def _assert_decision_matches(got: Decision, expected: dict) -> None:
    as_dict = decision_to_dict(got)
    for key, want in expected.items():
        got_v = as_dict[key]
        if key in ("confidence", "margin"):
            assert _approx_equal(got_v, want), f"{key}: got={got_v!r} want={want!r}"
        else:
            assert got_v == want, f"{key}: got={got_v!r} want={want!r}"


@pytest.fixture(scope="module")
def lexicon() -> RiskyLexicon:
    return RiskyLexicon.load(RISKY)


def test_enable_list_is_placeholder() -> None:
    doc = json.loads(ENABLE.read_text(encoding="utf-8"))
    assert doc["status"] == "not_generated"
    assert doc["classes"] == []
    assert doc["threshold"] is None


def test_risky_pairs_only_inventory_glosses() -> None:
    manifest = json.loads(
        (SAFETY.parent / "dataset" / "lexical_manifest.json").read_text(encoding="utf-8")
    )
    inventory = {c["gloss"] for c in manifest["clips"]}
    doc = json.loads(RISKY.read_text(encoding="utf-8"))
    assert doc["status"] == "draft"
    assert doc["review_required"] is True
    for p in doc["pairs"]:
        assert p["a"] in inventory, p["a"]
        assert p["b"] in inventory, p["b"]
        assert p["severity"] in ("critical", "high", "medium")
    for g in doc["groups"]:
        for m in g["members"]:
            assert m in inventory, m
    for miss in doc["missing_wanted"]:
        assert miss["concept"] not in inventory


def test_lexicon_lookup_pair_and_group(lexicon: RiskyLexicon) -> None:
    hit = lexicon.lookup("yes", "no")
    assert hit is not None
    assert hit.severity == "critical"
    assert hit.source == "pair"

    hit2 = lexicon.lookup("no", "yes")
    assert hit2 is not None
    assert hit2.severity == "critical"

    num = lexicon.lookup("one", "nineteen")
    assert num is not None
    assert num.source == "group"
    assert num.category == "numbers"
    assert num.severity == "high"

    assert lexicon.lookup("water", "book") is None
    assert lexicon.lookup("yes", "yes") is None


def test_decide_emit_and_abstain_reasons(lexicon: RiskyLexicon) -> None:
    d = decide(
        [ScoredLabel("water", 0.9), ScoredLabel("book", 0.05)],
        lexicon,
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    )
    assert d.action == "emit"
    assert d.reason == "ok"
    assert d.label == "water"

    d2 = decide([("yes", 0.6), ("no", 0.5)], lexicon)
    assert d2.action == "abstain"
    assert d2.reason == "risky_confusion"
    assert d2.label is None
    assert d2.guess == "yes"
    assert d2.risky_pair == ("no", "yes")


def test_golden_decision_vectors(lexicon: RiskyLexicon) -> None:
    doc = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert doc["cases"], "golden set must not be empty"
    for case in doc["cases"]:
        result = decide(
            case["topk"],
            lexicon,
            confidence_threshold=float(case["confidence_threshold"]),
            margin_threshold=case["margin_threshold"],
        )
        try:
            _assert_decision_matches(result, case["expected"])
        except AssertionError as exc:
            raise AssertionError(f"golden case {case['id']}: {exc}") from exc


def test_build_enable_list_thresholding() -> None:
    classes = [
        {"gloss": "water", "accuracy": 0.97, "n": 10, "correct": 10},
        {"gloss": "yes", "accuracy": 0.90, "n": 10, "correct": 9},
        {"gloss": "book", "accuracy": 0.95, "n": 4, "correct": 4},
    ]
    doc = build_enable_list(classes, 0.95, source_accuracy_file="mem://test.json")
    assert doc["status"] == "generated"
    assert doc["classes"] == ["book", "water"]
    assert [e["gloss"] for e in doc["excluded"]] == ["yes"]
    assert doc["stats"]["num_enabled"] == 2
    assert doc["stats"]["num_excluded"] == 1


def test_load_per_class_accuracies_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "acc.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "split": "test",
                "metric": "top1_accuracy",
                "classes": [
                    {"gloss": "a", "accuracy": 1.0, "n": 2, "correct": 2},
                    {"gloss": "b", "accuracy": 0.5, "n": 2, "correct": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    classes, meta = load_per_class_accuracies(path)
    assert meta["split"] == "test"
    assert len(classes) == 2
    doc = build_enable_list(classes, 0.95)
    assert doc["classes"] == ["a"]


def test_load_per_class_accuracies_rejects_bad_accuracy(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"classes": [{"gloss": "a", "accuracy": 1.5}]}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        load_per_class_accuracies(path)


if __name__ == "__main__":
    # Lightweight runner without requiring pytest discovery flags.
    lex = RiskyLexicon.load(RISKY)
    test_enable_list_is_placeholder()
    test_risky_pairs_only_inventory_glosses()
    test_lexicon_lookup_pair_and_group(lex)
    test_decide_emit_and_abstain_reasons(lex)
    test_golden_decision_vectors(lex)
    test_build_enable_list_thresholding()
    with tempfile.TemporaryDirectory() as td:
        test_load_per_class_accuracies_roundtrip(Path(td))
        bad = Path(td) / "bad.json"
        bad.write_text(
            json.dumps({"classes": [{"gloss": "a", "accuracy": 1.5}]}),
            encoding="utf-8",
        )
        try:
            load_per_class_accuracies(bad)
            raise AssertionError("expected SystemExit")
        except SystemExit:
            pass
    print("all safety decide checks passed")
    sys.exit(0)
