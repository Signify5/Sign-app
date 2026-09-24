"""Unit tests for the shared punctuation/casing normalization pass."""
from __future__ import annotations

from postprocess import normalize


def test_capitalizes_first_letter():
    assert normalize("hello world") == "Hello world."


def test_adds_terminal_period_when_missing():
    assert normalize("this is a test") == "This is a test."


def test_preserves_existing_terminal_punctuation():
    assert normalize("is this working?") == "Is this working?"
    assert normalize("stop!") == "Stop!"


def test_collapses_internal_whitespace():
    assert normalize("hello   there\nworld") == "Hello there world."


def test_empty_input_stays_empty():
    assert normalize("") == ""
    assert normalize("   ") == ""
