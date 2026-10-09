import numpy as np
import pytest

import library
from clips import Clip, ClipIndex


def test_delta_round_trips_with_wraparound():
    a = np.array([[100, -120], [-120, 127], [5, 5]], "i1")
    assert np.array_equal(np.cumsum(library._delta(a), axis=0, dtype="i1"), a)
    b = np.array([[32000, -5], [-32000, 7]], "<i2")
    assert np.array_equal(np.cumsum(library._delta(b), axis=0, dtype="<i2"), b)


def test_vocabulary_leaves_letters_to_the_alphabet_but_keeps_the_pronoun():
    clips = [Clip(gloss=g, path="/x.npz", clip_id=g, signer_id=1, quality=1.0) for g in ("a", "i", "thank you", "go")]
    assert sorted(library.vocabulary(ClipIndex(clips))) == ["GO", "I", "THANK YOU"]


def test_presets_must_be_signable(monkeypatch):
    monkeypatch.setattr(library, "PRESETS", ["THANK YOU VEDANT"])
    library.check_presets(["THANK YOU"])  # multiword sign plus a name that is meant to be spelled
    monkeypatch.setattr(library, "PRESETS", ["THANK YOU FRIEND"])
    with pytest.raises(SystemExit):
        library.check_presets(["THANK YOU"])


def test_vocabulary_leaves_out_signs_that_fail_a_sign_check():
    clips = [Clip("go", "/x.npz", "go", 1, 1.0), Clip("cat", "/x.npz", "cat-1", 1, 1.0, faults=10),
             Clip("dog", "/x.npz", "dog-1", 1, 1.0, faults=10), Clip("dog", "/x.npz", "dog-2", 1, 0.5, faults=1)]
    index = ClipIndex(clips)
    assert library.vocabulary(index) == ["DOG", "GO"]  # DOG has a signing that passes
    assert library.failed_glosses(index) == ["CAT"]
    assert library.vocabulary(index, keep_failed=True) == ["CAT", "DOG", "GO"]
