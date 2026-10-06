from __future__ import annotations

import pytest

from clips import Clip, ClipIndex, ClipIndexError


def test_write_load_roundtrip_with_relative_paths(tmp_path):
    video = tmp_path / "wlasl" / "BOOK" / "1.npz"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    letter = tmp_path / "letters" / "A.npz"
    letter.parent.mkdir()
    letter.write_bytes(b"x")

    clips = [Clip("book", str(video), "wlasl-1", 7, 0.8)]
    ClipIndex.write(tmp_path / "index.json", clips, {"A": Clip("A", str(letter), "letter-A")})

    loaded = ClipIndex.load(tmp_path)
    got = loaded.best("BOOK")
    assert got.path == str(video.resolve())
    assert (got.quality, got.signer_id) == (0.8, 7)
    assert loaded.letter("a").clip_id == "letter-A"
    assert loaded.glosses == ["BOOK"]


def test_missing_index_raises_actionable_error(tmp_path):
    with pytest.raises(ClipIndexError, match="setup_clips.py"):
        ClipIndex.load(tmp_path / "nope.json")


def test_signer_coverage_breaks_quality_ties():
    same = dict(quality=0.9)
    index = ClipIndex([
        Clip("a", "/1", "a-rare", 99, **same),
        Clip("a", "/2", "a-common", 1, **same),
        Clip("b", "/3", "b1", 1, **same),
    ])
    assert index.best("A").clip_id == "a-common"
