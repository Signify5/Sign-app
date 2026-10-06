from __future__ import annotations

from clips import Clip, ClipIndex
from playlist import FINGERSPELL, SIGN, build_playlist


def clip(gloss, clip_id, signer=1, quality=0.9):
    return Clip(gloss=gloss, path=f"/x/{clip_id}.npz", clip_id=clip_id, signer_id=signer, quality=quality)


def make_index():
    clips = [
        clip("hello", "h-low", signer=1, quality=0.5),
        clip("hello", "h-high", signer=2, quality=1.0),
        clip("world", "w1"),
        clip("a", "a1"),
        clip("b", "b1"),
    ]
    return ClipIndex(clips, letters={"C": clip("C", "letter-c")})


def collect():
    events = []
    return events, events.append


def test_items_follow_input_order():
    playlist = build_playlist(["world", "hello"], make_index())
    assert [i.gloss for i in playlist.items] == ["WORLD", "HELLO"]
    assert all(i.kind == SIGN for i in playlist.items)


def test_best_quality_clip_is_picked_deterministically():
    index = make_index()
    first = build_playlist(["HELLO"], index).clips()
    again = build_playlist(["hello"], index).clips()
    assert first == again
    assert first[0].clip_id == "h-high"


def test_tie_broken_by_clip_id():
    index = ClipIndex([clip("x", "b"), clip("x", "a")])
    assert index.best("X").clip_id == "a"


def test_missing_gloss_is_fingerspelled_with_warning():
    events, on_status = collect()
    playlist = build_playlist(["ABC"], make_index(), on_status)
    item = playlist.items[0]
    assert item.kind == FINGERSPELL
    assert [c.clip_id for c in item.clips] == ["a1", "b1", "letter-c"]
    assert [e.state for e in events] == ["warning"]
    assert "ABC" in events[0].detail


def test_dedicated_letter_clip_beats_wlasl_one_letter_gloss():
    index = ClipIndex([clip("a", "wlasl-a")], letters={"A": clip("A", "letter-a")})
    assert index.letter("a").clip_id == "letter-a"


def test_missing_letter_reported_but_rest_still_plays():
    events, on_status = collect()
    playlist = build_playlist(["AZB"], make_index(), on_status)
    assert [c.clip_id for c in playlist.items[0].clips] == ["a1", "b1"]
    assert any("Z" in e.detail for e in events)


def test_unplayable_gloss_skipped_and_others_kept():
    events, on_status = collect()
    playlist = build_playlist(["HELLO", "ZZZ", "WORLD"], make_index(), on_status)
    assert [i.gloss for i in playlist.items] == ["HELLO", "WORLD"]
    assert any("skipped" in e.detail for e in events)
    assert not any(e.state == "error" for e in events)


def test_nothing_playable_emits_error_not_exception():
    events, on_status = collect()
    playlist = build_playlist(["QQQ"], make_index(), on_status)
    assert len(playlist) == 0
    assert events[-1].state == "error"


def test_blank_and_empty_input():
    assert len(build_playlist([], make_index())) == 0
    assert len(build_playlist(["  ", ""], make_index())) == 0


def test_authored_letters_spell_without_letter_clips():
    events, on_status = collect()
    playlist = build_playlist(["hello", "xyz-2"], make_index(), on_status=on_status, authored_letters=True)
    spelled = playlist.items[1]
    assert (spelled.gloss, spelled.kind, spelled.clips) == ("XYZ-2", FINGERSPELL, ())
    assert any("cannot spell 2" in e.detail for e in events)
