from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from clips import Clip, ClipIndex, normalize_gloss
from engine import StatusEvent

SIGN = "sign"
FINGERSPELL = "fingerspell"
FS_PREFIX = "FS:"


@dataclass(frozen=True)
class PlaylistItem:
    gloss: str
    kind: str
    clips: Tuple[Clip, ...]


@dataclass
class Playlist:
    items: List[PlaylistItem] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def clips(self) -> List[Clip]:
        return [clip for item in self.items for clip in item.clips]

    def __len__(self) -> int:
        return len(self.items)


def build_playlist(
    glosses: Sequence[str],
    index: ClipIndex,
    on_status: Optional[Callable[[StatusEvent], None]] = None,
    authored_letters: bool = False,
) -> Playlist:
    """Ordered clips for a gloss list. Never raises for a missing word.

    A gloss with no clip is fingerspelled (so is gloss.py's FS:WORD, unless the library has WORD); characters with no letter clip, and
    glosses that cannot be played at all, are reported as 'warning' status events.
    With `authored_letters` the player spells from its own handshapes (fingerspell.py), so a
    fingerspelled item carries no clips and only characters outside A-Z are dropped.
    """
    playlist = Playlist()

    def warn(detail: str) -> None:
        playlist.warnings.append(detail)
        if on_status:
            on_status(StatusEvent("warning", detail))

    for raw in glosses:
        gloss = normalize_gloss(raw)
        if gloss.startswith(FS_PREFIX):  # gloss.py marks words it has no sign for: FS:WORD
            gloss = gloss[len(FS_PREFIX):].strip()
        if not gloss:
            continue

        clip = index.best(gloss)
        if clip is not None:
            playlist.items.append(PlaylistItem(gloss, SIGN, (clip,)))
            continue

        letters = [c for c in gloss if c.isalnum()]
        if authored_letters:
            spelled = "".join(c for c in letters if "A" <= c <= "Z")
            if spelled != "".join(letters):
                warn(f"cannot spell {', '.join(sorted(set(letters) - set(spelled)))} (in '{gloss}')")
            if not spelled:
                warn(f"'{gloss}' skipped: no clip and nothing to fingerspell")
                continue
            warn(f"no clip for '{gloss}', fingerspelling it")
            playlist.items.append(PlaylistItem(gloss, FINGERSPELL, ()))
            continue
        letter_clips = []
        missing = []
        for char in letters:
            letter_clip = index.letter(char)
            (letter_clips if letter_clip else missing).append(letter_clip or char)

        if missing:
            warn(f"no letter clip for {', '.join(sorted(set(missing)))} (in '{gloss}')")
        if not letter_clips:
            warn(f"'{gloss}' skipped: no clip and nothing to fingerspell")
            continue
        warn(f"no clip for '{gloss}', fingerspelling it")
        playlist.items.append(PlaylistItem(gloss, FINGERSPELL, tuple(letter_clips)))

    if not playlist.items and on_status:
        on_status(StatusEvent("error", "nothing playable for the given glosses"))
    return playlist
