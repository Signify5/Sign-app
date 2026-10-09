from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

from clips import Clip
from motion import Motion, MotionError, Timeline, compose
from playlist import Playlist
from render import SIZE, render_frame


class ExportError(RuntimeError):
    pass


def load_items(clips: List[Clip], labels: List[str]) -> List[Tuple[Motion, str]]:
    try:
        return [(Motion.load(c.path), label) for c, label in zip(clips, labels)]
    except MotionError as exc:
        raise ExportError(str(exc)) from exc


def build_timeline(playlist: Playlist, gap_s: float = 0.0, speed: float = 1.0) -> Timeline:
    """One continuous animation for a playlist (labels are the gloss, or the letter when fingerspelling)."""
    clips, labels = [], []
    for item in playlist.items:
        for clip in item.clips:
            clips.append(clip)
            labels.append(item.gloss if item.kind == "sign" else f"{item.gloss} ({clip.gloss.upper()})")
    if not clips:
        raise ExportError("nothing to export: playlist has no clips")
    try:
        return compose(load_items(clips, labels), gap_s=gap_s, speed=speed)
    except MotionError as exc:
        raise ExportError(str(exc)) from exc


def export_playlist(
    playlist: Playlist,
    out_path: Union[Path, str],
    gap_s: float = 0.0,
    speed: float = 1.0,
    size: int = SIZE,
    mirror: bool = False,
) -> Path:
    """Render a playlist to one MP4 (H.264). Raises ExportError on failure."""
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise ExportError("MP4 export needs `pip install imageio-ffmpeg`") from exc

    timeline = build_timeline(playlist, gap_s, speed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(out_path), (size, size), fps=timeline.fps, codec="libx264", pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p", quality=7, macro_block_size=1,
    )
    writer.send(None)
    try:
        for i in range(timeline.frames):
            writer.send(render_frame(timeline.pose(i), size, mirror=mirror))
    except Exception as exc:
        raise ExportError(f"export failed: {exc}") from exc
    finally:
        writer.close()
    return out_path
