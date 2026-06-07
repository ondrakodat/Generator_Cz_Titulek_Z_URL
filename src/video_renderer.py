from __future__ import annotations

import logging
import subprocess
from pathlib import Path


def _ffmpeg_filter_path(path: Path) -> str:
    # ffmpeg subtitles filter expects forward slashes and escaped drive colon on Windows.
    resolved = path.resolve().as_posix()
    if len(resolved) > 2 and resolved[1] == ":":
        resolved = resolved[0] + r"\:" + resolved[2:]
    return resolved.replace("'", r"\'")


def burn_subtitles(
    video_path: Path,
    srt_path: Path,
    output_path: Path,
    box_height_percent: float = 18.0,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    box_ratio = max(1.0, min(box_height_percent, 60.0)) / 100.0
    box_y = 1.0 - box_ratio
    srt_filter_path = _ffmpeg_filter_path(srt_path)
    video_filter = (
        f"drawbox=x=0:y=ih*{box_y:.4f}:w=iw:h=ih*{box_ratio:.4f}:"
        f"color=black@0.75:t=fill,subtitles='{srt_filter_path}'"
    )

    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        video_filter,
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    logging.info("Vypaluji české titulky do videa s kompatibilním H.264/AAC MP4 výstupem.")
    subprocess.run(args, check=True)

    logging.info("Hotové video uloženo: %s", output_path)
    return output_path
