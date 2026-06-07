from __future__ import annotations

import logging
import subprocess
from pathlib import Path


def _yt_dlp_args(url: str, output_path: Path, cookies_browser: str | None = None) -> list[str]:
    args = [
        "yt-dlp",
        "--no-part",
        "--merge-output-format",
        "mp4",
        "-f",
        "bv*+ba/best",
        "-o",
        str(output_path),
        url,
    ]
    if cookies_browser:
        args[1:1] = ["--cookies-from-browser", cookies_browser]
    return args


def download_video(url: str, output_path: Path, cookies_browser: str | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    attempts: list[tuple[str, list[str]]] = [
        ("běžné stažení přes yt-dlp", _yt_dlp_args(url, output_path)),
    ]
    if cookies_browser:
        attempts.append(
            (
                f"stažení přes yt-dlp s cookies z prohlížeče {cookies_browser}",
                _yt_dlp_args(url, output_path, cookies_browser),
            )
        )

    last_error: subprocess.CalledProcessError | None = None
    for label, args in attempts:
        logging.info("Zkouším %s.", label)
        try:
            subprocess.run(args, check=True)
            logging.info("Video uloženo do %s.", output_path)
            return output_path
        except subprocess.CalledProcessError as exc:
            logging.warning("Nepovedlo se: %s.", label)
            last_error = exc

    if cookies_browser is None:
        logging.info(
            "Pokud video vyžaduje přihlášení, zkuste --cookies-browser chrome nebo --cookies-browser edge."
        )
    raise RuntimeError("Stažení videa selhalo.") from last_error

