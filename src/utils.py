from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from pathlib import Path


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise RuntimeError(
            f"Program '{command}' nebyl nalezen v PATH. Nainstalujte ho a spusťte příkaz znovu."
        )


def check_required_tools(require_ytdlp: bool = True) -> None:
    require_command("ffmpeg")
    if require_ytdlp:
        require_command("yt-dlp")


def run_command(args: list[str], description: str) -> None:
    logging.info("%s", description)
    logging.debug("Spouštím: %s", " ".join(args))
    try:
        subprocess.run(args, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Příkaz selhal ({description}) s kódem {exc.returncode}.") from exc


def parse_subtitle_area(area: str) -> float:
    kind, _, value = area.partition(":")
    if kind != "bottom" or not value:
        raise ValueError("Parametr --subtitle-area musí mít formát například bottom:25.")
    percent = float(value)
    if percent <= 0 or percent > 80:
        raise ValueError("Hodnota subtitle-area musí být mezi 0 a 80 procenty.")
    return percent / 100.0


def parse_hhmmss(value: str | None, option_name: str) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if not re.fullmatch(r"\d{2}:\d{2}:\d{2}", value):
        raise ValueError(f"Parametr {option_name} musi mit format HH:MM:SS.")
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Parametr {option_name} musi mit minuty a sekundy v rozsahu 00-59.")
    return float(hours * 3600 + minutes * 60 + seconds)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def is_running_in_docker() -> bool:
    if os.environ.get("RUNNING_IN_DOCKER") == "1":
        return True
    return Path("/.dockerenv").exists()
