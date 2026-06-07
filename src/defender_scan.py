from __future__ import annotations

import logging
import platform
import subprocess
from pathlib import Path

from src.utils import is_running_in_docker


def scan_file_with_defender(path: Path) -> bool:
    if is_running_in_docker():
        logging.warning("Windows Defender kontrola není dostupná uvnitř Docker kontejneru.")
        return False

    if platform.system().lower() != "windows":
        logging.warning("Microsoft Defender scan je dostupný pouze na Windows.")
        return False

    scan_path = path.resolve()
    if not scan_path.exists():
        logging.error("Soubor pro Defender scan neexistuje: %s", scan_path)
        return False

    escaped_path = str(scan_path).replace("'", "''")
    command = (
        "Start-MpScan "
        "-ScanType CustomScan "
        f"-ScanPath '{escaped_path}'"
    )
    args = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        command,
    ]

    logging.info("Spouštím Microsoft Defender kontrolu souboru: %s", scan_path)
    try:
        result = subprocess.run(args, check=False, capture_output=True, text=True)
    except OSError as exc:
        logging.warning("PowerShell nebo Microsoft Defender není dostupný: %s", exc)
        return False

    if result.returncode == 0:
        logging.info("Microsoft Defender kontrola proběhla bez chyby.")
        return True

    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    details = stderr or stdout or f"návratový kód {result.returncode}"
    logging.warning("Microsoft Defender kontrola selhala: %s", details)
    return False
