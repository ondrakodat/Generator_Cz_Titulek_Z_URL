from __future__ import annotations

import json
import logging
import re
from datetime import timedelta
from pathlib import Path


HINDU_GLOSSARY = {
    "krishna": "Krishna",
    "radha": "Radha",
    "vishnu": "Vishnu",
    "shiva": "Shiva",
    "mahadev": "Mahadev",
    "narayan": "Narayan",
    "narayana": "Narayana",
    "lakshmi": "Lakshmi",
    "laxmi": "Laxmi",
    "brahma": "Brahma",
    "saraswati": "Saraswati",
    "parvati": "Parvati",
    "ganesha": "Ganesha",
    "ganesh": "Ganesh",
    "arjuna": "Arjuna",
    "yashoda": "Yashoda",
    "vrindavan": "Vrindavan",
    "mathura": "Mathura",
    "gokul": "Gokul",
    "dwarka": "Dwarka",
    "kurukshetra": "Kurukshetra",
    "dharma": "dharma",
    "karma": "karma",
    "bhakti": "bhakti",
    "maya": "maya",
    "avatar": "avatar",
    "asura": "asura",
    "deva": "deva",
    "mantra": "mantra",
    "puja": "puja",
    "yajna": "yajna",
    "prasad": "prasad",
    "leela": "leela",
    "gopi": "gopi",
    "gopika": "gopika",
    "guru": "guru",
    "rishi": "rishi",
    "muni": "muni",
    "shakti": "shakti",
    "atma": "atma",
    "moksha": "moksha",
    "samsara": "samsara",
    "vedas": "Vedas",
    "upanishads": "Upanishads",
    "bhagavad gita": "Bhagavad Gita",
}


def _format_time(seconds: float) -> str:
    td = timedelta(seconds=max(0.0, seconds))
    total_ms = int(td.total_seconds() * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _replace_glossary_terms(text: str) -> tuple[str, list[str]]:
    reasons: list[str] = []
    corrected = text
    for source, replacement in HINDU_GLOSSARY.items():
        pattern = re.compile(rf"\b{re.escape(source)}\b", re.IGNORECASE)
        if pattern.search(corrected):
            updated = pattern.sub(replacement, corrected)
            if updated != corrected:
                reasons.append(f"glossary:{replacement}")
                corrected = updated
    return corrected, reasons


def correct_hindu_context(blocks: list[dict], output_path: Path, corrections_path: Path) -> list[dict]:
    """Conservative post-OCR context correction.

    The function preserves block count and timing. It only applies deterministic
    glossary casing fixes and records every changed block for audit.
    """
    corrected_blocks: list[dict] = []
    corrections: list[dict] = []

    for index, block in enumerate(blocks, start=1):
        original = str(block.get("text", ""))
        corrected_text, reasons = _replace_glossary_terms(original)
        corrected_block = {**block, "text": corrected_text}
        corrected_blocks.append(corrected_block)
        if corrected_text != original:
            corrections.append(
                {
                    "index": index,
                    "start": _format_time(float(block["start"])),
                    "end": _format_time(float(block["end"])),
                    "original": original,
                    "corrected": corrected_text,
                    "reason": ", ".join(reasons) or "conservative glossary correction",
                    "confidence": float(block.get("confidence", 0.0)),
                }
            )

    corrections_path.parent.mkdir(parents=True, exist_ok=True)
    corrections_path.write_text(json.dumps(corrections, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Hindu context correction: %d oprav, pocet bloku zachovan: %d.", len(corrections), len(corrected_blocks))
    return corrected_blocks
