from __future__ import annotations

import logging
import re
from datetime import timedelta
from difflib import SequenceMatcher
from pathlib import Path

import srt

from src.ocr_reader import analyze_ocr_noise, normalized_for_merge, score_ocr_candidate


CYRILLIC_CHAR_RE = re.compile(r"[\u0400-\u04FF]")
CYRILLIC_TOKEN_RE = re.compile(r"[\u0400-\u04FF]+")
LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
GREEK_CHAR_RE = re.compile(r"[\u0370-\u03FF]")

SAFE_SHORT_SUBTITLES = {
    "да",
    "нет",
    "что",
    "кто",
    "радха",
    "кришна",
    "скажите",
    "господь",
}
SAFE_WORDS = {
    "бабочке",
    "бабочку",
    "более",
    "важнее",
    "верите",
    "господь",
    "доказал",
    "жизнь",
    "имя",
    "кришна",
    "любви",
    "любовь",
    "нет",
    "пепел",
    "посмотрите",
    "преданность",
    "произнесите",
    "произнесли",
    "радха",
    "радхи",
    "скажите",
    "теперь",
    "точку",
    "червяк",
    "ценно",
}

SAFE_WORDS.update({
    "\u0431\u0430\u0431\u043e\u0447\u043a\u0430",
    "\u0434\u0435\u0432\u0443\u0448\u043a\u0430",
    "\u0435\u0441\u043b\u0438",
    "\u0438\u0441\u0442\u0438\u043d\u0430",
    "\u043f\u043e\u0442\u043e\u043c\u0443",
    "\u043f\u043e\u0447\u0435\u043c\u0443",
    "\u0447\u0442\u043e",
})


def _similar(a: str, b: str) -> float:
    norm_a = normalized_for_merge(a)
    norm_b = normalized_for_merge(b)
    if norm_a and norm_b:
        return SequenceMatcher(None, norm_a, norm_b).ratio()
    return SequenceMatcher(None, a, b).ratio()


def _choose_voted_text(votes: list[dict]) -> tuple[str, float, float]:
    groups: list[dict] = []
    for vote in votes:
        text = str(vote["text"])
        key = normalized_for_merge(text) or text
        item = None
        for group in groups:
            if SequenceMatcher(None, key, str(group["key"])).ratio() >= 0.82:
                item = group
                break
        if item is None:
            item = {"key": key, "text": text, "count": 0, "confidence": 0.0, "score": 0.0}
            groups.append(item)
        item["count"] += 1
        item["confidence"] = max(float(item["confidence"]), float(vote.get("confidence", 0.0)))
        noise = analyze_ocr_noise(text)
        language_bonus = 8.0 if not noise.suspicious else -12.0
        vote_score = float(vote.get("score", score_ocr_candidate(text))) + language_bonus
        item["score"] = max(float(item["score"]), vote_score)
        if len(text) > len(str(item["text"])) and float(vote.get("confidence", 0.0)) >= float(item["confidence"]) * 0.80 and not noise.suspicious:
            item["text"] = text
    best = max(
        groups,
        key=lambda item: (int(item["count"]), float(item["confidence"]), float(item["score"]), len(str(item["text"]))),
    )
    return str(best["text"]), float(best["score"]), float(best["confidence"])


def _finalize_block(current: dict, min_duration: float) -> dict:
    text, score, confidence = _choose_voted_text(current.get("votes", []))
    finalized = {key: value for key, value in current.items() if key != "votes"}
    finalized["text"] = text
    finalized["score"] = score
    finalized["confidence"] = confidence
    finalized["end"] = max(float(finalized["end"]), float(finalized["start"]) + min_duration)
    return finalized


def build_subtitle_blocks(
    ocr_results: list[dict],
    sample_rate: float,
    min_duration: float = 0.7,
    max_join_gap: float = 1.2,
    similarity_threshold: float = 0.86,
) -> list[dict]:
    if not ocr_results:
        return []

    blocks: list[dict] = []
    current = {
        "start": float(ocr_results[0]["time"]),
        "end": float(ocr_results[0].get("end", float(ocr_results[0]["time"]) + sample_rate)),
        "text": str(ocr_results[0]["text"]),
        "score": float(ocr_results[0].get("score", score_ocr_candidate(str(ocr_results[0]["text"])))),
        "confidence": float(ocr_results[0].get("confidence", 0.0)),
        "votes": [
            {
                "text": str(ocr_results[0]["text"]),
                "score": float(ocr_results[0].get("score", score_ocr_candidate(str(ocr_results[0]["text"])))),
                "confidence": float(ocr_results[0].get("confidence", 0.0)),
            }
        ],
    }

    for item in ocr_results[1:]:
        item_time = float(item["time"])
        item_end = float(item.get("end", item_time + sample_rate))
        item_text = str(item["text"])
        item_score = float(item.get("score", score_ocr_candidate(item_text)))
        item_confidence = float(item.get("confidence", 0.0))
        gap = item_time - float(current["end"])
        is_same = _similar(str(current["text"]), item_text) >= similarity_threshold

        if is_same and gap <= max_join_gap:
            current["end"] = max(float(current["end"]), item_end)
            current.setdefault("votes", []).append({"text": item_text, "score": item_score, "confidence": item_confidence})
            voted_text, voted_score, voted_confidence = _choose_voted_text(current["votes"])
            current["text"] = voted_text
            current["score"] = voted_score
            current["confidence"] = voted_confidence
        else:
            blocks.append(_finalize_block(current, min_duration))
            current = {
                "start": item_time,
                "end": item_end,
                "text": item_text,
                "score": item_score,
                "confidence": item_confidence,
                "votes": [{"text": item_text, "score": item_score, "confidence": item_confidence}],
            }

    blocks.append(_finalize_block(current, min_duration))
    logging.info("Vytvořeno %d časových bloků titulků.", len(blocks))
    return blocks


def _has_safe_language_anchor(text: str) -> bool:
    normalized = re.sub(r"[^\u0400-\u04FF]+", " ", text.lower()).strip()
    if normalized in SAFE_SHORT_SUBTITLES:
        return True
    tokens = CYRILLIC_TOKEN_RE.findall(text.lower())
    return any(len(token) >= 3 and token in SAFE_WORDS for token in tokens)


def _is_suspicious_short_noise(text: str) -> bool:
    cyrillic_count = len(CYRILLIC_CHAR_RE.findall(text))
    if cyrillic_count == 0 or cyrillic_count >= 20:
        return False
    if _has_safe_language_anchor(text):
        return False

    tokens = CYRILLIC_TOKEN_RE.findall(text.lower())
    if not tokens:
        return bool(LATIN_CHAR_RE.search(text) or GREEK_CHAR_RE.search(text))

    short_fragments = sum(1 for token in tokens if len(token) <= 2)
    mostly_short_fragments = short_fragments / max(1, len(tokens)) > 0.50
    all_tiny_unknown = all(len(token) <= 3 for token in tokens)
    mixed_scripts = sum(bool(regex.search(text)) for regex in (CYRILLIC_CHAR_RE, LATIN_CHAR_RE, GREEK_CHAR_RE)) >= 2
    vowel_count = sum(1 for char in "".join(tokens) if char in "аеёиоуыэюя")
    low_language_shape = cyrillic_count >= 4 and vowel_count / max(1, cyrillic_count) < 0.20
    return mostly_short_fragments or all_tiny_unknown or mixed_scripts or low_language_shape


def _looks_like_sentence(tokens: list[str]) -> bool:
    longish_tokens = [token for token in tokens if len(token) >= 4]
    return len(longish_tokens) >= 2


def _is_suspicious_fragment_noise(text: str) -> bool:
    cyrillic_count = len(CYRILLIC_CHAR_RE.findall(text))
    if cyrillic_count < 20:
        return False
    if _has_safe_language_anchor(text):
        return False
    tokens = CYRILLIC_TOKEN_RE.findall(text.lower())
    if len(tokens) < 4:
        return False
    short_fragments = sum(1 for token in tokens if len(token) <= 2)
    high_short_fragment_ratio = short_fragments / max(1, len(tokens)) >= 0.55
    return high_short_fragment_ratio and not _looks_like_sentence(tokens)


def _near_real_neighbor(block: dict, previous_block: dict | None, next_block: dict | None) -> bool:
    text = str(block.get("text", ""))
    norm_text = normalized_for_merge(text)
    if not norm_text:
        return False
    for neighbor in (previous_block, next_block):
        if not neighbor:
            continue
        neighbor_text = str(neighbor.get("text", ""))
        if _is_suspicious_short_noise(neighbor_text):
            continue
        norm_neighbor = normalized_for_merge(neighbor_text)
        if norm_neighbor and SequenceMatcher(None, norm_text, norm_neighbor).ratio() >= 0.55:
            return True
    return False


def filter_obvious_ocr_noise_blocks(blocks: list[dict]) -> list[dict]:
    filtered: list[dict] = []
    suspicious_blocks = 0
    removed_noise_blocks = 0
    for index, block in enumerate(blocks):
        previous_block = blocks[index - 1] if index > 0 else None
        next_block = blocks[index + 1] if index + 1 < len(blocks) else None
        text = str(block.get("text", ""))
        if _is_suspicious_short_noise(text) or _is_suspicious_fragment_noise(text):
            suspicious_blocks += 1
            if not _near_real_neighbor(block, previous_block, next_block):
                removed_noise_blocks += 1
                continue
        filtered.append(block)
    logging.info("Post subtitle noise filter: suspicious_blocks=%d removed_noise_blocks=%d", suspicious_blocks, removed_noise_blocks)
    return filtered


def write_srt(blocks: list[dict], output_path: Path) -> Path:
    subtitles = [
        srt.Subtitle(
            index=index,
            start=timedelta(seconds=float(block["start"])),
            end=timedelta(seconds=float(block["end"])),
            content=str(block["text"]),
        )
        for index, block in enumerate(blocks, start=1)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(srt.compose(subtitles), encoding="utf-8")
    logging.info("SRT uloženo: %s", output_path)
    return output_path
