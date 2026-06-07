from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import re
import shutil
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from statistics import median
from urllib.parse import urljoin

import cv2
import numpy as np
import pytesseract
import requests

from .frame_extractor import ExtractionStats, FrameCrop
from .utils import normalize_text

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
CYRILLIC_WORD_RE = re.compile(r"[А-Яа-яЁё]{1,}")
LATIN_RE = re.compile(r"[A-Za-z]")
LEADING_NOISE_RE = re.compile(r"^(?:N{1,4}|Н{2,4}|П{2,4})\s*[:\-–—.]?\s*", re.IGNORECASE)
CREDIT_PATTERNS = (
    "vk.com/starseriales",
    "starseriales",
    "перево",
    "перевод",
    "тайминг",
    "редактор",
    "маржан",
    "аржан",
    "жамилова",
    "жамилова",
    "сериал",
)
SYMBOLS = set(".-_|\\/`^*%$#=+~:;,.!?()[]{}<>\"'")
SHORT_WORDS = {"я", "и", "в", "не", "но", "да", "он", "мы", "вы", "ты", "на", "за", "к", "у", "с", "о", "а"}
RUSSIAN_VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
COMMON_RU_WORDS = {
    "и", "в", "не", "что", "как", "это", "но", "на", "я", "ты", "он", "она", "мы", "вы", "они",
    "мне", "тебя", "меня", "когда", "почему", "нужно", "нашем", "мире", "каждого", "господа",
    "богини", "имя", "радхи", "любви", "есть", "только", "одна", "землю", "история", "началась",
}
# Keep Cyrillic handling independent of the process code page.
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
CYRILLIC_WORD_RE = re.compile(r"[\u0400-\u04FF]{1,}")
LEADING_NOISE_RE = re.compile(r"^(?:N{1,4}|\u041b{2,4}|\u041c{2,4})\s*[:\-–—.]?\s*", re.IGNORECASE)
CREDIT_PATTERNS = (
    "vk.com/starseriales",
    "starseriales",
    "\u043f\u0435\u0440\u0435\u0432\u043e",
    "\u043f\u0435\u0440\u0435\u0432\u043e\u0434",
    "\u0442\u0430\u0439\u043c\u0438\u043d\u0433",
    "\u0440\u0435\u0434\u0430\u043a\u0442\u043e\u0440",
    "\u0441\u0435\u0440\u0438\u0430\u043b",
)
SHORT_WORDS = {"\u044f", "\u0438", "\u0432", "\u043d\u0435", "\u043d\u043e", "\u0434\u0430", "\u043e\u043d", "\u043c\u044b", "\u0432\u044b", "\u0442\u044b", "\u043d\u0430", "\u0437\u0430", "\u043a", "\u0443", "\u0441", "\u043e", "\u0430"}
VALID_SHORT_SUBTITLES = {
    "\u0434\u0430", "\u043d\u0435\u0442", "\u0447\u0442\u043e", "\u043a\u0442\u043e", "\u0441\u0442\u043e\u0439", "\u0438\u0434\u0438", "\u0443\u0439\u0434\u0438", "\u0431\u0435\u0433\u0438",
    "\u044f", "\u0442\u044b", "\u043e\u043d", "\u043e\u043d\u0430", "\u043c\u044b", "\u0432\u044b", "\u043e\u043d\u0438", "\u043d\u0443", "\u043e\u0439", "\u0430\u0445", "\u044d\u0439",
    "\u0440\u0430\u0434\u0445\u0430", "\u043a\u0440\u0438\u0448\u043d\u0430", "\u0433\u043e\u0441\u043f\u043e\u0434\u044c",
}
RUSSIAN_VOWELS = set("\u0430\u0435\u0451\u0438\u043e\u0443\u044b\u044d\u044e\u044f\u0410\u0415\u0401\u0418\u041e\u0423\u042b\u042d\u042e\u042f")
COMMON_RU_WORDS = {
    "\u0438", "\u0432", "\u043d\u0435", "\u0447\u0442\u043e", "\u043a\u0430\u043a", "\u044d\u0442\u043e", "\u043d\u043e", "\u043d\u0430", "\u044f", "\u0442\u044b", "\u043e\u043d", "\u043e\u043d\u0430", "\u043c\u044b", "\u0432\u044b", "\u043e\u043d\u0438",
    "\u043c\u043d\u0435", "\u0442\u0435\u0431\u044f", "\u043c\u0435\u043d\u044f", "\u043a\u043e\u0433\u0434\u0430", "\u043f\u043e\u0447\u0435\u043c\u0443", "\u043d\u0443\u0436\u043d\u043e", "\u043d\u0430\u0448\u0435\u043c", "\u043c\u0438\u0440\u0435",
    "\u0431\u043e\u0433\u0438\u043d\u0438", "\u0438\u043c\u044f", "\u0440\u0430\u0434\u0445\u0438", "\u043b\u044e\u0431\u0432\u0438", "\u0435\u0441\u0442\u044c", "\u0442\u043e\u043b\u044c\u043a\u043e",
}
SPACE_HINT_WORDS = {
    "\u044f", "\u0442\u044b", "\u043e\u043d", "\u043e\u043d\u0430", "\u043c\u044b", "\u0432\u044b", "\u043e\u043d\u0438", "\u043d\u0435", "\u043d\u0430", "\u0432", "\u0438", "\u0447\u0442\u043e", "\u043a\u0430\u043a", "\u044d\u0442\u043e",
    "\u0442\u0435\u0431\u044f", "\u043c\u0435\u043d\u044f", "\u043c\u043d\u0435", "\u0442\u0432\u043e\u0435\u0439", "\u043c\u043e\u0435\u0439", "\u043f\u0440\u0435\u0434\u0430\u043d\u043d\u043e\u0441\u0442\u0438",
    "\u0441\u043e\u043c\u043d\u0435\u0432\u0430\u044e\u0441\u044c", "\u0433\u043e\u0441\u043f\u043e\u0434\u044c", "\u0440\u0430\u0434\u0445\u0430", "\u043a\u0440\u0438\u0448\u043d\u0430",
}
OLLAMA_SANITY_PROMPT = (
    "Je tento text pravděpodobně poškozená ruská věta z OCR, nebo jen náhodný šum? "
    "Odpověz pouze VALID nebo NOISE.\n\n{text}"
)


@dataclass
class OcrEngineResult:
    text: str
    confidence: float
    boxes: list | None = None


@dataclass
class OcrNoiseDecision:
    suspicious: bool
    reasons: list[str]
    cleaned_text: str
    meaningful_chars: int
    is_valid_short: bool = False


@dataclass
class DeepOcrStats:
    attempted: int = 0
    recovered: int = 0
    rejected: int = 0


def similar(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def normalized_for_merge(text: str) -> str:
    text = normalize_text(text).lower()
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def _tesseract_lang(lang: str) -> str:
    return {"ru": "rus"}.get(lang, lang)


def _count_cyrillic(text: str) -> int:
    return len(CYRILLIC_RE.findall(text))


def _count_letters(text: str) -> int:
    return sum(1 for char in text if char.isalpha())


def _count_latin(text: str) -> int:
    return len(LATIN_RE.findall(text))


def _count_symbols(text: str) -> int:
    return sum(1 for char in text if not char.isalnum() and not char.isspace())


def _meaningful_text(text: str) -> str:
    return "".join(char for char in normalize_text(text) if char.isalnum())


def _is_valid_short_subtitle(text: str) -> bool:
    cleaned = normalize_text(text).strip(" .,-_|\\/`^*%$#;:!?\"'").lower()
    words = CYRILLIC_WORD_RE.findall(cleaned)
    joined = "".join(words)
    return bool(joined and (joined in VALID_SHORT_SUBTITLES or (len(words) == 1 and words[0] in VALID_SHORT_SUBTITLES)))


def analyze_ocr_noise(text: str) -> OcrNoiseDecision:
    cleaned = normalize_text(text)
    meaningful = _meaningful_text(cleaned)
    meaningful_chars = len(meaningful)
    valid_short = _is_valid_short_subtitle(cleaned)
    reasons: list[str] = []
    cyrillic_count = _count_cyrillic(cleaned)
    non_space_len = max(1, len(cleaned.replace(" ", "")))
    cyrillic_ratio = cyrillic_count / non_space_len
    if cyrillic_ratio < 0.40 and not valid_short:
        reasons.append("low_cyrillic_ratio")
    words = CYRILLIC_WORD_RE.findall(cleaned)
    if len(words) >= 3:
        one_char_words = sum(1 for word in words if len(word) == 1)
        if one_char_words / max(1, len(words)) > 0.50:
            reasons.append("too_many_one_char_words")
    if len(words) >= 4 and not valid_short:
        short_words = sum(1 for word in words if len(word) <= 2)
        common_words = sum(1 for word in words if word.lower() in COMMON_RU_WORDS)
        if short_words / max(1, len(words)) > 0.75 and common_words < max(2, len(words) * 0.35):
            reasons.append("too_many_short_fragments")
    if meaningful_chars < 4 and not valid_short:
        reasons.append("too_few_meaningful_chars")
    upper_words = [word for word in words if len(word) >= 4 and word.isupper()]
    if upper_words and sum(len(word) for word in upper_words) >= 6 and not valid_short:
        reasons.append("random_uppercase_sequence")
    repeated_pairs = re.findall(r"([\u0400-\u04FF])\1{1,}", cleaned, flags=re.IGNORECASE)
    if len(repeated_pairs) >= 2 or re.search(r"([\u0400-\u04FF])\1{2,}", cleaned, flags=re.IGNORECASE):
        reasons.append("repeated_letters")
    if words and not valid_short:
        vowel_count = sum(1 for char in "".join(words) if char in RUSSIAN_VOWELS)
        if len("".join(words)) >= 8 and vowel_count / max(1, len("".join(words))) < 0.12:
            reasons.append("too_few_vowels")
    return OcrNoiseDecision(bool(reasons), reasons, cleaned, meaningful_chars, valid_short)


def analyze_legacy_good_noise(text: str) -> OcrNoiseDecision:
    cleaned = normalize_text(text)
    meaningful_chars = len(_meaningful_text(cleaned))
    valid_short = _is_valid_short_subtitle(cleaned)
    cyrillic_count = _count_cyrillic(cleaned)
    reasons: list[str] = []
    if not cleaned:
        reasons.append("empty")
    if meaningful_chars <= 3 and not valid_short:
        reasons.append("very_short_noise")
    non_space_len = max(1, len(cleaned.replace(" ", "")))
    cyrillic_ratio = cyrillic_count / non_space_len
    if cyrillic_ratio < 0.12 and not valid_short:
        reasons.append("extremely_low_cyrillic_ratio")
    if cyrillic_count > 15:
        reasons = [reason for reason in reasons if reason not in {"very_short_noise"}]
    if cyrillic_count > 15 and cyrillic_ratio >= 0.12:
        reasons = []
    return OcrNoiseDecision(bool(reasons), reasons, cleaned, meaningful_chars, valid_short)


def common_russian_word_count(text: str) -> int:
    return sum(1 for word in CYRILLIC_WORD_RE.findall(normalize_text(text).lower()) if word in COMMON_RU_WORDS)


def has_good_language_validity(text: str) -> bool:
    cleaned = normalize_text(text)
    words = CYRILLIC_WORD_RE.findall(cleaned)
    if not words:
        return False
    decision = analyze_ocr_noise(cleaned)
    if decision.is_valid_short:
        return True
    cyrillic_chars = "".join(words)
    vowel_count = sum(1 for char in cyrillic_chars if char in RUSSIAN_VOWELS)
    longish_words = sum(1 for word in words if len(word) >= 3)
    common_count = common_russian_word_count(cleaned)
    vowel_ratio = vowel_count / max(1, len(cyrillic_chars))
    return (
        not decision.suspicious
        and len(cyrillic_chars) >= 5
        and vowel_ratio >= 0.22
        and (common_count > 0 or longish_words >= 2 or len(cyrillic_chars) >= 12)
    )


def is_acceptable_deep_recovery(text: str) -> bool:
    decision = analyze_ocr_noise(text)
    return decision.is_valid_short or common_russian_word_count(text) > 0 or has_good_language_validity(text)


def repair_joined_russian_spacing(text: str) -> str:
    cleaned = normalize_text(text)
    words = CYRILLIC_WORD_RE.findall(cleaned)
    if " " in cleaned and len(words) >= 3:
        return cleaned
    compact = normalized_for_merge(cleaned)
    phrase_fixes = {
        "\u044f\u043d\u0435\u0441\u043e\u043c\u043d\u0435\u0432\u0430\u044e\u0441\u044c\u0432\u0442\u0432\u043e\u0435\u0439\u043f\u0440\u0435\u0434\u0430\u043d\u043d\u043e\u0441\u0442\u0438": "\u042f \u043d\u0435 \u0441\u043e\u043c\u043d\u0435\u0432\u0430\u044e\u0441\u044c \u0432 \u0442\u0432\u043e\u0435\u0439 \u043f\u0440\u0435\u0434\u0430\u043d\u043d\u043e\u0441\u0442\u0438",
    }
    if compact in phrase_fixes:
        return phrase_fixes[compact]
    repaired = cleaned
    for word in sorted((item for item in SPACE_HINT_WORDS if len(item) >= 4), key=len, reverse=True):
        pattern = re.compile(rf"(?<!^)(?={re.escape(word)})|(?<={re.escape(word)})(?!$)", re.IGNORECASE)
        repaired = pattern.sub(" ", repaired)
    repaired = normalize_text(re.sub(r"\s+", " ", repaired))
    return repaired if _count_cyrillic(repaired) >= _count_cyrillic(cleaned) else cleaned


def looks_like_damaged_russian_sentence(text: str) -> bool:
    cleaned = normalize_text(text)
    cyrillic_count = _count_cyrillic(cleaned)
    if cyrillic_count <= 20:
        return False
    words = CYRILLIC_WORD_RE.findall(cleaned)
    if len(words) < 3:
        words = CYRILLIC_WORD_RE.findall(repair_joined_russian_spacing(cleaned))
    cyrillic_chars = "".join(words)
    if not cyrillic_chars:
        return False
    vowel_count = sum(1 for char in cyrillic_chars if char in RUSSIAN_VOWELS)
    vowel_ratio = vowel_count / max(1, len(cyrillic_chars))
    short_words = sum(1 for word in words if len(word) <= 2)
    uppercase = sum(1 for char in cyrillic_chars if char.isupper())
    return (
        len(words) >= 3
        and vowel_ratio >= 0.18
        and short_words / max(1, len(words)) <= 0.55
        and uppercase / max(1, len(cyrillic_chars)) <= 0.65
    )


def ollama_language_sanity_check(text: str, model: str, base_url: str, timeout_seconds: float = 8.0) -> str | None:
    try:
        response = requests.post(
            urljoin(base_url.rstrip("/") + "/", "api/generate"),
            json={
                "model": model,
                "prompt": OLLAMA_SANITY_PROMPT.format(text=text),
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        answer = str(response.json().get("response", "")).strip().upper()
    except Exception as exc:
        logging.debug("Ollama OCR sanity check unavailable: %s", exc)
        return None
    if answer.startswith("VALID"):
        return "VALID"
    if answer.startswith("NOISE"):
        return "NOISE"
    return None


def language_sanity_check(
    text: str,
    *,
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
    use_ollama: bool = False,
) -> tuple[bool, str, str]:
    repaired = repair_joined_russian_spacing(text)
    if common_russian_word_count(repaired) > 0 or looks_like_damaged_russian_sentence(repaired):
        return True, repaired, "local_language_sanity"
    if use_ollama and ollama_model and ollama_base_url:
        answer = ollama_language_sanity_check(repaired, ollama_model, ollama_base_url)
        if answer == "VALID":
            return True, repaired, "ollama_language_sanity"
        if answer == "NOISE":
            return False, repaired, "ollama_language_noise"
    return False, repaired, "language_sanity_noise"


def clean_ocr_text(text: str, min_cyrillic_ratio: float = 0.35, min_cyrillic_chars: int = 4, preset: str = "strict") -> str:
    cleaned, _ = clean_ocr_text_with_reason(text, min_cyrillic_ratio, min_cyrillic_chars, preset)
    return cleaned


def clean_ocr_text_with_reason(
    text: str,
    min_cyrillic_ratio: float = 0.35,
    min_cyrillic_chars: int = 4,
    preset: str = "strict",
) -> tuple[str, str]:
    lines: list[str] = []
    for line in text.splitlines():
        cleaned = normalize_text(line)
        cleaned = LEADING_NOISE_RE.sub("", cleaned).strip()
        if preset == "legacy_good":
            cyrillic_count = _count_cyrillic(cleaned)
            if cyrillic_count > 15:
                lines.append(cleaned)
                continue
            if not cleaned:
                continue
            non_space_len = max(1, len(cleaned.replace(" ", "")))
            if len(_meaningful_text(cleaned)) <= 3 and not _is_valid_short_subtitle(cleaned):
                continue
            if cyrillic_count / non_space_len < 0.12 and not _is_valid_short_subtitle(cleaned):
                continue
            lines.append(cleaned)
            continue
        valid_short_line = _is_valid_short_subtitle(cleaned)
        if len(cleaned) < 3 and not valid_short_line:
            continue
        lowered_line = cleaned.lower()
        if any(pattern in lowered_line for pattern in CREDIT_PATTERNS):
            continue
        line_cyrillic = _count_cyrillic(cleaned)
        line_latin = _count_latin(cleaned)
        if line_latin > max(3, line_cyrillic):
            continue
        if preset not in {"recall", "maximum_recall"} and _count_symbols(cleaned) > max(3, _count_letters(cleaned)):
            continue
        words = CYRILLIC_WORD_RE.findall(cleaned)
        words = _trim_noise_words(words)
        if preset == "strict" and not valid_short_line and _looks_like_gibberish_words(words):
            continue
        if preset == "balanced" and not valid_short_line and _looks_like_gibberish_words(words, relaxed=True):
            continue
        joined = " ".join(words)
        if len(joined) < 3 and not valid_short_line:
            continue
        lines.append(joined if joined else cleaned)

    cleaned = normalize_text(" ".join(lines))
    if not cleaned:
        return "", "empty_after_cleanup"
    lowered = cleaned.lower()
    if any(pattern in lowered for pattern in CREDIT_PATTERNS):
        return "", "credit_or_technical_text"
    valid_short = _is_valid_short_subtitle(cleaned)
    if preset == "strict" and len(cleaned) > 160:
        return "", "too_long_strict"
    if preset == "balanced" and len(cleaned) > 240:
        return "", "too_long_balanced"
    cyrillic_count = _count_cyrillic(cleaned)
    if preset == "legacy_good":
        decision = analyze_legacy_good_noise(cleaned)
        return (cleaned, "accepted") if not decision.suspicious else ("", ",".join(decision.reasons))
    if cyrillic_count < min_cyrillic_chars and not valid_short:
        return "", "too_few_cyrillic_chars"
    latin_count = _count_latin(cleaned)
    if latin_count > max(2, cyrillic_count * 0.45):
        return "", "too_many_latin_letters"
    non_space_len = max(1, len(cleaned.replace(" ", "")))
    cyrillic_ratio = cyrillic_count / non_space_len
    if cyrillic_ratio < min_cyrillic_ratio and not valid_short:
        return "", "low_cyrillic_ratio"
    letters = _count_letters(cleaned)
    symbols = _count_symbols(cleaned)
    if preset not in {"recall", "maximum_recall"} and symbols > letters:
        return "", "more_symbols_than_letters"
    symbol_only = sum(1 for char in cleaned if char in SYMBOLS or char.isspace())
    if preset == "strict" and symbol_only / max(1, len(cleaned)) > 0.55:
        return "", "mostly_symbols"
    if preset == "balanced" and symbol_only / max(1, len(cleaned)) > 0.70:
        return "", "mostly_symbols"
    return cleaned, "accepted"


def _looks_like_gibberish_words(words: list[str], relaxed: bool = False) -> bool:
    if not words:
        return True
    if not relaxed and len(words) <= 2 and max(len(word) for word in words) < 5:
        return True
    meaningful_words = [word for word in words if len(word) > 1 or word.lower() in SHORT_WORDS]
    if len(meaningful_words) < max(1, len(words) * (0.45 if relaxed else 0.6)):
        return True
    short_noise = sum(1 for word in words if len(word) <= 2 and word.lower() not in SHORT_WORDS)
    if short_noise / max(1, len(words)) > (0.65 if relaxed else 0.45):
        return True
    cyrillic_chars = "".join(words)
    vowel_count = sum(1 for char in cyrillic_chars if char in RUSSIAN_VOWELS)
    if len(cyrillic_chars) >= 8 and vowel_count / len(cyrillic_chars) < (0.14 if relaxed else 0.22):
        return True
    upper_count = sum(1 for char in cyrillic_chars if char.isupper())
    if len(cyrillic_chars) >= 5 and upper_count / len(cyrillic_chars) > (0.75 if relaxed else 0.45):
        return True
    common_count = sum(1 for word in words if word.lower() in COMMON_RU_WORDS)
    if not relaxed and len(cyrillic_chars) < 8 and common_count == 0:
        return True
    if not relaxed and len(words) >= 4 and common_count == 0 and len(cyrillic_chars) < 42:
        return True
    long_words = [word for word in words if len(word) >= 3]
    if long_words:
        no_vowel_words = sum(1 for word in long_words if not any(char in RUSSIAN_VOWELS for char in word))
        if no_vowel_words / len(long_words) > (0.55 if relaxed else 0.35):
            return True
    return False


def _trim_noise_words(words: list[str]) -> list[str]:
    if not words:
        return []
    start = 0
    while start < len(words) - 1 and len(words[start]) == 1 and words[start].lower() not in {"я"}:
        start += 1
    end = len(words)
    while end > start + 1 and len(words[end - 1]) == 1 and words[end - 1].lower() not in {"я"}:
        end -= 1
    trimmed = words[start:end]
    if len(trimmed) >= 3:
        single_count = sum(1 for word in trimmed if len(word) == 1 and word.lower() not in SHORT_WORDS)
        if single_count > len(trimmed) * 0.35:
            trimmed = [word for word in trimmed if len(word) > 1 or word.lower() in SHORT_WORDS]
    return trimmed


def _trim_noise_words(words: list[str]) -> list[str]:
    if not words:
        return []
    start = 0
    while start < len(words) - 1 and len(words[start]) == 1 and words[start].lower() not in {"\u044f"}:
        start += 1
    end = len(words)
    while end > start + 1 and len(words[end - 1]) == 1 and words[end - 1].lower() not in {"\u044f"}:
        end -= 1
    trimmed = words[start:end]
    if len(trimmed) >= 3:
        single_count = sum(1 for word in trimmed if len(word) == 1 and word.lower() not in SHORT_WORDS)
        if single_count > len(trimmed) * 0.35:
            trimmed = [word for word in trimmed if len(word) > 1 or word.lower() in SHORT_WORDS]
    return trimmed


def score_ocr_candidate(text: str, confidence: float = 0.0) -> float:
    if not text:
        return 0.0
    cyrillic = _count_cyrillic(text)
    letters = _count_letters(text)
    symbols = _count_symbols(text)
    non_space_len = max(1, len(text.replace(" ", "")))
    cyrillic_ratio = cyrillic / non_space_len
    repeated_penalty = 8 if re.search(r"(.)\1{4,}", text) else 0
    no_space_penalty = 10 if len(text) > 28 and " " not in text else 0
    upper_count = sum(1 for char in text if char.isupper())
    upper_penalty = 8 if cyrillic >= 6 and upper_count / max(1, cyrillic) > 0.55 else 0
    length_penalty = max(0, len(text) - 150) * 0.04
    return (
        confidence * 35.0
        + cyrillic * 1.8
        + letters * 0.2
        + cyrillic_ratio * 25.0
        - symbols * 1.2
        - repeated_penalty
        - no_space_penalty
        - upper_penalty
        - length_penalty
    )


def preprocess_image(image: np.ndarray, mode: str = "subtitle", subtitle_style: str = "auto") -> np.ndarray:
    if mode == "none":
        return image
    if mode == "auto":
        mode = "subtitle"
    if mode == "grayscale":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    if mode == "contrast":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        return cv2.convertScaleAbs(gray, alpha=1.75, beta=12)
    if mode == "threshold":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.convertScaleAbs(gray, alpha=1.55, beta=8)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    if mode == "adaptive":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -6)
    if mode == "denoise":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.fastNlMeansDenoising(gray, None, 12, 7, 21)
        gray = cv2.convertScaleAbs(gray, alpha=1.55, beta=10)
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -6)
    if mode == "simple":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.convertScaleAbs(gray, alpha=1.35, beta=10)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    upscaled = cv2.resize(image, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    if subtitle_style == "auto":
        subtitle_style = detect_subtitle_style(upscaled)

    if subtitle_style == "yellow":
        hsv = cv2.cvtColor(upscaled, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([15, 45, 120]), np.array([45, 255, 255]))
    else:
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 175, 255)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -8)
        mask = cv2.bitwise_or(mask, adaptive)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    sharp_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(mask, -1, sharp_kernel)


def preprocessing_variants(mode: str, preset: str) -> list[str]:
    if mode == "auto" or preset == "maximum_recall":
        return ["subtitle", "adaptive", "threshold", "denoise", "contrast", "grayscale", "simple"]
    return [mode]


def choose_variant_result(candidates: list[tuple[str, str, float, str, OcrEngineResult | None, OcrEngineResult | None, np.ndarray]]):
    return max(candidates, key=lambda item: (1 if item[1] else 0, item[2], score_ocr_candidate(item[1], item[2])))


def choose_best_preprocessed_image(
    image: np.ndarray,
    mode: str,
    preset: str,
    subtitle_style: str,
    lang: str,
    tesseract_psm: int,
    min_cyrillic_ratio: float,
    min_cyrillic_chars: int,
    timeout_seconds: float,
) -> np.ndarray:
    variants = preprocessing_variants(mode, preset)
    if len(variants) == 1:
        return preprocess_image(image, variants[0], subtitle_style)

    best_processed = preprocess_image(image, variants[0], subtitle_style)
    best_score = -1.0
    for variant in variants:
        processed = preprocess_image(image, variant, subtitle_style)
        try:
            result = run_with_timeout(lambda: read_tesseract(processed, lang, tesseract_psm), timeout_seconds)
        except Exception as exc:
            logging.debug("Preprocessing variant %s skipped: %s", variant, exc)
            continue
        cleaned, reason = clean_ocr_text_with_reason(result.text, min_cyrillic_ratio, min_cyrillic_chars, preset)
        if not cleaned:
            score = result.confidence * 10.0
        else:
            score = score_ocr_candidate(cleaned, result.confidence) + result.confidence * 25.0
        if score > best_score:
            best_score = score
            best_processed = processed
    return best_processed


def detect_subtitle_style(image: np.ndarray) -> str:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, np.array([15, 45, 120]), np.array([45, 255, 255]))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    white = cv2.inRange(gray, 190, 255)
    return "yellow" if int(yellow.sum()) > int(white.sum()) * 0.35 else "white"


def read_tesseract(processed: np.ndarray, lang: str, psm: int) -> OcrEngineResult:
    config = f"--psm {psm} -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(processed, lang=_tesseract_lang(lang), config=config, output_type=pytesseract.Output.DICT)
    words: list[str] = []
    confidences: list[float] = []
    for text, conf in zip(data.get("text", []), data.get("conf", [])):
        text = str(text).strip()
        try:
            conf_value = float(conf)
        except ValueError:
            conf_value = -1
        if text:
            words.append(text)
        if conf_value >= 0:
            confidences.append(conf_value / 100.0)
    return OcrEngineResult(text=" ".join(words), confidence=sum(confidences) / len(confidences) if confidences else 0.0)


_PADDLE_OCR = None
_PADDLE_FAILURE_LOGGED = False


def get_paddle_ocr():
    global _PADDLE_OCR
    if _PADDLE_OCR is None:
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:
            raise RuntimeError("PaddleOCR není dostupný. Nainstalujte paddleocr a paddlepaddle.") from exc
        _PADDLE_OCR = PaddleOCR(
            lang="ru",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
    return _PADDLE_OCR


def read_paddle(processed: np.ndarray) -> OcrEngineResult:
    paddle_img = ensure_bgr_for_paddle(processed)
    if paddle_img is None:
        return OcrEngineResult(text="", confidence=0.0, boxes=[])
    ocr = get_paddle_ocr()
    if hasattr(ocr, "predict"):
        result = ocr.predict(
            paddle_img,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
    else:
        result = ocr.ocr(paddle_img)
    return parse_paddle_result(result)


def _paddle_worker(request_queue, response_queue) -> None:
    try:
        get_paddle_ocr()
        response_queue.put(("ready", None))
    except Exception as exc:
        response_queue.put(("init_error", repr(exc)))
        return

    while True:
        command, payload = request_queue.get()
        if command == "stop":
            return
        if command != "predict":
            response_queue.put(("error", f"unknown command: {command}"))
            continue
        try:
            result = read_paddle(payload)
            response_queue.put(("result", (result.text, result.confidence, result.boxes)))
        except Exception as exc:
            response_queue.put(("error", repr(exc)))


class IsolatedPaddleOCR:
    def __init__(self) -> None:
        self.context = mp.get_context("spawn")
        self.request_queue = self.context.Queue(maxsize=1)
        self.response_queue = self.context.Queue(maxsize=1)
        self.process: mp.Process | None = None
        self.last_error = ""

    def start(self, timeout_seconds: float) -> bool:
        logging.info("Starting PaddleOCR in isolated process")
        self.process = self.context.Process(target=_paddle_worker, args=(self.request_queue, self.response_queue), daemon=True)
        self.process.start()
        try:
            status, payload = self.response_queue.get(timeout=timeout_seconds)
        except queue.Empty:
            self.last_error = "timeout"
            logging.error("PaddleOCR child process timed out, killing child only")
            self.close(kill=True)
            return False
        if status == "ready":
            return True
        self.last_error = str(payload)
        logging.error("PaddleOCR unavailable, continuing with Tesseract fallback: %s", payload)
        self.close(kill=True)
        return False

    def predict(self, image: np.ndarray, timeout_seconds: float) -> OcrEngineResult:
        if self.process is None or not self.process.is_alive():
            raise RuntimeError("PaddleOCR child process is not running")
        self.request_queue.put(("predict", image), timeout=timeout_seconds)
        try:
            status, payload = self.response_queue.get(timeout=timeout_seconds)
        except queue.Empty:
            logging.error("PaddleOCR child process timed out, killing child only")
            self.close(kill=True)
            raise TimeoutError(f"PaddleOCR child prediction timeout after {timeout_seconds:.1f}s")
        if status == "result":
            text, confidence, boxes = payload
            return OcrEngineResult(text=text, confidence=confidence, boxes=boxes)
        raise RuntimeError(str(payload))

    def close(self, kill: bool = False) -> None:
        process = self.process
        if process is None:
            return
        if process.is_alive() and not kill:
            try:
                self.request_queue.put(("stop", None), timeout=0.2)
            except Exception:
                pass
            process.join(timeout=2)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        self.process = None


def ensure_bgr_for_paddle(img: np.ndarray | None) -> np.ndarray | None:
    if img is None:
        return None
    if img.size == 0:
        return None
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif len(img.shape) != 3 or img.shape[2] != 3:
        return None
    return img


def parse_paddle_result(result) -> OcrEngineResult:
    texts: list[str] = []
    confidences: list[float] = []
    boxes: list = []

    if isinstance(result, list) and result and isinstance(result[0], dict):
        for page in result:
            page_texts = to_plain_sequence(get_first_non_empty(page, ["rec_texts", "texts"]))
            page_scores = to_plain_sequence(get_first_non_empty(page, ["rec_scores", "scores"]))
            page_boxes = to_plain_sequence(get_first_non_empty(page, ["rec_polys", "dt_polys", "rec_boxes", "boxes"]))
            if not page_texts:
                continue
            for idx, text in enumerate(page_texts):
                text = str(text).strip()
                if not text:
                    continue
                texts.append(text)
                if idx < len(page_scores):
                    try:
                        confidences.append(float(page_scores[idx]))
                    except (TypeError, ValueError):
                        pass
                if idx < len(page_boxes):
                    boxes.append(_box_to_list(page_boxes[idx]))
        return OcrEngineResult(text=" ".join(texts), confidence=sum(confidences) / len(confidences) if confidences else 0.0, boxes=boxes)

    entries = result[0] if result and isinstance(result, list) else []
    for entry in entries or []:
        if not entry or len(entry) < 2:
            continue
        box, rec = entry[0], entry[1]
        if not rec:
            continue
        text = str(rec[0]).strip()
        conf = float(rec[1]) if len(rec) > 1 else 0.0
        if text:
            texts.append(text)
            confidences.append(conf)
            boxes.append(_box_to_list(box))
    return OcrEngineResult(text=" ".join(texts), confidence=sum(confidences) / len(confidences) if confidences else 0.0, boxes=boxes)


def _box_to_list(box) -> list:
    if hasattr(box, "tolist"):
        return box.tolist()
    return box


def get_first_non_empty(page: dict, keys: list[str]):
    for key in keys:
        value = page.get(key)
        if value is None:
            continue
        if hasattr(value, "size"):
            if value.size > 0:
                return value
            continue
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return value
    return []


def to_plain_sequence(value) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def log_paddle_failure_once(exc: Exception, context: str) -> None:
    global _PADDLE_FAILURE_LOGGED
    if not _PADDLE_FAILURE_LOGGED:
        logging.exception("%s PaddleOCR se vypíná a pokračuji přes Tesseract fallback.", context)
        _PADDLE_FAILURE_LOGGED = True
    else:
        logging.debug("%s PaddleOCR stále selhává: %s", context, exc)

def run_with_timeout(func, timeout_seconds: float):
    result_queue: queue.Queue = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result_queue.put((True, func()))
        except Exception as exc:
            result_queue.put((False, exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        ok, value = result_queue.get(timeout=timeout_seconds)
    except queue.Empty:
        raise TimeoutError(f"OCR frame timeout after {timeout_seconds:.1f}s")
    if ok:
        return value
    raise value


def self_test_paddle() -> bool:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    if not self_test_parse_paddle_result():
        return False
    image = np.full((96, 480), 255, dtype=np.uint8)
    cv2.putText(image, "TEST", (24, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)
    _, image = cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)
    worker = IsolatedPaddleOCR()
    try:
        if not worker.start(60.0):
            logging.error("PaddleOCR self-test init failed in isolated process.")
            return False
        result = worker.predict(image, 30.0)
    except Exception:
        logging.exception("PaddleOCR self-test selhal.")
        return False
    finally:
        worker.close()
    print(f"PaddleOCR self-test OK. Text: {result.text!r}, confidence: {result.confidence:.3f}")
    logging.info("PaddleOCR self-test OK. Text: %r, confidence: %.3f", result.text, result.confidence)
    return True


def self_test_parse_paddle_result() -> bool:
    list_result = [{
        "rec_texts": ["Привет", "мир"],
        "rec_scores": [0.9, 0.8],
        "rec_polys": [[[0, 0], [10, 0], [10, 10], [0, 10]], [[20, 0], [30, 0], [30, 10], [20, 10]]],
    }]
    numpy_result = [{
        "rec_texts": np.array(["Радха", "Кришна"]),
        "rec_scores": np.array([0.95, 0.85]),
        "rec_polys": np.array([
            [[0, 0], [10, 0], [10, 10], [0, 10]],
            [[20, 0], [30, 0], [30, 10], [20, 10]],
        ]),
    }]
    empty_numpy_result = [{
        "rec_texts": np.array([]),
        "rec_scores": np.array([]),
        "rec_polys": np.array([]),
    }]

    parsed_list = parse_paddle_result(list_result)
    parsed_numpy = parse_paddle_result(numpy_result)
    parsed_empty = parse_paddle_result(empty_numpy_result)
    ok = (
        parsed_list.text == "Привет мир"
        and parsed_numpy.text == "Радха Кришна"
        and parsed_empty.text == ""
    )
    if not ok:
        print("PaddleOCR parser self-test selhal.")
    return ok


def choose_ocr_result(
    tesseract_result: OcrEngineResult | None,
    paddle_result: OcrEngineResult | None,
    min_cyrillic_ratio: float,
    min_cyrillic_chars: int,
    preset: str,
) -> tuple[str, str, float, str]:
    candidates: list[tuple[str, str, float, str]] = []
    for engine, result in (("tesseract", tesseract_result), ("paddle", paddle_result)):
        if result is None:
            continue
        cleaned, reason = clean_ocr_text_with_reason(result.text, min_cyrillic_ratio, min_cyrillic_chars, preset)
        if cleaned:
            candidates.append((engine, cleaned, score_ocr_candidate(cleaned, result.confidence), "accepted"))
        else:
            candidates.append((engine, "", 0.0, reason))
    accepted = [candidate for candidate in candidates if candidate[1]]
    if not accepted:
        reason = "; ".join(f"{engine}:{reason}" for engine, _, _, reason in candidates) or "empty"
        return "", "", 0.0, reason
    engine, text, score, _ = max(accepted, key=lambda item: item[2])
    confidence = 0.0
    if engine == "tesseract" and tesseract_result:
        confidence = tesseract_result.confidence
    if engine == "paddle" and paddle_result:
        confidence = paddle_result.confidence
    return engine, text, confidence, "accepted"


def needs_deep_ocr(text: str, decision: OcrNoiseDecision) -> bool:
    words = CYRILLIC_WORD_RE.findall(text)
    return len(words) < 2 or decision.meaningful_chars < 4 or decision.suspicious


def should_run_deep_ocr(text: str, decision: OcrNoiseDecision, deep_ocr: str) -> bool:
    if deep_ocr == "none":
        return False
    if deep_ocr == "short":
        return decision.is_valid_short and needs_deep_ocr(text, decision)
    if deep_ocr == "aggressive":
        return needs_deep_ocr(text, decision)
    raise ValueError("--deep-ocr musi byt none, short nebo aggressive.")


def _format_seconds_hhmmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _shift_image(image: np.ndarray, y_fraction: float) -> np.ndarray:
    height, width = image.shape[:2]
    shift = int(round(height * y_fraction))
    if shift == 0:
        return image
    matrix = np.float32([[1, 0, 0], [0, 1, shift]])
    return cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)


def _invert_image(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.bitwise_not(image)
    return cv2.bitwise_not(image)


def _deep_image_variants(raw_image: np.ndarray, preprocess_mode: str, preset: str, subtitle_style: str) -> list[tuple[str, np.ndarray]]:
    spatial_variants = [
        ("crop", raw_image),
        ("crop_up", _shift_image(raw_image, -0.06)),
        ("crop_down", _shift_image(raw_image, 0.06)),
    ]
    modes = ["subtitle", "grayscale", "adaptive", "contrast", "threshold", "denoise", "simple"]
    if preprocess_mode not in {"auto", "subtitle"} and preprocess_mode not in modes:
        modes.insert(0, preprocess_mode)
    variants: list[tuple[str, np.ndarray]] = []
    for spatial_name, image in spatial_variants:
        for mode in modes:
            processed = preprocess_image(image, mode, subtitle_style)
            variants.append((f"{spatial_name}:{mode}", processed))
        variants.append((f"{spatial_name}:invert", _invert_image(preprocess_image(image, "grayscale", subtitle_style))))
    return variants


def _neighbor_crops(crops: list[FrameCrop], current_index: int, window_seconds: float = 1.0, limit: int = 5) -> list[FrameCrop]:
    current = crops[current_index]
    candidates = [
        crop for crop in crops
        if crop.has_subtitle and crop.changed and abs(float(crop.time) - float(current.time)) <= window_seconds
    ]
    if current not in candidates:
        candidates.append(current)
    candidates.sort(key=lambda crop: (abs(float(crop.time) - float(current.time)), float(crop.time)))
    return candidates[:limit]


def _group_similar_votes(votes: list[dict]) -> list[dict]:
    groups: list[dict] = []
    for vote in votes:
        text = str(vote["text"])
        placed = False
        for group in groups:
            if similar(normalized_for_merge(text), normalized_for_merge(str(group["text"]))) >= 0.78:
                group["votes"].append(vote)
                if score_ocr_candidate(text, float(vote.get("confidence", 0.0))) > score_ocr_candidate(str(group["text"]), float(group.get("confidence", 0.0))):
                    group["text"] = text
                    group["confidence"] = float(vote.get("confidence", 0.0))
                placed = True
                break
        if not placed:
            groups.append({"text": text, "confidence": float(vote.get("confidence", 0.0)), "votes": [vote]})
    return groups


def _choose_deep_vote(votes: list[dict]) -> tuple[str, float, str]:
    if not votes:
        return "", 0.0, "deep_no_text"
    groups = _group_similar_votes(votes)
    best = max(
        groups,
        key=lambda group: (
            len(group["votes"]),
            max(float(vote.get("confidence", 0.0)) for vote in group["votes"]),
            max(score_ocr_candidate(str(vote["text"]), float(vote.get("confidence", 0.0))) for vote in group["votes"]),
            len(str(group["text"])),
        ),
    )
    text = str(best["text"])
    confidence = max(float(vote.get("confidence", 0.0)) for vote in best["votes"])
    decision = analyze_ocr_noise(text)
    if not is_acceptable_deep_recovery(text):
        return "", 0.0, "deep_rejected_language_validity:" + ",".join(decision.reasons or ["no_common_word"])
    if len(best["votes"]) >= 2:
        return text, confidence, "deep_vote"
    if decision.is_valid_short:
        return text, confidence, "deep_valid_short"
    if not decision.suspicious and decision.meaningful_chars >= 4:
        return text, confidence, "deep_single_valid"
    return "", 0.0, "deep_unconfirmed_noise:" + ",".join(decision.reasons or ["unknown"])


def deep_ocr_analysis(
    crops: list[FrameCrop],
    current_index: int,
    lang: str,
    preprocess_mode: str,
    preset: str,
    subtitle_style: str,
    tesseract_psm: int,
    min_cyrillic_ratio: float,
    min_cyrillic_chars: int,
    timeout_seconds: float,
) -> tuple[str, float, str]:
    current_time = float(crops[current_index].time)
    logging.info("Deep OCR analysis for short subtitle at %s", _format_seconds_hhmmss(current_time))
    votes: list[dict] = []
    per_variant_timeout = max(1.0, min(timeout_seconds, 8.0))
    for crop in _neighbor_crops(crops, current_index):
        raw_image = cv2.imread(str(crop.image_path))
        if raw_image is None:
            continue
        for variant_name, processed in _deep_image_variants(raw_image, preprocess_mode, preset, subtitle_style):
            try:
                result = run_with_timeout(lambda img=processed: read_tesseract(img, lang, tesseract_psm), per_variant_timeout)
            except Exception as exc:
                logging.debug("Deep OCR variant skipped at %.3f (%s): %s", crop.time, variant_name, exc)
                continue
            cleaned, reason = clean_ocr_text_with_reason(result.text, min_cyrillic_ratio, min_cyrillic_chars, preset)
            if not cleaned:
                continue
            decision = analyze_ocr_noise(cleaned)
            if decision.suspicious and not decision.is_valid_short:
                continue
            votes.append({
                "text": cleaned,
                "confidence": result.confidence,
                "time": crop.time,
                "variant": variant_name,
                "reason": reason,
            })
            if len(votes) >= 24:
                break
        if len(votes) >= 24:
            break
    return _choose_deep_vote(votes)


def _prepare_debug_dirs(temp_dir: Path) -> tuple[Path, Path, Path]:
    raw_dir = temp_dir / "debug_crops" / "raw"
    processed_dir = temp_dir / "debug_crops" / "processed"
    paddle_dir = temp_dir / "debug_crops" / "paddle_boxes"
    shutil.rmtree(raw_dir, ignore_errors=True)
    shutil.rmtree(processed_dir, ignore_errors=True)
    shutil.rmtree(paddle_dir, ignore_errors=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    paddle_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, processed_dir, paddle_dir


def draw_paddle_boxes(image: np.ndarray, boxes: list | None, output_path: Path) -> None:
    canvas = image.copy()
    for box in boxes or []:
        pts = np.array(box, dtype=np.int32)
        cv2.polylines(canvas, [pts], True, (0, 255, 0), 2)
    cv2.imwrite(str(output_path), canvas)


def _read_ocr_legacy(
    crops: list[FrameCrop],
    lang: str = "ru",
    similarity_threshold: float = 0.94,
    preprocess_mode: str = "subtitle",
    tesseract_psm: int = 6,
    min_cyrillic_ratio: float = 0.35,
    min_cyrillic_chars: int = 4,
    preset: str = "strict",
    ocr_engine: str = "paddle",
    deep_ocr: str = "none",
    subtitle_style: str = "auto",
    debug_crops: bool = False,
    temp_dir: Path | None = None,
    samples_path: Path | None = None,
    compare_path: Path | None = None,
    report_path: Path | None = None,
    max_ocr_frames: int | None = None,
    max_actual_ocr_calls: int | None = None,
    ocr_frame_timeout: float = 30.0,
    paddle_init_timeout: float = 60.0,
    disable_paddle: bool = False,
    allow_tesseract_fallback: bool = True,
    language_sanity_mode: str = "off",
    noise_filter_mode: str = "standard",
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
    extraction_stats: ExtractionStats | None = None,
) -> list[dict]:
    raise RuntimeError("Legacy OCR reader is disabled; use isolated read_ocr instead.")
    if ocr_engine not in {"tesseract", "paddle", "compare"}:
        raise ValueError("--ocr-engine musí být tesseract, paddle nebo compare.")
    if paddle_init_timeout <= 0:
        raise ValueError("--paddle-init-timeout musi byt vetsi nez 0.")
    if disable_paddle and ocr_engine in {"paddle", "compare"}:
        logging.warning("PaddleOCR je vypnuty, pokracuji pres Tesseract fallback.")
        ocr_engine = "tesseract"
    if max_ocr_frames is not None:
        if max_ocr_frames <= 0:
            raise ValueError("--max-ocr-frames musí být větší než 0.")
        limited_crops: list[FrameCrop] = []
        subtitle_frame_count = 0
        for crop in crops:
            if crop.has_subtitle:
                if subtitle_frame_count >= max_ocr_frames:
                    break
                subtitle_frame_count += 1
            limited_crops.append(crop)
        if len(limited_crops) < len(crops):
            logging.info(
                "OCR testovací limit: zpracovávám prvních %d subtitle framů z %d kandidátů.",
                subtitle_frame_count,
                sum(1 for crop in crops if crop.has_subtitle),
            )
            crops = limited_crops

    if max_actual_ocr_calls is not None:
        if max_actual_ocr_calls <= 0:
            raise ValueError("--max-actual-ocr-calls musi byt vetsi nez 0.")
        actual_candidates = [crop for crop in crops if crop.has_subtitle and crop.changed]
        if len(actual_candidates) > max_actual_ocr_calls:
            logging.warning(
                "Max actual OCR calls limit: zpracovavam prvnich %d z %d OCR kandidatu.",
                max_actual_ocr_calls,
                len(actual_candidates),
            )
            crops = actual_candidates[:max_actual_ocr_calls]

    actual_input_frames = sum(1 for crop in crops if crop.has_subtitle and crop.changed)
    logging.info("Spouštím OCR pro %d výřezů, engine: %s, preset: %s.", actual_input_frames, ocr_engine, preset)
    results: list[dict] = []
    raw_debug_dir = processed_debug_dir = paddle_debug_dir = None
    if debug_crops and temp_dir is not None:
        raw_debug_dir, processed_debug_dir, paddle_debug_dir = _prepare_debug_dirs(temp_dir)

    accepted_samples: list[str] = []
    dropped_samples: list[tuple[str, str]] = []
    compare_lines: list[str] = []
    confidences: list[float] = []
    engine_counts = {"tesseract": 0, "paddle": 0}
    last_text = ""
    last_confidence = 0.0
    paddle_disabled = False
    ocr_started_at = time.monotonic()
    total_progress_frames = sum(1 for crop in crops if crop.has_subtitle and crop.changed)
    processed_ocr_frames = 0
    actual_ocr_calls = 0
    accepted_ocr_calls = 0

    if ocr_engine in {"paddle", "compare"}:
        try:
            logging.info("Inicializuji PaddleOCR...")
            run_with_timeout(lambda: get_paddle_ocr(), paddle_init_timeout)
            logging.info("PaddleOCR inicializovan.")
        except TimeoutError:
            logging.error("PaddleOCR initialization timeout, switching to Tesseract fallback")
            logging.info("Pokračuji přes Tesseract fallback")
            paddle_disabled = True
        except Exception as exc:
            log_paddle_failure_once(exc, "Inicializace PaddleOCR selhala.")
            logging.info("Pokračuji přes Tesseract fallback")
            paddle_disabled = True

    for index, crop in enumerate(crops, start=1):
        if not crop.has_subtitle:
            last_text = ""
            last_confidence = 0.0
            continue

        if not crop.changed and last_text:
            results.append({
                "time": crop.time,
                "end": crop.end_time if crop.end_time is not None else crop.time,
                "text": last_text,
                "score": score_ocr_candidate(last_text, last_confidence),
                "confidence": last_confidence,
            })
            continue

        actual_ocr_calls += 1
        processed_ocr_frames += 1
        raw_image = cv2.imread(str(crop.image_path))
        if raw_image is None:
            _log_ocr_progress(processed_ocr_frames, total_progress_frames, ocr_started_at, interval=10 if ocr_engine in {"paddle", "compare"} else 100)
            continue
        processed = preprocess_image(raw_image, preprocess_mode, subtitle_style)
        tesseract_result = paddle_result = None
        if ocr_engine in {"tesseract", "compare"}:
            tesseract_result = run_with_timeout(lambda: read_tesseract(processed, lang, tesseract_psm), ocr_frame_timeout)
        if ocr_engine in {"paddle", "compare"} and not paddle_disabled:
            try:
                paddle_result = run_with_timeout(lambda: read_paddle(processed), ocr_frame_timeout)
            except TimeoutError:
                logging.error("PaddleOCR frame %.3f preskocen po timeoutu %.1f s.", crop.time, ocr_frame_timeout)
                _log_ocr_progress(processed_ocr_frames, total_progress_frames, ocr_started_at, interval=10)
                continue
            except Exception as exc:
                log_paddle_failure_once(exc, "PaddleOCR selhal při zpracování snímku.")
                paddle_disabled = True

        if ocr_engine == "paddle" and paddle_disabled and tesseract_result is None:
            tesseract_result = run_with_timeout(lambda: read_tesseract(processed, lang, tesseract_psm), ocr_frame_timeout)

        chosen_engine, text, confidence, reason = choose_ocr_result(
            tesseract_result,
            paddle_result,
            min_cyrillic_ratio,
            min_cyrillic_chars,
            preset,
        )

        if debug_crops and index <= 20 and raw_debug_dir and processed_debug_dir and paddle_debug_dir:
            shutil.copy2(crop.image_path, raw_debug_dir / crop.image_path.name)
            cv2.imwrite(str(processed_debug_dir / crop.image_path.name), processed)
            draw_paddle_boxes(processed, paddle_result.boxes if paddle_result else None, paddle_debug_dir / crop.image_path.name)

        if compare_path is not None and len(compare_lines) < 500:
            compare_lines.extend([
                f"time: {crop.time:.3f}",
                f"tesseract: {normalize_text(tesseract_result.text) if tesseract_result else ''}",
                f"paddle: {normalize_text(paddle_result.text) if paddle_result else ''}",
                f"selected: {text} ({chosen_engine or reason})",
                "",
            ])

        if not text:
            raw_for_sample = normalize_text((paddle_result.text if paddle_result else "") or (tesseract_result.text if tesseract_result else ""))
            if raw_for_sample and len(dropped_samples) < 100:
                dropped_samples.append((reason, raw_for_sample[:300]))
            _log_ocr_progress(processed_ocr_frames, total_progress_frames, ocr_started_at, interval=10 if ocr_engine in {"paddle", "compare"} else 100)
            continue

        engine_counts[chosen_engine] = engine_counts.get(chosen_engine, 0) + 1
        accepted_ocr_calls += 1
        confidences.append(confidence)
        results.append({
            "time": crop.time,
            "end": crop.end_time if crop.end_time is not None else crop.time,
            "text": text,
            "score": score_ocr_candidate(text, confidence),
            "confidence": confidence,
        })
        last_text = text
        last_confidence = confidence
        if len(accepted_samples) < 100:
            accepted_samples.append(text)
        _log_ocr_progress(processed_ocr_frames, total_progress_frames, ocr_started_at, interval=10 if ocr_engine in {"paddle", "compare"} else 100)

    dropped = actual_ocr_calls - accepted_ocr_calls
    logging.info("OCR našlo %d použitelných výsledků, zahozeno %d šumových výřezů.", len(results), dropped)
    logging.info(
        "OCR counters: candidates=%d accepted=%d rejected=%d successful_frames=%d timeout_frames=%d failed_frames=%d.",
        actual_ocr_calls,
        accepted_ocr_calls,
        dropped,
        successful_frames,
        timeout_frames,
        failed_frames,
    )
    if samples_path is not None:
        _write_ocr_samples(samples_path, accepted_samples, dropped_samples)
    if compare_path is not None:
        compare_path.write_text("\n".join(compare_lines), encoding="utf-8")
    if report_path is not None:
        _write_ocr_report(
            report_path,
            total_progress_frames,
            len(results),
            dropped,
            confidences,
            dropped_samples,
            engine_counts,
            crops,
            actual_ocr_calls,
            extraction_stats,
            timeout_frames=timeout_frames,
            failed_frames=failed_frames,
            successful_frames=successful_frames,
        )
    return results


def read_ocr(
    crops: list[FrameCrop],
    lang: str = "ru",
    similarity_threshold: float = 0.94,
    preprocess_mode: str = "subtitle",
    tesseract_psm: int = 6,
    min_cyrillic_ratio: float = 0.35,
    min_cyrillic_chars: int = 4,
    preset: str = "strict",
    ocr_engine: str = "paddle",
    deep_ocr: str = "none",
    subtitle_style: str = "auto",
    debug_crops: bool = False,
    temp_dir: Path | None = None,
    samples_path: Path | None = None,
    compare_path: Path | None = None,
    report_path: Path | None = None,
    max_ocr_frames: int | None = None,
    max_actual_ocr_calls: int | None = None,
    ocr_frame_timeout: float = 30.0,
    paddle_init_timeout: float = 60.0,
    disable_paddle: bool = False,
    allow_tesseract_fallback: bool = True,
    language_sanity_mode: str = "off",
    noise_filter_mode: str = "standard",
    subtitle_detect_mode: str = "hybrid",
    quality_profile: str = "default",
    ollama_model: str | None = None,
    ollama_base_url: str | None = None,
    extraction_stats: ExtractionStats | None = None,
) -> list[dict]:
    if ocr_engine not in {"tesseract", "paddle", "compare"}:
        raise ValueError("--ocr-engine musi byt tesseract, paddle nebo compare.")
    if deep_ocr not in {"none", "short", "aggressive"}:
        raise ValueError("--deep-ocr musi byt none, short nebo aggressive.")
    if language_sanity_mode not in {"off", "local", "ollama"}:
        raise ValueError("language_sanity_mode musi byt off, local nebo ollama.")
    if noise_filter_mode not in {"standard", "mild", "off"}:
        raise ValueError("noise_filter_mode musi byt standard, mild nebo off.")
    if subtitle_detect_mode not in {"diff", "white-text", "hybrid"}:
        raise ValueError("subtitle_detect_mode musi byt diff, white-text nebo hybrid.")
    if ocr_frame_timeout <= 0:
        raise ValueError("--ocr-frame-timeout musi byt vetsi nez 0.")
    if paddle_init_timeout <= 0:
        raise ValueError("--paddle-init-timeout musi byt vetsi nez 0.")
    if disable_paddle and ocr_engine in {"paddle", "compare"}:
        logging.warning("PaddleOCR je vypnuty, pokracuji pres Tesseract fallback.")
        ocr_engine = "tesseract"

    if max_ocr_frames is not None:
        if max_ocr_frames <= 0:
            raise ValueError("--max-ocr-frames musi byt vetsi nez 0.")
        crops = crops[:max_ocr_frames]
    if max_actual_ocr_calls is not None:
        if max_actual_ocr_calls <= 0:
            raise ValueError("--max-actual-ocr-calls musi byt vetsi nez 0.")
        actual_candidates = [crop for crop in crops if crop.has_subtitle and crop.changed]
        if len(actual_candidates) > max_actual_ocr_calls:
            logging.warning(
                "Max actual OCR calls limit: zpracovavam prvnich %d z %d OCR kandidatu.",
                max_actual_ocr_calls,
                len(actual_candidates),
            )
            crops = actual_candidates[:max_actual_ocr_calls]

    total_progress_frames = sum(1 for crop in crops if crop.has_subtitle and crop.changed)
    logging.info("Spoustim OCR pro %d vyrezu, engine: %s, preset: %s.", total_progress_frames, ocr_engine, preset)
    heuristic_seconds_per_candidate = 1.1 if ocr_engine in {"paddle", "compare"} else 0.45
    if deep_ocr != "none":
        heuristic_seconds_per_candidate += 0.25 if deep_ocr == "short" else 1.8
    estimated_ocr_minutes = (total_progress_frames * heuristic_seconds_per_candidate) / 60.0
    logging.info(
        "OCR candidates: %d, estimated OCR time: %.1f min, estimated_total_runtime_minutes: %.1f",
        total_progress_frames,
        estimated_ocr_minutes,
        estimated_ocr_minutes + 8.0,
    )

    raw_debug_dir = processed_debug_dir = paddle_debug_dir = None
    if debug_crops and temp_dir is not None:
        raw_debug_dir, processed_debug_dir, paddle_debug_dir = _prepare_debug_dirs(temp_dir)

    paddle_worker: IsolatedPaddleOCR | None = None
    paddle_disabled = False
    if ocr_engine in {"paddle", "compare"}:
        logging.info("Inicializuji PaddleOCR...")
        paddle_worker = IsolatedPaddleOCR()
        if paddle_worker.start(paddle_init_timeout):
            logging.info("PaddleOCR inicializovan.")
        else:
            if paddle_worker.last_error == "timeout":
                logging.error("PaddleOCR initialization timeout, switching to Tesseract fallback")
            logging.info("PaddleOCR unavailable, continuing with Tesseract fallback")
            logging.info("Pokračuji přes Tesseract fallback")
            paddle_worker = None
            paddle_disabled = True

    results: list[dict] = []
    accepted_samples: list[str] = []
    dropped_samples: list[tuple[str, str]] = []
    compare_lines: list[str] = []
    confidences: list[float] = []
    engine_counts = {"tesseract": 0, "paddle": 0}
    ocr_started_at = time.monotonic()
    actual_ocr_calls = 0
    accepted_ocr_calls = 0
    timeout_frames = 0
    failed_frames = 0
    successful_frames = 0
    empty_ocr_results = 0
    deep_stats = DeepOcrStats()
    rejection_reasons: Counter[str] = Counter()

    try:
        for index, crop in enumerate(crops, start=1):
            if not crop.has_subtitle:
                continue
            actual_ocr_calls += 1
            raw_image = cv2.imread(str(crop.image_path))
            if raw_image is None:
                _log_ocr_progress(actual_ocr_calls, total_progress_frames, ocr_started_at, interval=10 if ocr_engine in {"paddle", "compare"} else 100)
                continue
            processed = choose_best_preprocessed_image(
                raw_image,
                preprocess_mode,
                preset,
                subtitle_style,
                lang,
                tesseract_psm,
                min_cyrillic_ratio,
                min_cyrillic_chars,
                ocr_frame_timeout,
            )

            tesseract_result = None
            paddle_result = None
            if ocr_engine in {"tesseract", "compare"} or (ocr_engine == "paddle" and paddle_disabled):
                try:
                    tesseract_result = run_with_timeout(lambda: read_tesseract(processed, lang, tesseract_psm), ocr_frame_timeout)
                except TimeoutError as exc:
                    timeout_frames += 1
                    logging.warning("OCR frame %.3f skipped after timeout %.1f s: %s", crop.time, ocr_frame_timeout, exc)
                    _log_ocr_progress(actual_ocr_calls, total_progress_frames, ocr_started_at, interval=10 if ocr_engine in {"paddle", "compare"} else 100)
                    continue
                except Exception as exc:
                    failed_frames += 1
                    logging.warning("OCR frame %.3f failed and was skipped: %s", crop.time, exc)
                    _log_ocr_progress(actual_ocr_calls, total_progress_frames, ocr_started_at, interval=10 if ocr_engine in {"paddle", "compare"} else 100)
                    continue
            if ocr_engine in {"paddle", "compare"} and not paddle_disabled and paddle_worker is not None:
                try:
                    paddle_result = paddle_worker.predict(processed, ocr_frame_timeout)
                except TimeoutError as exc:
                    timeout_frames += 1
                    logging.warning("PaddleOCR frame %.3f skipped after timeout %.1f s: %s", crop.time, ocr_frame_timeout, exc)
                    _log_ocr_progress(actual_ocr_calls, total_progress_frames, ocr_started_at, interval=10)
                    continue
                except Exception as exc:
                    failed_frames += 1
                    logging.warning("PaddleOCR child failed on frame %.3f and was skipped/fallbacked: %s", crop.time, exc)
                    logging.info("PaddleOCR unavailable, continuing with Tesseract fallback")
                    logging.info("Pokračuji přes Tesseract fallback")
                    paddle_disabled = True
                    paddle_worker.close(kill=True)
                    paddle_worker = None
                    if tesseract_result is None:
                        try:
                            tesseract_result = run_with_timeout(lambda: read_tesseract(processed, lang, tesseract_psm), ocr_frame_timeout)
                        except TimeoutError as exc:
                            timeout_frames += 1
                            logging.warning("OCR fallback frame %.3f skipped after timeout %.1f s: %s", crop.time, ocr_frame_timeout, exc)
                            _log_ocr_progress(actual_ocr_calls, total_progress_frames, ocr_started_at, interval=10)
                            continue
                        except Exception as exc:
                            failed_frames += 1
                            logging.warning("OCR fallback frame %.3f failed and was skipped: %s", crop.time, exc)
                            _log_ocr_progress(actual_ocr_calls, total_progress_frames, ocr_started_at, interval=10)
                            continue

            chosen_engine, text, confidence, reason = choose_ocr_result(
                tesseract_result,
                paddle_result,
                min_cyrillic_ratio,
                min_cyrillic_chars,
                preset,
            )
            if text and noise_filter_mode == "off":
                noise_decision = OcrNoiseDecision(False, [], normalize_text(text), len(_meaningful_text(text)), _is_valid_short_subtitle(text))
            else:
                noise_decision = (
                    analyze_legacy_good_noise(text)
                    if text and noise_filter_mode == "mild"
                    else analyze_ocr_noise(text) if text else OcrNoiseDecision(True, [reason or "empty"], "", 0, False)
                )
            if text and should_run_deep_ocr(text, noise_decision, deep_ocr):
                deep_stats.attempted += 1
                deep_text, deep_confidence, deep_reason = deep_ocr_analysis(
                    crops,
                    index - 1,
                    lang,
                    preprocess_mode,
                    preset,
                    subtitle_style,
                    tesseract_psm,
                    min_cyrillic_ratio,
                    min_cyrillic_chars,
                    ocr_frame_timeout,
                )
                if deep_text:
                    if deep_text != text:
                        logging.info("Recovered short subtitle:\n%s", deep_text)
                    text = deep_text
                    confidence = max(confidence, deep_confidence)
                    reason = deep_reason
                    chosen_engine = chosen_engine or "tesseract"
                    deep_stats.recovered += 1
                    noise_decision = analyze_legacy_good_noise(text) if noise_filter_mode == "mild" else analyze_ocr_noise(text)
                elif noise_decision.suspicious and not noise_decision.is_valid_short:
                    deep_stats.rejected += 1
                    reason = deep_reason
                    logging.info("Rejected OCR noise:\n%s", text)
                    text = ""
            elif text and noise_decision.suspicious and not noise_decision.is_valid_short:
                if _count_cyrillic(text) > 20 and language_sanity_mode != "off":
                    sane, repaired_text, sanity_reason = language_sanity_check(
                        text,
                        ollama_model=ollama_model,
                        ollama_base_url=ollama_base_url,
                        use_ollama=language_sanity_mode == "ollama",
                    )
                    if sane:
                        logging.info("OCR language sanity accepted suspicious text (%s):\n%s", sanity_reason, repaired_text)
                        text = repaired_text
                        reason = sanity_reason
                        noise_decision = analyze_legacy_good_noise(text) if noise_filter_mode == "mild" else analyze_ocr_noise(text)
                    else:
                        reason = sanity_reason
                        logging.info("Rejected OCR noise:\n%s", repaired_text)
                        text = ""
                else:
                    reason = "noise_filter:" + ",".join(noise_decision.reasons or ["suspicious"])
                    logging.info("Rejected OCR noise:\n%s", text)
                    text = ""
            if not text and allow_tesseract_fallback and ocr_engine == "paddle" and tesseract_result is None:
                try:
                    tesseract_result = run_with_timeout(lambda: read_tesseract(processed, lang, tesseract_psm), ocr_frame_timeout)
                    chosen_engine, text, confidence, reason = choose_ocr_result(
                        tesseract_result,
                        paddle_result,
                        min_cyrillic_ratio,
                        min_cyrillic_chars,
                        preset,
                    )
                except TimeoutError as exc:
                    timeout_frames += 1
                    logging.warning("Tesseract fallback frame %.3f skipped after timeout %.1f s: %s", crop.time, ocr_frame_timeout, exc)
                except Exception as exc:
                    failed_frames += 1
                    logging.warning("Tesseract fallback frame %.3f failed and was skipped: %s", crop.time, exc)

            if debug_crops and index <= 20 and raw_debug_dir and processed_debug_dir and paddle_debug_dir:
                shutil.copy2(crop.image_path, raw_debug_dir / crop.image_path.name)
                cv2.imwrite(str(processed_debug_dir / crop.image_path.name), processed)
                draw_paddle_boxes(processed, paddle_result.boxes if paddle_result else None, paddle_debug_dir / crop.image_path.name)
            if compare_path is not None and len(compare_lines) < 500:
                compare_lines.extend([
                    f"time: {crop.time:.3f}",
                    f"tesseract: {normalize_text(tesseract_result.text) if tesseract_result else ''}",
                    f"paddle: {normalize_text(paddle_result.text) if paddle_result else ''}",
                    f"selected: {text} ({chosen_engine or reason})",
                    "",
                ])

            if not text:
                empty_ocr_results += 1
                for item in (reason or "empty").split(","):
                    rejection_reasons[item.strip() or "empty"] += 1
                raw_for_sample = normalize_text((paddle_result.text if paddle_result else "") or (tesseract_result.text if tesseract_result else ""))
                if raw_for_sample and len(dropped_samples) < 100:
                    dropped_samples.append((reason, raw_for_sample[:300]))
                _log_ocr_progress(actual_ocr_calls, total_progress_frames, ocr_started_at, interval=10 if ocr_engine in {"paddle", "compare"} else 100)
                continue

            engine_counts[chosen_engine] = engine_counts.get(chosen_engine, 0) + 1
            accepted_ocr_calls += 1
            successful_frames += 1
            confidences.append(confidence)
            raw_text = ""
            if chosen_engine == "paddle" and paddle_result is not None:
                raw_text = paddle_result.text
            elif tesseract_result is not None:
                raw_text = tesseract_result.text
            results.append({
                "time": crop.time,
                "end": crop.end_time if crop.end_time is not None else crop.time,
                "timestamp": crop.time,
                "raw_text": normalize_text(raw_text),
                "normalized_text": text,
                "text": text,
                "score": score_ocr_candidate(text, confidence),
                "confidence": confidence,
                "engine": chosen_engine,
                "frame_id": crop.image_path.stem,
                "image_path": str(crop.image_path),
                "crop": {
                    "text_bbox_y1": crop.text_bbox_y1,
                    "text_bbox_y2": crop.text_bbox_y2,
                    "text_bbox_area": crop.text_bbox_area,
                },
            })
            if len(accepted_samples) < 100:
                accepted_samples.append(text)
            _log_ocr_progress(actual_ocr_calls, total_progress_frames, ocr_started_at, interval=10 if ocr_engine in {"paddle", "compare"} else 100)
    finally:
        if paddle_worker is not None:
            paddle_worker.close()

    dropped = actual_ocr_calls - accepted_ocr_calls
    elapsed_ocr_seconds = max(0.001, time.monotonic() - ocr_started_at)
    average_seconds_per_ocr = elapsed_ocr_seconds / max(1, actual_ocr_calls)
    logging.info("OCR naslo %d pouzitelnych vysledku, zahozeno %d sumovych vyrezu.", len(results), dropped)
    logging.info("OCR noise filter: accepted=%d rejected=%d", accepted_ocr_calls, dropped)
    logging.info(
        "OCR timing: candidates=%d average_seconds_per_ocr=%.3f estimated_total_runtime_minutes=%.1f",
        actual_ocr_calls,
        average_seconds_per_ocr,
        (average_seconds_per_ocr * max(total_progress_frames, actual_ocr_calls)) / 60.0 + 8.0,
    )
    logging.info("Filtrovaný šum: %d OCR výřezů.", dropped)
    logging.info(
        "OCR counters: candidates=%d accepted=%d rejected=%d empty_results=%d successful_frames=%d timeout_frames=%d failed_frames=%d.",
        actual_ocr_calls,
        accepted_ocr_calls,
        dropped,
        empty_ocr_results,
        successful_frames,
        timeout_frames,
        failed_frames,
    )
    if samples_path is not None:
        _write_ocr_samples(samples_path, accepted_samples, dropped_samples)
    if compare_path is not None:
        compare_path.write_text("\n".join(compare_lines), encoding="utf-8")
    if report_path is not None:
        _write_ocr_report(
            report_path,
            total_progress_frames,
            len(results),
            dropped,
            confidences,
            dropped_samples,
            engine_counts,
            crops,
            actual_ocr_calls,
            extraction_stats,
            timeout_frames=timeout_frames,
            failed_frames=failed_frames,
            successful_frames=successful_frames,
            empty_ocr_results=empty_ocr_results,
            deep_stats=deep_stats,
            rejection_reasons=dict(rejection_reasons),
            quality_profile=quality_profile,
            subtitle_detect_mode=subtitle_detect_mode,
        )
    return results


def _log_ocr_progress(index: int, total: int, started_at: float, interval: int = 100) -> None:
    if total <= 0:
        return
    if index % interval != 0 and index != total:
        return
    elapsed = max(0.0, time.monotonic() - started_at)
    eta_seconds = (elapsed / index) * (total - index) if index > 0 else 0.0
    logging.info("OCR frame %d/%d, ETA %s.", index, total, _format_eta(eta_seconds))


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _write_ocr_samples(samples_path: Path, accepted: list[str], dropped: list[tuple[str, str]]) -> None:
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["ACCEPTED OCR SAMPLES", "=" * 40]
    lines.extend(f"{index}. {text}" for index, text in enumerate(accepted, start=1))
    lines.extend(["", "DROPPED OCR SAMPLES", "=" * 40])
    lines.extend(f"{index}. [{reason}] {text}" for index, (reason, text) in enumerate(dropped, start=1))
    samples_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def confidence_histogram(confidences: list[float]) -> dict[str, int]:
    bins = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
    for value in confidences:
        if value < 0.2:
            bins["0.0-0.2"] += 1
        elif value < 0.4:
            bins["0.2-0.4"] += 1
        elif value < 0.6:
            bins["0.4-0.6"] += 1
        elif value < 0.8:
            bins["0.6-0.8"] += 1
        else:
            bins["0.8-1.0"] += 1
    return bins


def _write_ocr_report(
    report_path: Path,
    total_frames: int,
    accepted: int,
    dropped: int,
    confidences: list[float],
    dropped_samples: list[tuple[str, str]],
    engine_counts: dict[str, int],
    crops: list[FrameCrop],
    actual_ocr_calls: int,
    extraction_stats: ExtractionStats | None = None,
    timeout_frames: int = 0,
    failed_frames: int = 0,
    successful_frames: int = 0,
    empty_ocr_results: int = 0,
    deep_stats: DeepOcrStats | None = None,
    rejection_reasons: dict[str, int] | None = None,
    quality_profile: str = "default",
    subtitle_detect_mode: str = "hybrid",
) -> None:
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    median_confidence = median(confidences) if confidences else 0.0
    histogram = confidence_histogram(confidences)
    white_text_frames = extraction_stats.white_text_frames if extraction_stats else sum(1 for crop in crops if crop.white_text_detected)
    skipped_same_subtitle_frames = extraction_stats.skipped_same_subtitle_frames if extraction_stats else sum(1 for crop in crops if crop.has_subtitle and not crop.changed)
    skipped_without_subtitles = extraction_stats.skipped_without_subtitles if extraction_stats else sum(1 for crop in crops if not crop.has_subtitle)
    sampled_frames_total = extraction_stats.sampled_frames_total if extraction_stats else len(crops)
    bbox_areas = [crop.text_bbox_area for crop in crops if crop.white_text_detected and crop.text_bbox_area > 0]
    average_bbox_area = sum(bbox_areas) / len(bbox_areas) if bbox_areas else 0.0
    deep_stats = deep_stats or DeepOcrStats()
    rejection_reasons = rejection_reasons or {}
    accepted_ratio = accepted / max(1, actual_ocr_calls)
    lines = [
        "OCR REPORT",
        "=" * 40,
        f"quality_profile: {quality_profile}",
        f"subtitle_detect_mode: {subtitle_detect_mode}",
        f"sampled_frames_total: {sampled_frames_total}",
        f"ocr_frames: {total_frames}",
        f"actual_ocr_calls: {actual_ocr_calls}",
        f"white_text_frames: {white_text_frames}",
        f"skipped_same_subtitle_frames: {skipped_same_subtitle_frames}",
        f"skipped_same_subtitle: {skipped_same_subtitle_frames}",
        f"white_text_detected_frames: {white_text_frames}",
        f"skipped_without_subtitles: {skipped_without_subtitles}",
        f"average_text_bbox_area: {average_bbox_area:.1f}",
        f"used_ocr_results: {accepted}",
        f"dropped_ocr_results: {dropped}",
        f"empty_ocr_results: {empty_ocr_results}",
        f"successful_frames: {successful_frames}",
        f"timeout_frames: {timeout_frames}",
        f"failed_frames: {failed_frames}",
        f"total_ocr_results: {actual_ocr_calls}",
        f"accepted_ocr_results: {accepted}",
        f"noise_rejected: {dropped}",
        f"accepted_ratio: {accepted_ratio:.3f}",
        f"accepted_ratio_warning: {'filter_too_aggressive' if accepted_ratio < 0.3 else ''}",
        f"deep_analysis_attempted: {deep_stats.attempted}",
        f"deep_analysis_recovered: {deep_stats.recovered}",
        f"deep_analysis_rejected: {deep_stats.rejected}",
        f"rejection_reasons: {rejection_reasons}",
        f"average_ocr_confidence: {avg_confidence:.3f}",
        f"median_ocr_confidence: {median_confidence:.3f}",
        f"confidence_histogram: {histogram}",
        f"engine_counts: {engine_counts}",
        "",
        "TOP OCR NOISE SAMPLES",
        "=" * 40,
    ]
    lines.extend(f"- [{reason}] {text}" for reason, text in dropped_samples[:25])
    lines.extend([
        "",
        "RECOMMENDATION",
        "=" * 40,
        "Pokud je titulků málo, zkuste --ocr-preset recall, širší crop nebo --subtitle-detect-mode white-text.",
        "Pokud je moc šumu, zkuste --subtitle-detect-mode hybrid, --ocr-preset balanced nebo strict a zkontrolujte temp/debug_detection.",
    ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if "--self-test-paddle" in sys.argv:
        raise SystemExit(0 if self_test_paddle() else 1)
    print("Použití: python -m src.ocr_reader --self-test-paddle")


if __name__ == "__main__":
    main()
