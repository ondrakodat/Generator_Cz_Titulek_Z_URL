from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import urljoin

import requests

from src.ollama_utils import check_ollama, resolve_ollama_base_url
from src.utils import normalize_text


PROMPT = (
    "Oprav OCR chyby v ruskem filmovem titulku. Nevymyslej nove vety, nemen vyznam, "
    "zachovej rustinu. Pokud je text zjevny nesmyslny sum nebo watermark, vrat prazdny retezec. "
    "Vrat pouze opraveny titulek.\n\n{text}"
)

OBVIOUS_NOISE_PATTERNS = (
    "vk.com/starseriales",
    "starseriales",
    "перевод",
    "тайминг",
    "редактор",
    "маржан",
    "жамилова",
)


@dataclass
class RussianCleanup:
    mode: str = "auto"
    model: str = "qwen2.5:7b"
    base_url: str = ""
    timeout: int = 45
    _ollama_available: bool | None = None

    @property
    def resolved_base_url(self) -> str:
        return resolve_ollama_base_url(self.base_url or None)

    @property
    def generate_url(self) -> str:
        return urljoin(self.resolved_base_url.rstrip("/") + "/", "api/generate")

    def is_ollama_available(self) -> bool:
        if self._ollama_available is not None:
            return self._ollama_available
        self._ollama_available = check_ollama(self.resolved_base_url, self.model)
        return self._ollama_available

    def clean_block(self, text: str) -> str:
        ollama_available = self.mode in {"ollama", "auto"} and self.is_ollama_available()
        heuristic = heuristic_cleanup_ru(text, allow_drop=ollama_available)
        if self.mode == "none" or not heuristic:
            return heuristic
        if ollama_available:
            try:
                return self._clean_with_ollama(heuristic)
            except Exception as exc:
                logging.warning("AI cleanup pres Ollama selhal, pouzivam bezpecnou heuristiku: %s", exc)
                return heuristic
        return heuristic

    def _clean_with_ollama(self, text: str) -> str:
        payload = {
            "model": self.model,
            "prompt": PROMPT.format(text=text),
            "stream": False,
            "options": {"temperature": 0.1},
        }
        response = requests.post(self.generate_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        cleaned = response.json().get("response", "").strip().strip('"')
        return heuristic_cleanup_ru(cleaned, allow_drop=True)


def heuristic_cleanup_ru(text: str, allow_drop: bool = False) -> str:
    cleaned = normalize_text(text)
    cleaned = re.sub(r"([а-яё])([А-ЯЁ])", r"\1 \2", cleaned)
    cleaned = re.sub(r"\s+([,.!?;:])", r"\1", cleaned)
    cleaned = cleaned.strip(" .,-_|\\/`^*%$#;:!?\"'")
    if not cleaned:
        return ""
    lowered = cleaned.lower()
    if any(pattern in lowered for pattern in OBVIOUS_NOISE_PATTERNS):
        return ""
    if not allow_drop:
        return normalize_text(cleaned)

    tokens = cleaned.split()
    if len(tokens) >= 3:
        short_noise = sum(1 for token in tokens if len(token) <= 2 and token.lower() not in {"и", "в", "не", "но", "на", "я", "ты", "он"})
        if short_noise / len(tokens) > 0.70:
            return ""
    if re.search(r"(.)\1{5,}", cleaned):
        return ""
    return normalize_text(cleaned)


def cleanup_blocks(blocks: list[dict], cleaner: RussianCleanup) -> list[dict]:
    if cleaner.mode == "none":
        return blocks
    logging.info("Spoustim AI/heuristicke cisteni rustiny rezimem %s.", cleaner.mode)
    cleaned_blocks: list[dict] = []
    dropped = 0
    for block in blocks:
        cleaned = cleaner.clean_block(str(block["text"]))
        if not cleaned:
            dropped += 1
            continue
        cleaned_blocks.append({**block, "text": cleaned})
    logging.info("AI cleanup ponechal %d bloku, zahodil %d sumovych bloku.", len(cleaned_blocks), dropped)
    return cleaned_blocks
