from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin

import requests

from src.config import TRANSLATION_CACHE
from src.ollama_utils import check_ollama, resolve_ollama_base_url
from src.utils import normalize_text


PROMPT = (
    "Přelož následující ruský filmový titulek do přirozené češtiny. "
    "Zachovej význam, emoce a stručnost. Nevysvětluj, vrať pouze překlad.\n\n{text}"
)


@dataclass
class Translator:
    model: str = "qwen2.5:7b"
    mode: str = "auto"
    target_lang: str = "cs"
    ollama_base_url: str | None = None
    cache_path: Path = TRANSLATION_CACHE
    timeout: int = 120
    _ollama_available: bool | None = field(default=None, init=False)
    _cache: dict[str, str] = field(default_factory=dict, init=False)
    _google_warned: bool = field(default=False, init=False)
    _google_fallback_logged: bool = field(default=False, init=False)
    failed_blocks: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.mode = self.mode.lower()
        if self.mode not in {"none", "ollama", "google", "auto"}:
            raise ValueError("--translator musí být none, ollama, google nebo auto.")
        self._cache = self._load_cache()

    @property
    def base_url(self) -> str:
        return resolve_ollama_base_url(self.ollama_base_url)

    @property
    def ollama_url(self) -> str:
        return urljoin(self.base_url.rstrip("/") + "/", "api/generate")

    def _load_cache(self) -> dict[str, str]:
        if not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            logging.warning("Cache překladů nejde načíst, začínám s prázdnou cache: %s", exc)
            return {}

    def save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def is_ollama_available(self) -> bool:
        if self._ollama_available is not None:
            return self._ollama_available
        self._ollama_available = check_ollama(self.base_url, self.model)
        return self._ollama_available

    def translate_ru_to_cs(self, text: str) -> str:
        cleaned = clean_text_for_translation(text)
        if not cleaned:
            return text
        if self.mode == "none":
            return cleaned
        if cleaned in self._cache:
            return self._cache[cleaned]

        translated = cleaned
        if self.mode == "ollama":
            if not self.is_ollama_available():
                raise RuntimeError("Ollama není dostupná a --translator ollama je vyžadován.")
            translated = self._translate_with_ollama(cleaned)
        elif self.mode == "google":
            translated = self._translate_with_google(cleaned)
        elif self.mode == "auto":
            if self.is_ollama_available():
                try:
                    translated = self._translate_with_ollama(cleaned)
                except Exception as exc:
                    logging.warning("Ollama překlad selhal, zkouším Google fallback: %s", exc)
                    translated = self._translate_with_google(cleaned)
            else:
                if not self._google_fallback_logged:
                    logging.info("Používám Google fallback překlad.")
                    self._google_fallback_logged = True
                translated = self._translate_with_google(cleaned)

        self._cache[cleaned] = translated
        return translated

    def _translate_with_ollama(self, text: str) -> str:
        payload = {
            "model": self.model,
            "prompt": PROMPT.format(text=text),
            "stream": False,
            "options": {"temperature": 0.2},
        }
        response = requests.post(self.ollama_url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        translated = response.json().get("response", "").strip()
        if not translated:
            raise RuntimeError("Ollama vrátila prázdný překlad.")
        return translated.strip('"')

    def _translate_with_google(self, text: str) -> str:
        try:
            from deep_translator import GoogleTranslator

            return GoogleTranslator(source="ru", target=self.target_lang).translate(text)
        except Exception as exc:
            if not self._google_warned:
                logging.warning("Google překlad selhal, ponechávám původní text pro neúspěšné položky: %s", exc)
                self._google_warned = True
            return text


def clean_text_for_translation(text: str) -> str:
    cleaned = normalize_text(text).strip(" .,-_|\\/`^*%$#;:!?\"'")
    cleaned = re.sub(r"([а-яё])([А-ЯЁ])", r"\1 \2", cleaned)
    return normalize_text(cleaned)


def translate_blocks(blocks: list[dict], translator: Translator, batch_size: int = 20) -> list[dict]:
    if translator.mode == "none":
        logging.info("Překlad byl přeskočen, subtitles_cs.srt budou kopií context-fixed titulků.")
        return [{**block, "text": clean_text_for_translation(str(block["text"]))} for block in blocks]

    logging.info("Překládám %d titulků do češtiny režimem %s.", len(blocks), translator.mode)
    translated_blocks: list[dict] = []
    unique_seen = 0
    batch_size = max(1, batch_size)
    for index, block in enumerate(blocks, start=1):
        source_text = clean_text_for_translation(str(block["text"]))
        if source_text and source_text not in translator._cache:
            unique_seen += 1
        try:
            translated = translator.translate_ru_to_cs(source_text)
        except Exception as exc:
            translator.failed_blocks += 1
            logging.warning("Překlad bloku %d selhal, ponechávám původní text: %s", index, exc)
            translated = source_text
        translated_blocks.append({**block, "text": translated})
        if index % batch_size == 0:
            logging.info(
                "Překlad průběh: %d/%d titulků, nových unikátních textů zatím %d.",
                index,
                len(blocks),
                unique_seen,
            )
    translator.save_cache()
    logging.info("Cache překladů uložena: %s", translator.cache_path)
    return translated_blocks
