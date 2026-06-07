from __future__ import annotations

import argparse
from pathlib import Path

from src.config import (
    DEFAULT_AI_CLEANUP,
    DEFAULT_BOX_HEIGHT,
    DEFAULT_DEEP_OCR,
    DEFAULT_MAX_SUBTITLE_GAP,
    DEFAULT_MERGE_SIMILARITY,
    DEFAULT_MIN_CYRILLIC_CHARS,
    DEFAULT_MIN_CYRILLIC_RATIO,
    DEFAULT_OCR_ENGINE,
    DEFAULT_OCR_LANG,
    DEFAULT_OCR_PRESET,
    DEFAULT_OCR_PREPROCESS,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPEED_MODE,
    DEFAULT_SUBTITLE_AREA,
    DEFAULT_SUBTITLE_STYLE,
    DEFAULT_TARGET_LANG,
    DEFAULT_TESSERACT_PSM,
    DEFAULT_TRANSLATION_BATCH_SIZE,
)
from src.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vytvoří české titulky z ruských titulků v obraze a volitelně je vypálí do videa."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--url", help="URL videa ke stažení.")
    source.add_argument("--input", help="Lokální video soubor.")
    parser.add_argument("--burn", action="store_true", help="Vytvoří video s vypálenými českými titulky.")
    parser.add_argument("--only-srt", action="store_true", help="Vytvoří pouze .srt soubory.")
    parser.add_argument("--burn-existing-srt", help="Vypálí existující SRT bez OCR a překladu.")
    parser.add_argument("--cookies-browser", choices=["chrome", "edge"], help="Použít cookies z prohlížeče.")
    parser.add_argument("--ocr-lang", default=DEFAULT_OCR_LANG, help="Jazyk OCR, výchozí ru.")
    parser.add_argument("--target-lang", default=DEFAULT_TARGET_LANG, help="Cílový jazyk, výchozí cs.")
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE, help="Interval snímků v sekundách.")
    parser.add_argument("--subtitle-area", default=DEFAULT_SUBTITLE_AREA, help="Oblast titulků, např. bottom:25.")
    parser.add_argument("--box-height", type=float, default=DEFAULT_BOX_HEIGHT, help="Výška překryvného pruhu v procentech.")
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL, help="Model pro Ollama překlad/cleanup.")
    parser.add_argument("--translator", choices=["none", "ollama", "google", "auto"], default=None, help="Režim překladu.")
    parser.add_argument("--translation-batch-size", type=int, default=DEFAULT_TRANSLATION_BATCH_SIZE, help="Velikost dávky pro logování/cache překladu.")
    parser.add_argument("--ocr-preset", choices=["strict", "balanced", "recall", "maximum_recall"], default=DEFAULT_OCR_PRESET, help="Preset citlivosti OCR.")
    parser.add_argument("--ocr-engine", choices=["tesseract", "paddle", "compare"], default=DEFAULT_OCR_ENGINE, help="OCR engine.")
    parser.add_argument("--deep-ocr", choices=["none", "short", "aggressive"], default=DEFAULT_DEEP_OCR, help="Draha zachrana kratkych/podezrelych titulku.")
    parser.add_argument("--speed-mode", choices=["fast", "balanced", "quality"], default=DEFAULT_SPEED_MODE, help="Rychlostni profil OCR pipeline.")
    parser.add_argument("--quality-profile", choices=["default", "legacy_good", "old_perfect"], default="default", help="Prednastaveny profil kvality OCR.")
    parser.add_argument("--subtitle-style", choices=["white", "yellow", "auto"], default=DEFAULT_SUBTITLE_STYLE, help="Styl vypálených titulků.")
    parser.add_argument("--ai-cleanup", choices=["none", "ollama", "auto"], default=DEFAULT_AI_CLEANUP, help="Oprava OCR ruštiny před překladem.")
    parser.add_argument("--detect-subtitle-changes", action="store_true", help="OCR spustí jen při změně subtitle oblasti.")
    parser.add_argument("--subtitle-detect-mode", choices=["diff", "white-text", "hybrid"], default="hybrid", help="Režim detekce titulků před OCR.")
    parser.add_argument("--crop-x1", type=float, default=None, help="Levý okraj OCR cropu relativně k šířce videa.")
    parser.add_argument("--crop-x2", type=float, default=None, help="Pravý okraj OCR cropu relativně k šířce videa.")
    parser.add_argument("--crop-y1", type=float, default=None, help="Horní okraj OCR cropu relativně k výšce videa.")
    parser.add_argument("--crop-y2", type=float, default=None, help="Dolní okraj OCR cropu relativně k výšce videa.")
    parser.add_argument("--ocr-preprocess", choices=["auto", "grayscale", "contrast", "threshold", "adaptive", "denoise", "simple", "subtitle", "none"], default=DEFAULT_OCR_PREPROCESS, help="Režim předzpracování OCR obrazu.")
    parser.add_argument("--tesseract-psm", type=int, default=DEFAULT_TESSERACT_PSM, help="Tesseract page segmentation mode.")
    parser.add_argument("--min-cyrillic-ratio", type=float, default=DEFAULT_MIN_CYRILLIC_RATIO, help="Minimální poměr cyrilice v OCR textu.")
    parser.add_argument("--min-cyrillic-chars", type=int, default=DEFAULT_MIN_CYRILLIC_CHARS, help="Minimální počet znaků cyrilice.")
    parser.add_argument("--merge-similarity", type=float, default=DEFAULT_MERGE_SIMILARITY, help="Fuzzy podobnost pro sloučení OCR duplicit.")
    parser.add_argument("--max-subtitle-gap", type=float, default=DEFAULT_MAX_SUBTITLE_GAP, help="Maximální mezera pro sloučení podobných titulků.")
    parser.add_argument("--start-time", help="Počáteční čas videa pro rychlý OCR test ve formátu HH:MM:SS.")
    parser.add_argument("--end-time", help="Koncový čas videa pro rychlý OCR test ve formátu HH:MM:SS.")
    parser.add_argument("--max-ocr-frames", type=int, help="Zastaví OCR po N subtitle framech.")
    parser.add_argument("--debug-crops", action="store_true", help="Uloží raw, processed a Paddle box cropy.")
    parser.add_argument("--debug-subtitle-detection", action="store_true", help="Uloží debug obrázky detekce bílých titulků.")
    parser.add_argument("--defender-scan", action="store_true", help="Zkontroluje vstupní video přes Microsoft Defender.")
    parser.add_argument("--defender-strict", action="store_true", help="Ukončí program, pokud Defender scan selže.")
    parser.add_argument("--verbose", action="store_true", help="Podrobnější logování.")
    parser.add_argument("--test-ocr-engine", choices=["paddle"], help="Otestuje OCR engine a skončí bez zpracování videa.")
    parser.add_argument("--ocr-min-gap", type=float, default=2.0, help="Minimalni odstup mezi opakovanym OCR stejneho titulku v sekundach.")
    parser.add_argument("--subtitle-mask-similarity", type=float, default=0.90, help="Podobnost bile textove masky, nad kterou se titulek bere jako stejny.")
    parser.add_argument("--ocr-frame-timeout", type=float, default=30.0, help="Timeout pro jedno OCR volani v sekundach.")
    parser.add_argument("--max-actual-ocr-calls", type=int, help="Zpracuje nejvyse N skutecnych OCR kandidatu.")
    parser.add_argument("--paddle-init-timeout", type=float, default=60.0, help="Timeout inicializace PaddleOCR v sekundach pred Tesseract fallbackem.")
    parser.add_argument("--disable-paddle", action="store_true", help="Vypne PaddleOCR a vynuti Tesseract fallback.")
    parser.add_argument("--cleanup-temp", action="store_true", help="Po uspesnem vytvoreni videa smaze pomocne OCR cropy, source.mp4 a docasne fragmenty.")
    parser.add_argument("--benchmark", action="store_true", help="Spusti OCR benchmark pro usek zadany pres --start-time a --end-time.")
    return parser.parse_args()


def ask_continue_after_defender_failure() -> bool:
    answer = input("Microsoft Defender scan selhal nebo není dostupný. Chcete pokračovat? [y/N]: ").strip().lower()
    return answer in {"y", "yes", "a", "ano"}


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    if args.test_ocr_engine == "paddle":
        from src.ocr_reader import self_test_paddle

        raise SystemExit(0 if self_test_paddle() else 1)
    if not args.url and not args.input:
        raise SystemExit("Zadejte --url, --input nebo --test-ocr-engine paddle.")
    from src.processor import ProcessOptions, process_video

    options = ProcessOptions(
        url=args.url,
        input_path=Path(args.input) if args.input else None,
        burn=args.burn,
        only_srt=args.only_srt,
        cookies_browser=args.cookies_browser,
        ocr_lang=args.ocr_lang,
        target_lang=args.target_lang,
        sample_rate=args.sample_rate,
        subtitle_area=args.subtitle_area,
        box_height=args.box_height,
        ollama_model=args.ollama_model,
        translator=args.translator or "none",
        translation_batch_size=args.translation_batch_size,
        ocr_preset=args.ocr_preset,
        ocr_engine=args.ocr_engine,
        deep_ocr=args.deep_ocr,
        speed_mode=args.speed_mode,
        quality_profile=args.quality_profile,
        subtitle_style=args.subtitle_style,
        ai_cleanup=args.ai_cleanup,
        detect_subtitle_changes=args.detect_subtitle_changes,
        crop_x1=args.crop_x1,
        crop_x2=args.crop_x2,
        crop_y1=args.crop_y1,
        crop_y2=args.crop_y2,
        ocr_preprocess=args.ocr_preprocess,
        tesseract_psm=args.tesseract_psm,
        min_cyrillic_ratio=args.min_cyrillic_ratio,
        min_cyrillic_chars=args.min_cyrillic_chars,
        merge_similarity=args.merge_similarity,
        max_subtitle_gap=args.max_subtitle_gap,
        debug_crops=args.debug_crops,
        burn_existing_srt=Path(args.burn_existing_srt) if args.burn_existing_srt else None,
        defender_scan=args.defender_scan,
        defender_strict=args.defender_strict,
        start_time=args.start_time,
        end_time=args.end_time,
        max_ocr_frames=args.max_ocr_frames,
        subtitle_detect_mode=args.subtitle_detect_mode,
        debug_subtitle_detection=args.debug_subtitle_detection,
        ocr_min_gap=args.ocr_min_gap,
        subtitle_mask_similarity=args.subtitle_mask_similarity,
        ocr_frame_timeout=args.ocr_frame_timeout,
        max_actual_ocr_calls=args.max_actual_ocr_calls,
        paddle_init_timeout=args.paddle_init_timeout,
        disable_paddle=args.disable_paddle,
        cleanup_temp=args.cleanup_temp,
        benchmark=args.benchmark,
    )
    try:
        process_video(options, defender_continue_callback=ask_continue_after_defender_failure)
    except Exception as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
