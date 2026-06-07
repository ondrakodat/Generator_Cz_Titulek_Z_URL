from __future__ import annotations

import logging
import ast
import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable

from src.config import (
    CS_SRT,
    CLEAN_RU_SRT,
    DEFAULT_AI_CLEANUP,
    DEFAULT_BOX_HEIGHT,
    DEFAULT_CROP_X1,
    DEFAULT_CROP_X2,
    DEFAULT_CROP_Y1,
    DEFAULT_CROP_Y2,
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
    INPUT_DIR,
    CONTEXT_CORRECTIONS,
    OCR_COMPARE,
    OCR_REPORT,
    OCR_REPORT_JSON,
    OCR_SAMPLES,
    OUTPUT_DIR,
    OUTPUT_VIDEO,
    RAW_RU_SRT,
    SOURCE_VIDEO,
    TEMP_DIR,
    TRANSLATION_REPORT,
)
from src.utils import check_required_tools, ensure_dirs, parse_hhmmss, parse_subtitle_area
from src.ollama_utils import check_ollama, resolve_ollama_base_url


@dataclass
class ProcessOptions:
    url: str | None = None
    input_path: Path | None = None
    burn: bool = False
    only_srt: bool = False
    cookies_browser: str | None = None
    ocr_lang: str = DEFAULT_OCR_LANG
    target_lang: str = DEFAULT_TARGET_LANG
    sample_rate: float = DEFAULT_SAMPLE_RATE
    subtitle_area: str = DEFAULT_SUBTITLE_AREA
    box_height: float = DEFAULT_BOX_HEIGHT
    ollama_model: str = DEFAULT_OLLAMA_MODEL
    translator: str = "none"
    ocr_preset: str = DEFAULT_OCR_PRESET
    ocr_engine: str = DEFAULT_OCR_ENGINE
    deep_ocr: str = DEFAULT_DEEP_OCR
    speed_mode: str = DEFAULT_SPEED_MODE
    quality_profile: str = "default"
    subtitle_style: str = DEFAULT_SUBTITLE_STYLE
    ai_cleanup: str = DEFAULT_AI_CLEANUP
    translation_batch_size: int = DEFAULT_TRANSLATION_BATCH_SIZE
    detect_subtitle_changes: bool = False
    crop_x1: float | None = None
    crop_x2: float | None = None
    crop_y1: float | None = None
    crop_y2: float | None = None
    ocr_preprocess: str = DEFAULT_OCR_PREPROCESS
    tesseract_psm: int = DEFAULT_TESSERACT_PSM
    min_cyrillic_ratio: float = DEFAULT_MIN_CYRILLIC_RATIO
    min_cyrillic_chars: int = DEFAULT_MIN_CYRILLIC_CHARS
    merge_similarity: float = DEFAULT_MERGE_SIMILARITY
    max_subtitle_gap: float = DEFAULT_MAX_SUBTITLE_GAP
    debug_crops: bool = False
    burn_existing_srt: Path | None = None
    defender_scan: bool = False
    defender_strict: bool = False
    start_time: str | None = None
    end_time: str | None = None
    max_ocr_frames: int | None = None
    max_actual_ocr_calls: int | None = None
    subtitle_detect_mode: str = "hybrid"
    debug_subtitle_detection: bool = False
    ocr_min_gap: float = 2.0
    subtitle_mask_similarity: float = 0.90
    ocr_frame_timeout: float = 30.0
    paddle_init_timeout: float = 60.0
    disable_paddle: bool = False
    cleanup_temp: bool = False
    benchmark: bool = False
    input_dir: Path = INPUT_DIR
    output_dir: Path = OUTPUT_DIR
    temp_dir: Path = TEMP_DIR
    source_video: Path = SOURCE_VIDEO
    raw_ru_srt: Path = RAW_RU_SRT
    clean_ru_srt: Path = CLEAN_RU_SRT
    cs_srt: Path = CS_SRT
    context_corrections: Path = CONTEXT_CORRECTIONS
    translation_report: Path = TRANSLATION_REPORT
    output_video: Path = OUTPUT_VIDEO
    ocr_samples: Path = OCR_SAMPLES
    ocr_compare: Path = OCR_COMPARE
    ocr_report: Path = OCR_REPORT
    ocr_report_json: Path = OCR_REPORT_JSON


@dataclass
class ProcessResult:
    raw_ru_srt: Path
    context_fixed_srt: Path
    cs_srt: Path
    ocr_report_json: Path
    translation_report: Path
    translation_status: str
    output_video: Path | None


def resolve_video(options: ProcessOptions) -> Path:
    if options.url:
        from src.downloader import download_video

        return download_video(options.url, options.source_video, options.cookies_browser)
    if options.input_path is None:
        raise ValueError("Musí být zadán vstupní soubor nebo URL.")
    input_path = options.input_path
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path
    if not input_path.exists():
        raise FileNotFoundError(f"Vstupní video neexistuje: {input_path}")
    logging.info("Používám lokální video: %s", input_path)
    options.source_video.parent.mkdir(parents=True, exist_ok=True)
    if input_path.resolve() != options.source_video.resolve():
        shutil.copy2(input_path, options.source_video)
        logging.info("Source video saved as: %s", options.source_video)
    return options.source_video


def handle_defender_scan(video_path: Path, strict: bool, should_continue: Callable[[], bool] | None = None) -> None:
    from src.defender_scan import scan_file_with_defender

    if scan_file_with_defender(video_path):
        return
    if strict:
        raise RuntimeError("Microsoft Defender scan selhal a strict režim je zapnutý. Ukončuji program.")
    if should_continue is not None and should_continue():
        logging.warning("Pokračuji i přes neúspěšnou Defender kontrolu.")
        return
    raise RuntimeError("Ukončeno po neúspěšné Defender kontrole.")


def user_crop_is_set(options: ProcessOptions) -> bool:
    return any(getattr(options, key) is not None for key in ("crop_x1", "crop_x2", "crop_y1", "crop_y2"))


def apply_quality_profile(options: ProcessOptions) -> None:
    if options.quality_profile not in {"default", "legacy_good", "old_perfect"}:
        raise ValueError("--quality-profile musi byt default, legacy_good nebo old_perfect.")
    if options.quality_profile == "old_perfect":
        options.ocr_engine = "paddle"
        options.subtitle_detect_mode = "white-text"
        options.ocr_preset = "recall"
        options.ocr_preprocess = "none"
        options.subtitle_style = "white"
        options.translator = options.translator or "none"
        options.ai_cleanup = "none"
        options.crop_x1 = 0.02
        options.crop_x2 = 0.98
        options.crop_y1 = 0.55
        options.crop_y2 = 0.98
        options.deep_ocr = "none"
        options.speed_mode = "quality"
        options.detect_subtitle_changes = False
        options.max_ocr_frames = None
        options.max_actual_ocr_calls = None
        options.sample_rate = 0.5
        logging.info("Quality profile old_perfect enabled: paddle/recall, white-text detection, no preprocess, white subtitles, historical full crop.")
        return
    if options.quality_profile != "legacy_good":
        return
    had_user_crop = user_crop_is_set(options)
    options.ocr_engine = "paddle"
    options.ocr_preset = "recall"
    options.deep_ocr = "none"
    options.subtitle_detect_mode = "hybrid"
    options.sample_rate = 0.5
    options.ocr_preprocess = "subtitle"
    options.speed_mode = "quality"
    if not had_user_crop:
        options.crop_x1 = 0.03
        options.crop_x2 = 0.97
        options.crop_y1 = 0.68
        options.crop_y2 = 0.96
    logging.info("Quality profile legacy_good enabled: paddle/recall, hybrid subtitle detection, mild noise filter, sample_rate=0.50.")


def apply_speed_profile(options: ProcessOptions) -> None:
    if options.speed_mode not in {"fast", "balanced", "quality"}:
        raise ValueError("--speed-mode musi byt fast, balanced nebo quality.")
    if options.deep_ocr not in {"none", "short", "aggressive"}:
        raise ValueError("--deep-ocr musi byt none, short nebo aggressive.")
    if options.speed_mode in {"fast", "balanced"}:
        if options.ocr_engine == "tesseract" and not options.disable_paddle:
            options.ocr_engine = "paddle"
        if options.ocr_preset == "maximum_recall":
            options.ocr_preset = "recall"
        if options.ocr_preprocess == "auto":
            options.ocr_preprocess = "subtitle"
        if options.speed_mode == "fast":
            options.deep_ocr = "none"
            options.detect_subtitle_changes = True
            options.subtitle_detect_mode = "white-text"
            options.sample_rate = max(options.sample_rate, 0.90)
        else:
            options.sample_rate = max(options.sample_rate, 0.75)
            if options.deep_ocr == "aggressive":
                options.deep_ocr = "short"
    elif options.speed_mode == "quality" and options.deep_ocr == "none" and options.quality_profile not in {"legacy_good", "old_perfect"}:
        options.deep_ocr = "short"
    logging.info(
        "Speed mode: %s, OCR engine: %s, preset: %s, deep OCR: %s, preprocess: %s, sample_rate: %.2f.",
        options.speed_mode,
        options.ocr_engine,
        options.ocr_preset,
        options.deep_ocr,
        options.ocr_preprocess,
        options.sample_rate,
    )


def auto_ocr_call_limit(duration_seconds: float, speed_mode: str) -> int | None:
    minutes = max(1.0, duration_seconds / 60.0)
    if speed_mode == "fast":
        return int(max(300, min(1600, minutes * 24)))
    if speed_mode == "balanced":
        return int(max(500, min(2200, minutes * 34)))
    return None


def process_video(options: ProcessOptions, defender_continue_callback: Callable[[], bool] | None = None) -> ProcessResult:
    ensure_dirs(options.input_dir, options.output_dir, options.temp_dir)
    if options.only_srt and options.burn:
        raise ValueError("Použijte buď --only-srt, nebo --burn, ne oboje najednou.")
    if options.target_lang != "cs":
        logging.warning("Tento MVP má připravený překladový prompt hlavně pro češtinu.")

    check_required_tools(require_ytdlp=bool(options.url))
    subtitle_area_ratio = parse_subtitle_area(options.subtitle_area)
    start_time = parse_hhmmss(options.start_time, "--start-time")
    end_time = parse_hhmmss(options.end_time, "--end-time")
    if options.max_ocr_frames is not None and options.max_ocr_frames <= 0:
        raise ValueError("--max-ocr-frames musí být větší než 0.")
    if options.max_actual_ocr_calls is not None and options.max_actual_ocr_calls <= 0:
        raise ValueError("--max-actual-ocr-calls musi byt vetsi nez 0.")
    if options.subtitle_detect_mode not in {"diff", "white-text", "hybrid"}:
        raise ValueError("--subtitle-detect-mode musí být diff, white-text nebo hybrid.")
    if options.ocr_min_gap < 0:
        raise ValueError("--ocr-min-gap nesmi byt zaporny.")
    if not 0 <= options.subtitle_mask_similarity <= 1:
        raise ValueError("--subtitle-mask-similarity musi byt v rozsahu 0.0 az 1.0.")
    if options.ocr_frame_timeout <= 0:
        raise ValueError("--ocr-frame-timeout musi byt vetsi nez 0.")
    if options.paddle_init_timeout <= 0:
        raise ValueError("--paddle-init-timeout musi byt vetsi nez 0.")
    if options.disable_paddle and options.ocr_engine in {"paddle", "compare"}:
        logging.info("PaddleOCR vypnut volbou --disable-paddle, pouzivam Tesseract.")
        options.ocr_engine = "tesseract"
    apply_quality_profile(options)
    apply_speed_profile(options)
    video_path = resolve_video(options)

    if options.defender_scan:
        handle_defender_scan(video_path, strict=options.defender_strict, should_continue=defender_continue_callback)

    if options.benchmark:
        return run_benchmark(video_path, options, subtitle_area_ratio, start_time, end_time)

    if options.burn_existing_srt is not None:
        from src.video_renderer import burn_subtitles

        srt_path = options.burn_existing_srt
        if not srt_path.is_absolute():
            srt_path = Path.cwd() / srt_path
        if not srt_path.exists():
            raise FileNotFoundError(f"Existující SRT nebylo nalezeno: {srt_path}")
        logging.info("Přeskakuji OCR i překlad, vypaluji existující SRT: %s", srt_path)
        output_video = burn_subtitles(video_path, srt_path, options.output_video, box_height_percent=options.box_height)
        if options.cleanup_temp and output_video.exists():
            cleanup_successful_job(options)
        return ProcessResult(
            raw_ru_srt=options.raw_ru_srt,
            context_fixed_srt=srt_path,
            cs_srt=srt_path,
            ocr_report_json=options.ocr_report_json,
            translation_report=options.translation_report,
            translation_status="not_required",
            output_video=output_video,
        )

    from src.context_correction import correct_hindu_context
    from src.frame_extractor import CropRegion, extract_subtitle_crops
    from src.subtitle_builder import build_subtitle_blocks, filter_obvious_ocr_noise_blocks, write_srt
    from src.translator import Translator, translate_blocks

    logging.info("Automatický režim kvality zapnut")
    if options.ocr_engine in {"paddle", "compare"} and options.disable_paddle:
        logging.info("PaddleOCR není povolen, používám bezpečný Tesseract.")
        options.ocr_engine = "tesseract"
    ollama_base_url = resolve_ollama_base_url()
    check_ollama(ollama_base_url, options.ollama_model)

    preset_values = apply_ocr_preset(options)
    video_info, crops, extraction_stats = extract_subtitle_crops(
        video_path=video_path,
        temp_dir=options.temp_dir,
        sample_rate=options.sample_rate,
        subtitle_area_ratio=subtitle_area_ratio,
        crop_region=CropRegion(
            preset_values["crop_x1"],
            preset_values["crop_x2"],
            preset_values["crop_y1"],
            preset_values["crop_y2"],
        ),
        detect_subtitle_changes=options.detect_subtitle_changes,
        start_time=start_time,
        end_time=end_time,
        subtitle_detect_mode=options.subtitle_detect_mode,
        debug_subtitle_detection=options.debug_subtitle_detection,
        ocr_min_gap=options.ocr_min_gap,
        subtitle_mask_similarity=options.subtitle_mask_similarity,
        use_text_bbox_crop=options.quality_profile != "old_perfect",
        legacy_white_text_detection=options.quality_profile == "old_perfect",
    )
    if options.max_actual_ocr_calls is None:
        options.max_actual_ocr_calls = auto_ocr_call_limit(video_info.duration, options.speed_mode)
        if options.max_actual_ocr_calls is not None:
            logging.info("Speed mode %s: auto max_actual_ocr_calls=%d.", options.speed_mode, options.max_actual_ocr_calls)
    preset_values, crops, extraction_stats = choose_best_automatic_ocr_settings(
        video_path=video_path,
        options=options,
        subtitle_area_ratio=subtitle_area_ratio,
        start_time=start_time,
        end_time=end_time,
        initial_preset_values=preset_values,
        initial_crops=crops,
        initial_stats=extraction_stats,
    )
    logging.info(
        "Vybraná OCR oblast: x %.2f-%.2f, y %.2f-%.2f",
        preset_values["crop_x1"],
        preset_values["crop_x2"],
        preset_values["crop_y1"],
        preset_values["crop_y2"],
    )
    logging.info("Vybraný OCR preset: %s", options.ocr_preset)
    actual_candidates = [crop for crop in crops if crop.has_subtitle and crop.changed]
    if options.max_actual_ocr_calls is not None and len(actual_candidates) > options.max_actual_ocr_calls:
        logging.warning(
            "Actual OCR calls %d prekrocil limit %d, zpracovavam jen prvnich %d vyrezu.",
            len(actual_candidates),
            options.max_actual_ocr_calls,
            options.max_actual_ocr_calls,
        )
        crops = actual_candidates[: options.max_actual_ocr_calls]
    subtitle_frame_count = sum(1 for crop in crops if crop.has_subtitle)
    ocr_frame_count = min(subtitle_frame_count, options.max_ocr_frames) if options.max_ocr_frames is not None else subtitle_frame_count
    ocr_results = read_ocr_for_options(crops, options, preset_values, extraction_stats)
    ru_blocks = build_subtitle_blocks(
        ocr_results,
        sample_rate=options.sample_rate,
        max_join_gap=options.max_subtitle_gap,
        similarity_threshold=preset_values["merge_similarity"],
    )
    ru_blocks = filter_obvious_ocr_noise_blocks(ru_blocks)
    logging.info("Po OCR bylo vytvořeno %d titulků.", len(ru_blocks))
    append_timing_to_report(options.ocr_report, ocr_frame_count, ru_blocks)
    log_ocr_detection_summary(options, extraction_stats, len(actual_candidates), len(ru_blocks))
    if options.quality_profile not in {"legacy_good", "old_perfect"} and options.speed_mode == "quality" and start_time is None and end_time is None and 50 * 60 <= video_info.duration <= 70 * 60 and len(ru_blocks) < 450:
        logging.warning("Quality guard triggered")
        logging.warning("Quality guard: too few subtitles, rerunning with maximum_recall")
        logging.warning("Automatic rerun started")
        preset_values, crops, extraction_stats = rerun_high_recall_crops(
            video_path=video_path,
            options=options,
            subtitle_area_ratio=subtitle_area_ratio,
            start_time=start_time,
            end_time=end_time,
        )
        subtitle_frame_count = sum(1 for crop in crops if crop.has_subtitle)
        ocr_frame_count = min(subtitle_frame_count, options.max_ocr_frames) if options.max_ocr_frames is not None else subtitle_frame_count
        ocr_results = read_ocr_for_options(crops, options, preset_values, extraction_stats)
        ru_blocks = build_subtitle_blocks(
            ocr_results,
            sample_rate=options.sample_rate,
            max_join_gap=options.max_subtitle_gap,
            similarity_threshold=preset_values["merge_similarity"],
        )
        ru_blocks = filter_obvious_ocr_noise_blocks(ru_blocks)
        logging.info("Po OCR recall fallback bylo vytvoreno %d titulku.", len(ru_blocks))
        append_timing_to_report(options.ocr_report, ocr_frame_count, ru_blocks)
        log_ocr_detection_summary(options, extraction_stats, sum(1 for crop in crops if crop.has_subtitle and crop.changed), len(ru_blocks))

    write_srt(ru_blocks, options.raw_ru_srt)
    if not options.raw_ru_srt.exists():
        raise RuntimeError("OCR failed: subtitles_original_raw.srt was not created.")

    context_fixed_blocks = correct_hindu_context(
        ru_blocks,
        output_path=options.clean_ru_srt,
        corrections_path=options.context_corrections,
    )
    write_srt(context_fixed_blocks, options.clean_ru_srt)
    if len(context_fixed_blocks) != len(ru_blocks):
        raise RuntimeError("Context correction changed subtitle block count.")
    if not options.clean_ru_srt.exists():
        raise RuntimeError("Context correction failed: subtitles_original_context_fixed.srt was not created.")

    translator = Translator(model=options.ollama_model, mode=options.translator, target_lang=options.target_lang, ollama_base_url=ollama_base_url)
    translation_status = "completed"
    try:
        if translator.mode == "ollama" and not translator.is_ollama_available():
            raise RuntimeError("Ollama is not available for required translation mode.")
        cs_blocks = translate_blocks(context_fixed_blocks, translator, batch_size=options.translation_batch_size)
        if len(cs_blocks) != len(context_fixed_blocks):
            raise RuntimeError("Translation changed subtitle block count.")
        if not subtitle_timings_match(context_fixed_blocks, cs_blocks):
            raise RuntimeError("Translation changed subtitle timing.")
        write_srt(cs_blocks, options.cs_srt)
        failed_blocks = int(getattr(translator, "failed_blocks", 0))
        if translator.mode == "none":
            translation_status = "not_requested"
        elif failed_blocks >= len(context_fixed_blocks) and context_fixed_blocks:
            raise RuntimeError("All subtitle blocks failed translation.")
        else:
            translation_status = "completed_with_errors" if failed_blocks else "completed"
        write_translation_report(
            options.translation_report,
            input_file=options.clean_ru_srt,
            output_file=options.cs_srt,
            total_blocks=len(context_fixed_blocks),
            translated_blocks=max(0, len(cs_blocks) - failed_blocks),
            failed_blocks=failed_blocks,
            engine=translator.mode,
            status=translation_status,
        )
    except Exception as exc:
        translation_status = "translation_failed"
        write_translation_report(
            options.translation_report,
            input_file=options.clean_ru_srt,
            output_file=options.cs_srt,
            total_blocks=len(context_fixed_blocks),
            translated_blocks=0,
            failed_blocks=len(context_fixed_blocks),
            engine=translator.mode,
            status=translation_status,
            error=str(exc),
        )
        logging.error("translation_failed: %s", exc)
        raise RuntimeError(f"translation_failed: {exc}") from exc
    if not options.cs_srt.exists():
        raise RuntimeError("Translation failed: subtitles_cs.srt was not created.")

    output_video = None
    if options.burn and not options.only_srt:
        from src.video_renderer import burn_subtitles

        output_video = burn_subtitles(video_path, options.cs_srt, options.output_video, box_height_percent=options.box_height)
        logging.info("Hotové video uloženo: %s", output_video)
        if options.cleanup_temp and output_video.exists():
            cleanup_successful_job(options)
    else:
        logging.info("Video se nevytváří, hotové jsou pouze SRT soubory.")

    write_ocr_json_report(options, extraction_stats, preset_values, ru_blocks)
    logging.info("Hotovo.")
    return ProcessResult(
        raw_ru_srt=options.raw_ru_srt,
        context_fixed_srt=options.clean_ru_srt,
        cs_srt=options.cs_srt,
        ocr_report_json=options.ocr_report_json,
        translation_report=options.translation_report,
        translation_status=translation_status,
        output_video=output_video,
    )


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def subtitle_timings_match(source_blocks: list[dict], translated_blocks: list[dict], tolerance: float = 0.001) -> bool:
    if len(source_blocks) != len(translated_blocks):
        return False
    for source, translated in zip(source_blocks, translated_blocks):
        if abs(float(source["start"]) - float(translated["start"])) > tolerance:
            return False
        if abs(float(source["end"]) - float(translated["end"])) > tolerance:
            return False
    return True


def write_translation_report(
    report_path: Path,
    *,
    input_file: Path,
    output_file: Path,
    total_blocks: int,
    translated_blocks: int,
    failed_blocks: int,
    engine: str,
    status: str,
    error: str | None = None,
) -> None:
    payload = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "total_blocks": total_blocks,
        "translated_blocks": translated_blocks,
        "failed_blocks": failed_blocks,
        "engine": engine,
        "status": status,
    }
    if error:
        payload["error"] = error
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_ocr_json_report(options: ProcessOptions, extraction_stats, preset_values: dict[str, float], blocks: list[dict]) -> None:
    confidences = [float(block.get("confidence", 0.0)) for block in blocks if block.get("confidence") is not None]
    payload = {
        "total_frames_sampled": int(getattr(extraction_stats, "sampled_frames_total", 0)),
        "frames_with_subtitles": int(getattr(extraction_stats, "white_text_frames", 0)),
        "white_text_frames": int(getattr(extraction_stats, "white_text_frames", 0)),
        "skipped_without_subtitles": int(getattr(extraction_stats, "skipped_without_subtitles", 0)),
        "skipped_same_subtitle": int(getattr(extraction_stats, "skipped_same_subtitle_frames", 0)),
        "skipped_same_subtitle_frames": int(getattr(extraction_stats, "skipped_same_subtitle_frames", 0)),
        "actual_ocr_calls": int(getattr(extraction_stats, "actual_ocr_calls", 0)),
        "successful_ocr_calls": parse_report_int(options.ocr_report, "successful_frames"),
        "failed_ocr_calls": parse_report_int(options.ocr_report, "failed_frames"),
        "timeout_ocr_calls": parse_report_int(options.ocr_report, "timeout_frames"),
        "total_ocr_results": parse_report_int(options.ocr_report, "total_ocr_results"),
        "accepted_ocr_results": parse_report_int(options.ocr_report, "accepted_ocr_results"),
        "noise_rejected": parse_report_int(options.ocr_report, "noise_rejected"),
        "quality_profile": options.quality_profile,
        "subtitle_detect_mode": options.subtitle_detect_mode,
        "average_text_bbox_area": parse_report_float(options.ocr_report, "average_text_bbox_area"),
        "accepted_ratio": parse_report_float(options.ocr_report, "accepted_ratio"),
        "accepted_ratio_warning": parse_report_value(options.ocr_report, "accepted_ratio_warning"),
        "deep_analysis_attempted": parse_report_int(options.ocr_report, "deep_analysis_attempted"),
        "deep_analysis_recovered": parse_report_int(options.ocr_report, "deep_analysis_recovered"),
        "deep_analysis_rejected": parse_report_int(options.ocr_report, "deep_analysis_rejected"),
        "rejection_reasons": parse_report_dict(options.ocr_report, "rejection_reasons"),
        "average_confidence": sum(confidences) / len(confidences) if confidences else parse_report_float(options.ocr_report, "average_ocr_confidence"),
        "median_confidence": median(confidences) if confidences else parse_report_float(options.ocr_report, "median_ocr_confidence"),
        "confidence_histogram": confidence_histogram(confidences),
        "used_crop": {
            "x1": preset_values["crop_x1"],
            "x2": preset_values["crop_x2"],
            "y1": preset_values["crop_y1"],
            "y2": preset_values["crop_y2"],
        },
        "auto_region_used": bool(preset_values.get("auto_region_used", False)),
        "subtitle_blocks_generated": len(blocks),
        "raw_srt": str(options.raw_ru_srt),
    }
    options.ocr_report_json.parent.mkdir(parents=True, exist_ok=True)
    options.ocr_report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_report_float(report_path: Path, key: str) -> float:
    if not report_path.exists():
        return 0.0
    pattern = re.compile(rf"^{re.escape(key)}:[ \t]*([0-9.]+)", re.MULTILINE)
    match = pattern.search(report_path.read_text(encoding="utf-8", errors="ignore"))
    return float(match.group(1)) if match else 0.0


def parse_report_value(report_path: Path, key: str) -> str:
    if not report_path.exists():
        return ""
    pattern = re.compile(rf"^{re.escape(key)}:[ \t]*(.*)$", re.MULTILINE)
    match = pattern.search(report_path.read_text(encoding="utf-8", errors="ignore"))
    return match.group(1).strip() if match else ""


def parse_report_dict(report_path: Path, key: str) -> dict:
    if not report_path.exists():
        return {}
    pattern = re.compile(rf"^{re.escape(key)}:[ \t]*(\{{.*\}})", re.MULTILINE)
    match = pattern.search(report_path.read_text(encoding="utf-8", errors="ignore"))
    if not match:
        return {}
    try:
        parsed = ast.literal_eval(match.group(1))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def confidence_histogram(confidences: list[float]) -> dict[str, int]:
    buckets = {
        "0.00-0.50": 0,
        "0.50-0.70": 0,
        "0.70-0.85": 0,
        "0.85-0.95": 0,
        "0.95-1.00": 0,
    }
    for confidence in confidences:
        if confidence < 0.50:
            buckets["0.00-0.50"] += 1
        elif confidence < 0.70:
            buckets["0.50-0.70"] += 1
        elif confidence < 0.85:
            buckets["0.70-0.85"] += 1
        elif confidence < 0.95:
            buckets["0.85-0.95"] += 1
        else:
            buckets["0.95-1.00"] += 1
    return buckets


OCR_CACHE_VERSION = 5


def language_sanity_mode_for_options(options: ProcessOptions) -> str:
    if options.quality_profile == "old_perfect":
        return "off"
    return "off" if options.speed_mode == "fast" else "ollama"


def noise_filter_mode_for_options(options: ProcessOptions) -> str:
    if options.quality_profile == "old_perfect":
        return "off"
    return "mild" if options.quality_profile == "legacy_good" else "standard"


def ocr_candidate_signature(crops: list) -> dict:
    candidates = [crop for crop in crops if getattr(crop, "has_subtitle", False) and getattr(crop, "changed", True)]
    first_time = float(candidates[0].time) if candidates else None
    last_time = float(candidates[-1].time) if candidates else None
    digest_source = [
        (
            round(float(crop.time), 3),
            round(float(getattr(crop, "end_time", crop.time) or crop.time), 3),
            Path(str(crop.image_path)).name,
            round(float(getattr(crop, "text_bbox_y1", 0.0)), 4),
            round(float(getattr(crop, "text_bbox_y2", 0.0)), 4),
        )
        for crop in candidates
    ]
    digest = hashlib.sha1(json.dumps(digest_source, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return {
        "candidate_count": len(candidates),
        "first_candidate_time": first_time,
        "last_candidate_time": last_time,
        "candidate_digest": digest,
    }


def ocr_cache_metadata(options: ProcessOptions, preset_values: dict[str, float], crops: list) -> dict:
    video_path = options.source_video if options.source_video.exists() else options.input_path
    video_hash = hash_file(Path(video_path)) if video_path is not None and Path(video_path).exists() else ""
    candidate_meta = ocr_candidate_signature(crops)
    return {
        "ocr_cache_version": OCR_CACHE_VERSION,
        "quality_profile": options.quality_profile,
        "input_video_hash": video_hash,
        "start_time": options.start_time,
        "end_time": options.end_time,
        "sample_rate": options.sample_rate,
        "crop": [preset_values["crop_x1"], preset_values["crop_x2"], preset_values["crop_y1"], preset_values["crop_y2"]],
        "ocr_engine": options.ocr_engine,
        "ocr_preset": options.ocr_preset,
        "ocr_preprocess": options.ocr_preprocess,
        "subtitle_detect_mode": options.subtitle_detect_mode,
        "speed_mode": options.speed_mode,
        "deep_ocr": options.deep_ocr,
        "language_sanity_mode": language_sanity_mode_for_options(options),
        "noise_filter_mode": noise_filter_mode_for_options(options),
        "max_ocr_frames": options.max_ocr_frames,
        "max_actual_ocr_calls": options.max_actual_ocr_calls,
        "ocr_min_gap": options.ocr_min_gap,
        "subtitle_mask_similarity": options.subtitle_mask_similarity,
        "detect_subtitle_changes": options.detect_subtitle_changes,
        **candidate_meta,
    }


def read_ocr_for_options(crops: list, options: ProcessOptions, preset_values: dict[str, float], extraction_stats) -> list[dict]:
    from src.ocr_reader import read_ocr

    cache_metadata = ocr_cache_metadata(options, preset_values, crops)
    cache_path = ocr_cache_path(options, preset_values, crops)
    if cache_path is not None and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("metadata") == cache_metadata and isinstance(cached.get("results"), list):
                logging.info(
                    "OCR cache hit: %s range=%s-%s sample_rate=%.2f crop=%s engine=%s preset=%s speed=%s deep=%s sanity=%s candidates=%s digest=%s",
                    cache_path,
                    cache_metadata["start_time"] or "FULL",
                    cache_metadata["end_time"] or "END",
                    cache_metadata["sample_rate"],
                    cache_metadata["crop"],
                    cache_metadata["ocr_engine"],
                    cache_metadata["ocr_preset"],
                    cache_metadata["speed_mode"],
                    cache_metadata["deep_ocr"],
                    cache_metadata["language_sanity_mode"],
                    cache_metadata["candidate_count"],
                    cache_metadata["candidate_digest"],
                )
                return cached["results"]
            logging.info("OCR cache metadata mismatch or legacy cache ignored: %s", cache_path)
        except Exception as exc:
            logging.warning("OCR cache read failed, running OCR again: %s", exc)

    language_sanity_mode = language_sanity_mode_for_options(options)
    noise_filter_mode = noise_filter_mode_for_options(options)
    ocr_filter_preset = "legacy_good" if noise_filter_mode == "mild" else options.ocr_preset
    results = read_ocr(
        crops,
        lang=options.ocr_lang,
        preprocess_mode=options.ocr_preprocess,
        tesseract_psm=options.tesseract_psm,
        min_cyrillic_ratio=preset_values["min_cyrillic_ratio"],
        min_cyrillic_chars=int(preset_values["min_cyrillic_chars"]),
        preset=ocr_filter_preset,
        ocr_engine=options.ocr_engine,
        deep_ocr=options.deep_ocr,
        subtitle_style=options.subtitle_style,
        debug_crops=options.debug_crops,
        temp_dir=options.temp_dir,
        samples_path=options.ocr_samples,
        compare_path=options.ocr_compare if options.ocr_engine == "compare" else None,
        report_path=options.ocr_report,
        max_ocr_frames=options.max_ocr_frames,
        max_actual_ocr_calls=options.max_actual_ocr_calls,
        ocr_frame_timeout=options.ocr_frame_timeout,
        paddle_init_timeout=options.paddle_init_timeout,
        disable_paddle=options.disable_paddle,
        allow_tesseract_fallback=options.speed_mode == "quality",
        language_sanity_mode=language_sanity_mode,
        noise_filter_mode=noise_filter_mode,
        subtitle_detect_mode=options.subtitle_detect_mode,
        quality_profile=options.quality_profile,
        ollama_model=options.ollama_model,
        ollama_base_url=resolve_ollama_base_url(),
        extraction_stats=extraction_stats,
    )
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"metadata": cache_metadata, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
            logging.info(
                "OCR cache saved: %s range=%s-%s candidates=%s digest=%s",
                cache_path,
                cache_metadata["start_time"] or "FULL",
                cache_metadata["end_time"] or "END",
                cache_metadata["candidate_count"],
                cache_metadata["candidate_digest"],
            )
        except Exception as exc:
            logging.warning("OCR cache save failed: %s", exc)
    return results


def run_benchmark(
    video_path: Path,
    options: ProcessOptions,
    subtitle_area_ratio: float,
    start_time: float | None,
    end_time: float | None,
) -> ProcessResult:
    from src.frame_extractor import CropRegion, extract_subtitle_crops
    from src.ocr_reader import read_ocr
    from src.subtitle_builder import build_subtitle_blocks, write_srt

    if start_time is None or end_time is None:
        raise ValueError("Benchmark mode requires --start-time and --end-time.")

    benchmark_dir = options.output_dir
    benchmark_temp_dir = options.temp_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark_temp_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        ("maximum_recall", {"crop_x1": DEFAULT_CROP_X1, "crop_x2": DEFAULT_CROP_X2, "crop_y1": DEFAULT_CROP_Y1, "crop_y2": DEFAULT_CROP_Y2}),
        ("recall", {"crop_x1": DEFAULT_CROP_X1, "crop_x2": DEFAULT_CROP_X2, "crop_y1": DEFAULT_CROP_Y1, "crop_y2": DEFAULT_CROP_Y2}),
        ("balanced", {"crop_x1": DEFAULT_CROP_X1, "crop_x2": DEFAULT_CROP_X2, "crop_y1": DEFAULT_CROP_Y1, "crop_y2": DEFAULT_CROP_Y2}),
        ("strict", {"crop_x1": DEFAULT_CROP_X1, "crop_x2": DEFAULT_CROP_X2, "crop_y1": DEFAULT_CROP_Y1, "crop_y2": DEFAULT_CROP_Y2}),
    ]
    report: dict[str, dict] = {}
    for preset, crop_values in configs:
        started = time.monotonic()
        options.ocr_preset = preset
        preset_values = values_for_validation_preset(options, crop_values, preset)
        _, crops, extraction_stats = extract_subtitle_crops(
            video_path=video_path,
            temp_dir=benchmark_temp_dir / f"temp_{preset}",
            sample_rate=options.sample_rate,
            subtitle_area_ratio=subtitle_area_ratio,
            crop_region=CropRegion(crop_values["crop_x1"], crop_values["crop_x2"], crop_values["crop_y1"], crop_values["crop_y2"]),
            detect_subtitle_changes=options.detect_subtitle_changes,
            start_time=start_time,
            end_time=end_time,
            subtitle_detect_mode=options.subtitle_detect_mode,
            debug_subtitle_detection=False,
            ocr_min_gap=options.ocr_min_gap,
            subtitle_mask_similarity=options.subtitle_mask_similarity,
        )
        ocr_report_path = benchmark_dir / f"{preset}_ocr_report.txt"
        ocr_results = read_ocr(
            crops,
            lang=options.ocr_lang,
            preprocess_mode=options.ocr_preprocess,
            tesseract_psm=options.tesseract_psm,
            min_cyrillic_ratio=preset_values["min_cyrillic_ratio"],
            min_cyrillic_chars=int(preset_values["min_cyrillic_chars"]),
            preset=preset,
            ocr_engine="tesseract",
            subtitle_style=options.subtitle_style,
            debug_crops=False,
            temp_dir=benchmark_temp_dir / f"temp_{preset}",
            samples_path=benchmark_dir / f"{preset}_samples.txt",
            compare_path=None,
            report_path=ocr_report_path,
            max_ocr_frames=options.max_ocr_frames,
            max_actual_ocr_calls=options.max_actual_ocr_calls,
            ocr_frame_timeout=options.ocr_frame_timeout,
            paddle_init_timeout=options.paddle_init_timeout,
            disable_paddle=True,
            extraction_stats=extraction_stats,
        )
        blocks = build_subtitle_blocks(
            ocr_results,
            sample_rate=options.sample_rate,
            max_join_gap=options.max_subtitle_gap,
            similarity_threshold=preset_values["merge_similarity"],
        )
        write_srt(blocks, benchmark_dir / f"{preset}.srt")
        elapsed = max(0.001, time.monotonic() - started)
        confidences = [float(item.get("confidence", 0.0)) for item in ocr_results]
        report[preset] = {
            "ocr_candidates": sum(1 for crop in crops if crop.has_subtitle and crop.changed),
            "ocr_hits": len(ocr_results),
            "subtitle_blocks": len(blocks),
            "average_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
            "median_confidence": median(confidences) if confidences else 0.0,
            "timeout_frames": parse_report_int(ocr_report_path, "timeout_frames"),
            "failed_frames": parse_report_int(ocr_report_path, "failed_frames"),
            "successful_frames": parse_report_int(ocr_report_path, "successful_frames"),
            "processing_seconds": elapsed,
            "processing_speed_candidates_per_second": len(crops) / elapsed,
        }
    json_path = benchmark_dir / "benchmark_report.json"
    html_path = benchmark_dir / "benchmark_report.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_benchmark_html(report, html_path)
    logging.info("Benchmark report generated: %s", json_path)
    logging.info("Benchmark report generated: %s", html_path)
    return ProcessResult(
        raw_ru_srt=benchmark_dir / "maximum_recall.srt",
        context_fixed_srt=benchmark_dir / "maximum_recall.srt",
        cs_srt=benchmark_dir / "balanced.srt",
        ocr_report_json=benchmark_dir / "benchmark_report.json",
        translation_report=benchmark_dir / "translation_report.json",
        translation_status="not_required",
        output_video=None,
    )


def parse_report_int(report_path: Path, key: str) -> int:
    if not report_path.exists():
        return 0
    pattern = re.compile(rf"^{re.escape(key)}:\s*(\d+)", re.MULTILINE)
    match = pattern.search(report_path.read_text(encoding="utf-8", errors="ignore"))
    return int(match.group(1)) if match else 0


def write_benchmark_html(report: dict[str, dict], html_path: Path) -> None:
    rows = []
    for preset, data in report.items():
        rows.append(
            "<tr>"
            f"<td>{preset}</td>"
            f"<td>{data['ocr_candidates']}</td>"
            f"<td>{data['ocr_hits']}</td>"
            f"<td>{data['subtitle_blocks']}</td>"
            f"<td>{data['average_confidence']:.3f}</td>"
            f"<td>{data['timeout_frames']}</td>"
            f"<td>{data['failed_frames']}</td>"
            f"<td>{data['processing_speed_candidates_per_second']:.2f}</td>"
            "</tr>"
        )
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>OCR benchmark</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}</style>"
        "<h1>OCR benchmark</h1><table><thead><tr><th>Preset</th><th>OCR candidates</th><th>OCR hits</th><th>Subtitle blocks</th><th>Average confidence</th><th>Timeout frames</th><th>Failed frames</th><th>Candidates/s</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>",
        encoding="utf-8",
    )


def ocr_cache_path(options: ProcessOptions, preset_values: dict[str, float], crops: list | None = None) -> Path | None:
    video_path = options.source_video if options.source_video.exists() else options.input_path
    if video_path is None or not Path(video_path).exists():
        return None
    metadata = ocr_cache_metadata(options, preset_values, crops or [])
    settings_digest = hashlib.sha1(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    video_digest = str(metadata.get("input_video_hash", ""))[:16] or "no_video_hash"
    return options.output_dir / "ocr_cache" / f"{video_digest}_{settings_digest}.json"


def hash_file(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def rerun_high_recall_crops(
    video_path: Path,
    options: ProcessOptions,
    subtitle_area_ratio: float,
    start_time: float | None,
    end_time: float | None,
) -> tuple[dict[str, float], list, object]:
    from src.frame_extractor import CropRegion, extract_subtitle_crops

    options.ocr_preset = "maximum_recall"
    values = apply_ocr_preset(options)
    fallback_values = {
        "min_cyrillic_ratio": 0.08,
        "min_cyrillic_chars": 1.0,
        "merge_similarity": 0.54,
    }
    if not user_crop_is_set(options):
        fallback_values.update({
            "crop_x1": 0.02,
            "crop_x2": 0.98,
            "crop_y1": 0.70,
            "crop_y2": 0.98,
        })
    values.update(fallback_values)
    _, crops, stats = extract_subtitle_crops(
        video_path=video_path,
        temp_dir=options.temp_dir,
        sample_rate=options.sample_rate,
        subtitle_area_ratio=subtitle_area_ratio,
        crop_region=CropRegion(values["crop_x1"], values["crop_x2"], values["crop_y1"], values["crop_y2"]),
        detect_subtitle_changes=options.detect_subtitle_changes,
        start_time=start_time,
        end_time=end_time,
        subtitle_detect_mode=options.subtitle_detect_mode,
        debug_subtitle_detection=options.debug_subtitle_detection,
        ocr_min_gap=options.ocr_min_gap,
        subtitle_mask_similarity=options.subtitle_mask_similarity,
    )
    logging.info(
        "Vybraná OCR oblast: x %.2f-%.2f, y %.2f-%.2f",
        values["crop_x1"],
        values["crop_x2"],
        values["crop_y1"],
        values["crop_y2"],
    )
    logging.info("Vybraný OCR preset: %s", options.ocr_preset)
    return values, crops, stats


def cleanup_successful_job(options: ProcessOptions) -> None:
    root = options.output_dir.resolve()
    logging.info("Cleanup: deleting temporary OCR crops")
    temp_dir = options.temp_dir.resolve()
    project_temp = TEMP_DIR.resolve()
    if is_within(temp_dir, root) or temp_dir == project_temp or is_within(temp_dir, project_temp):
        shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        logging.info("Cleanup: temp dir is outside current job directory, skipping: %s", temp_dir)

    logging.info("Cleanup: deleting source video")
    source_video = options.source_video.resolve()
    input_root = options.input_dir.resolve()
    if is_within(source_video, root) or source_video == (input_root / "source.mp4").resolve():
        source_video.unlink(missing_ok=True)
    else:
        logging.info("Cleanup: source video is outside current job directory, skipping: %s", source_video)

    if not options.debug_crops:
        options.ocr_samples.unlink(missing_ok=True)
        options.ocr_compare.unlink(missing_ok=True)
    for pattern in ("*.webm", "*.part", "*.temp", "*.tmp"):
        for path in root.rglob(pattern):
            if path.is_file() and is_within(path, root):
                path.unlink(missing_ok=True)
    logging.info("Cleanup complete")


def quality_settings_cache_path(video_path: Path, options: ProcessOptions) -> Path:
    try:
        stat = video_path.stat()
        identity = f"{video_path.resolve()}:{stat.st_size}:{int(stat.st_mtime)}"
    except OSError:
        identity = str(video_path)
    settings = {
        "identity": identity,
        "speed_mode": options.speed_mode,
        "preset": options.ocr_preset,
        "sample_rate": options.sample_rate,
        "detect": options.subtitle_detect_mode,
        "crop": [options.crop_x1, options.crop_x2, options.crop_y1, options.crop_y2],
    }
    digest = hashlib.sha1(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return options.output_dir / "ocr_cache" / f"quality_settings_{digest}.json"


def choose_balanced_automatic_ocr_settings(
    video_path: Path,
    options: ProcessOptions,
    subtitle_area_ratio: float,
    start_time: float | None,
    end_time: float | None,
    initial_preset_values: dict[str, float],
    initial_crops: list,
    initial_stats,
) -> tuple[dict[str, float], list, object]:
    from src.frame_extractor import CropRegion, extract_subtitle_crops

    cache_path = quality_settings_cache_path(video_path, options)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            values = cached.get("values") if isinstance(cached, dict) else None
            if isinstance(values, dict):
                values = {**initial_preset_values, **values}
                if values.get("crop_x1") == initial_preset_values.get("crop_x1") and values.get("crop_y1") == initial_preset_values.get("crop_y1"):
                    logging.info("Balanced quality settings cache hit: %s", cache_path)
                    return values, initial_crops, initial_stats
        except Exception as exc:
            logging.warning("Balanced quality settings cache read failed: %s", exc)

    detected_region = detect_common_subtitle_region(initial_crops)
    if detected_region is None:
        logging.info("Speed mode balanced: OCR-heavy validation skipped, keeping default crop.")
        return initial_preset_values, initial_crops, initial_stats

    validation_end = max((crop.end_time or crop.time for crop in initial_crops[:120]), default=start_time or 0.0) + options.sample_rate
    if end_time is not None:
        validation_end = min(validation_end, end_time)
    candidate_values = dict(initial_preset_values)
    candidate_values.update({
        "crop_x1": DEFAULT_CROP_X1,
        "crop_x2": DEFAULT_CROP_X2,
        "crop_y1": detected_region[0],
        "crop_y2": detected_region[1],
        "auto_region_used": True,
    })
    _, candidate_crops, candidate_stats = extract_subtitle_crops(
        video_path=video_path,
        temp_dir=options.temp_dir / "auto_validation" / "balanced_auto_region",
        sample_rate=options.sample_rate,
        subtitle_area_ratio=subtitle_area_ratio,
        crop_region=CropRegion(candidate_values["crop_x1"], candidate_values["crop_x2"], candidate_values["crop_y1"], candidate_values["crop_y2"]),
        detect_subtitle_changes=options.detect_subtitle_changes,
        start_time=start_time,
        end_time=validation_end,
        subtitle_detect_mode=options.subtitle_detect_mode,
        debug_subtitle_detection=False,
        ocr_min_gap=options.ocr_min_gap,
        subtitle_mask_similarity=options.subtitle_mask_similarity,
    )
    initial_hits = sum(1 for crop in initial_crops[:120] if crop.has_subtitle)
    candidate_hits = sum(1 for crop in candidate_crops if crop.has_subtitle)
    use_candidate = candidate_hits >= max(4, int(initial_hits * 0.65))
    if use_candidate:
        logging.info("Speed mode balanced: using cached/detected subtitle region y %.2f-%.2f.", candidate_values["crop_y1"], candidate_values["crop_y2"])
        _, full_crops, full_stats = extract_subtitle_crops(
            video_path=video_path,
            temp_dir=options.temp_dir,
            sample_rate=options.sample_rate,
            subtitle_area_ratio=subtitle_area_ratio,
            crop_region=CropRegion(candidate_values["crop_x1"], candidate_values["crop_x2"], candidate_values["crop_y1"], candidate_values["crop_y2"]),
            detect_subtitle_changes=options.detect_subtitle_changes,
            start_time=start_time,
            end_time=end_time,
            subtitle_detect_mode=options.subtitle_detect_mode,
            debug_subtitle_detection=options.debug_subtitle_detection,
            ocr_min_gap=options.ocr_min_gap,
            subtitle_mask_similarity=options.subtitle_mask_similarity,
        )
        selected_values, selected_crops, selected_stats = candidate_values, full_crops, full_stats
    else:
        logging.info("Speed mode balanced: detected region looked weak, keeping default crop.")
        selected_values, selected_crops, selected_stats = initial_preset_values, initial_crops, initial_stats
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"values": selected_values}, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Balanced quality settings cache saved: %s", cache_path)
    except Exception as exc:
        logging.warning("Balanced quality settings cache save failed: %s", exc)
    return selected_values, selected_crops, selected_stats


def choose_best_automatic_ocr_settings(
    video_path: Path,
    options: ProcessOptions,
    subtitle_area_ratio: float,
    start_time: float | None,
    end_time: float | None,
    initial_preset_values: dict[str, float],
    initial_crops: list,
    initial_stats,
) -> tuple[dict[str, float], list, object]:
    from src.frame_extractor import CropRegion, extract_subtitle_crops
    from src.ocr_reader import read_ocr

    if options.quality_profile == "old_perfect":
        logging.info("Quality profile old_perfect: preskakuji automatickou validaci cropu/presetu.")
        return initial_preset_values, initial_crops, initial_stats
    if user_crop_is_set(options):
        logging.info("Automaticka validace cropu: uzivatelsky crop je nastaven, neprepisuji ho.")
        return initial_preset_values, initial_crops, initial_stats
    if options.speed_mode == "fast":
        logging.info("Speed mode fast: preskakuji OCR-heavy automatic quality validation.")
        return initial_preset_values, initial_crops, initial_stats
    if options.speed_mode == "balanced":
        return choose_balanced_automatic_ocr_settings(
            video_path,
            options,
            subtitle_area_ratio,
            start_time,
            end_time,
            initial_preset_values,
            initial_crops,
            initial_stats,
        )

    sample_limit = 100
    sample_crops = [crop for crop in initial_crops if crop.has_subtitle and crop.changed][:sample_limit]
    if len(sample_crops) < 4:
        logging.info("Automatická validace cropu: málo prvních subtitle framů, ponechávám výchozí oblast.")
        return initial_preset_values, initial_crops, initial_stats

    validation_end = max(crop.end_time or crop.time for crop in sample_crops) + options.sample_rate
    if end_time is not None:
        validation_end = min(validation_end, end_time)
    crop_candidates = [
        ("maximum", "maximum_recall", DEFAULT_CROP_X1, DEFAULT_CROP_X2, DEFAULT_CROP_Y1, DEFAULT_CROP_Y2, initial_crops, initial_stats),
        ("A", "recall", DEFAULT_CROP_X1, DEFAULT_CROP_X2, DEFAULT_CROP_Y1, DEFAULT_CROP_Y2, None, None),
        ("B", "balanced", DEFAULT_CROP_X1, DEFAULT_CROP_X2, DEFAULT_CROP_Y1, DEFAULT_CROP_Y2, None, None),
        ("C", "strict", DEFAULT_CROP_X1, DEFAULT_CROP_X2, DEFAULT_CROP_Y1, DEFAULT_CROP_Y2, None, None),
    ]
    detected_region = detect_common_subtitle_region(initial_crops)
    if detected_region is not None:
        crop_candidates.append(("auto-region", "recall", 0.03, 0.97, detected_region[0], detected_region[1], None, None))
        logging.info("Automatic subtitle region candidate: y %.2f-%.2f", detected_region[0], detected_region[1])
    evaluations: list[dict] = []
    validation_root = options.temp_dir / "auto_validation"
    shutil.rmtree(validation_root, ignore_errors=True)
    validation_root.mkdir(parents=True, exist_ok=True)

    try:
        for label, preset, crop_x1, crop_x2, crop_y1, crop_y2, candidate_crops, candidate_stats in crop_candidates:
            values = dict(initial_preset_values)
            values.update({
                "crop_x1": crop_x1,
                "crop_x2": crop_x2,
                "crop_y1": crop_y1,
                "crop_y2": crop_y2,
            })
            if candidate_crops is None:
                _, candidate_crops, candidate_stats = extract_subtitle_crops(
                    video_path=video_path,
                    temp_dir=validation_root / label,
                    sample_rate=options.sample_rate,
                    subtitle_area_ratio=subtitle_area_ratio,
                    crop_region=CropRegion(values["crop_x1"], values["crop_x2"], values["crop_y1"], values["crop_y2"]),
                    detect_subtitle_changes=options.detect_subtitle_changes,
                    start_time=start_time,
                    end_time=validation_end,
                    subtitle_detect_mode=options.subtitle_detect_mode,
                    debug_subtitle_detection=False,
                    ocr_min_gap=options.ocr_min_gap,
                    subtitle_mask_similarity=options.subtitle_mask_similarity,
                )
            candidate_sample = [crop for crop in candidate_crops if crop.has_subtitle and crop.changed][:sample_limit]
            preset_values = values_for_validation_preset(options, values, preset)
            results = read_ocr(
                candidate_sample,
                lang=options.ocr_lang,
                preprocess_mode=options.ocr_preprocess,
                tesseract_psm=options.tesseract_psm,
                min_cyrillic_ratio=preset_values["min_cyrillic_ratio"],
                min_cyrillic_chars=int(preset_values["min_cyrillic_chars"]),
                preset=preset,
                ocr_engine="tesseract",
                subtitle_style=options.subtitle_style,
                debug_crops=False,
                temp_dir=validation_root / label,
                samples_path=None,
                compare_path=None,
                report_path=None,
                max_ocr_frames=sample_limit,
                max_actual_ocr_calls=sample_limit,
                ocr_frame_timeout=options.ocr_frame_timeout,
                paddle_init_timeout=options.paddle_init_timeout,
                disable_paddle=True,
                extraction_stats=candidate_stats,
            )
            accepted = len(results)
            avg_score = sum(float(result.get("score", 0.0)) for result in results) / accepted if accepted else 0.0
            avg_confidence = sum(float(result.get("confidence", 0.0)) for result in results) / accepted if accepted else 0.0
            dropped = max(0, len(candidate_sample) - accepted)
            noise_ratio = dropped / max(1, len(candidate_sample))
            evaluations.append({
                "label": label,
                "preset": preset,
                "values": preset_values,
                "score": accepted * 10000.0 + avg_confidence * 100.0 + avg_score - noise_ratio,
                "accepted": accepted,
                "dropped": dropped,
                "avg_confidence": avg_confidence,
                "noise_ratio": noise_ratio,
                "crops": candidate_crops,
                "stats": candidate_stats,
            })
        best = max(evaluations, key=lambda item: (item["accepted"], item["avg_confidence"], -item["noise_ratio"], item["score"]))
        logging.info("OCR quality candidates: %s", [
            {
                "label": item["label"],
                "preset": item["preset"],
                "accepted": item["accepted"],
                "avg_confidence": round(item["avg_confidence"], 3),
                "noise_ratio": round(item["noise_ratio"], 3),
            }
            for item in evaluations
        ])
        logging.info(
            "Filtrovaný šum: automatická validace zahodila %d/%d vzorků u vítězného nastavení.",
            best["dropped"],
            best["accepted"] + best["dropped"],
        )
        options.ocr_preset = best["preset"]
        best_values = best["values"]
        best_values["auto_region_used"] = best["label"] == "auto-region"
        if best_values["auto_region_used"]:
            logging.info("Subtitle region detection: using auto region.")
        else:
            logging.info("Subtitle region detection: using default/candidate region.")
        if best["label"] == "maximum":
            return best_values, initial_crops, initial_stats
        _, full_crops, full_stats = extract_subtitle_crops(
            video_path=video_path,
            temp_dir=options.temp_dir,
            sample_rate=options.sample_rate,
            subtitle_area_ratio=subtitle_area_ratio,
            crop_region=CropRegion(
                best_values["crop_x1"],
                best_values["crop_x2"],
                best_values["crop_y1"],
                best_values["crop_y2"],
            ),
            detect_subtitle_changes=options.detect_subtitle_changes,
            start_time=start_time,
            end_time=end_time,
            subtitle_detect_mode=options.subtitle_detect_mode,
            debug_subtitle_detection=options.debug_subtitle_detection,
            ocr_min_gap=options.ocr_min_gap,
            subtitle_mask_similarity=options.subtitle_mask_similarity,
        )
        return best_values, full_crops, full_stats
    finally:
        if not options.debug_crops and not options.debug_subtitle_detection:
            shutil.rmtree(validation_root, ignore_errors=True)


def values_for_validation_preset(options: ProcessOptions, base_values: dict[str, float], preset: str) -> dict[str, float]:
    values = dict(base_values)
    if preset == "strict":
        values.update({"min_cyrillic_ratio": 0.35, "min_cyrillic_chars": 4.0, "merge_similarity": 0.72})
    elif preset == "balanced":
        values.update({"min_cyrillic_ratio": 0.25, "min_cyrillic_chars": 3.0, "merge_similarity": 0.68})
    elif preset == "recall":
        values.update({"min_cyrillic_ratio": 0.12, "min_cyrillic_chars": 2.0, "merge_similarity": 0.58})
    elif preset == "maximum_recall":
        values.update({"min_cyrillic_ratio": 0.08, "min_cyrillic_chars": 1.0, "merge_similarity": 0.54})
    else:
        values.update({"min_cyrillic_ratio": options.min_cyrillic_ratio, "min_cyrillic_chars": float(options.min_cyrillic_chars), "merge_similarity": options.merge_similarity})
    return values


def detect_common_subtitle_region(crops: list) -> tuple[float, float] | None:
    y1_values = [float(crop.text_bbox_y1) for crop in crops[:200] if getattr(crop, "text_bbox_y1", 0.0) > 0]
    y2_values = [float(crop.text_bbox_y2) for crop in crops[:200] if getattr(crop, "text_bbox_y2", 0.0) > 0]
    if len(y1_values) < 10 or len(y2_values) < 10:
        return None
    y1 = max(0.55, min(0.82, median(y1_values) - 0.06))
    y2 = max(y1 + 0.12, min(0.99, median(y2_values) + 0.06))
    return y1, y2


def apply_ocr_preset(options: ProcessOptions) -> dict[str, float]:
    if options.ocr_preset not in {"strict", "balanced", "recall", "maximum_recall"}:
        raise ValueError("--ocr-preset musí být strict, balanced, recall nebo maximum_recall.")
    values = {
        "crop_x1": DEFAULT_CROP_X1,
        "crop_x2": DEFAULT_CROP_X2,
        "crop_y1": DEFAULT_CROP_Y1,
        "crop_y2": DEFAULT_CROP_Y2,
        "min_cyrillic_ratio": options.min_cyrillic_ratio,
        "min_cyrillic_chars": float(options.min_cyrillic_chars),
        "merge_similarity": options.merge_similarity,
        "auto_region_used": False,
    }
    if options.ocr_preset == "strict":
        values.update({"min_cyrillic_ratio": 0.35, "min_cyrillic_chars": 4.0, "merge_similarity": 0.72})
    elif options.ocr_preset == "balanced":
        values.update({"min_cyrillic_ratio": 0.25, "min_cyrillic_chars": 3.0, "merge_similarity": 0.68})
    elif options.ocr_preset == "recall":
        values.update({
            "crop_x1": DEFAULT_CROP_X1,
            "crop_x2": DEFAULT_CROP_X2,
            "crop_y1": DEFAULT_CROP_Y1,
            "crop_y2": DEFAULT_CROP_Y2,
            "min_cyrillic_ratio": 0.12,
            "min_cyrillic_chars": 2.0,
            "merge_similarity": 0.58,
        })
    elif options.ocr_preset == "maximum_recall":
        values.update({
            "crop_x1": DEFAULT_CROP_X1,
            "crop_x2": DEFAULT_CROP_X2,
            "crop_y1": DEFAULT_CROP_Y1,
            "crop_y2": DEFAULT_CROP_Y2,
            "min_cyrillic_ratio": 0.08,
            "min_cyrillic_chars": 1.0,
            "merge_similarity": 0.54,
        })
    for key in ("crop_x1", "crop_x2", "crop_y1", "crop_y2"):
        user_value = getattr(options, key)
        if user_value is not None:
            values[key] = float(user_value)
    return values


def log_ocr_detection_summary(options: ProcessOptions, extraction_stats, ocr_candidates: int, subtitle_blocks: int) -> None:
    accepted = parse_report_int(options.ocr_report, "accepted_ocr_results")
    rejected = parse_report_int(options.ocr_report, "noise_rejected")
    accepted_ratio = parse_report_float(options.ocr_report, "accepted_ratio")
    logging.info(
        "OCR detection summary: detect_mode=%s quality_profile=%s white_text_frames=%d skipped_without_subtitles=%d skipped_same_subtitle_frames=%d ocr_candidates=%d accepted=%d rejected=%d accepted_ratio=%.3f subtitle_blocks=%d",
        options.subtitle_detect_mode,
        options.quality_profile,
        int(getattr(extraction_stats, "white_text_frames", 0)),
        int(getattr(extraction_stats, "skipped_without_subtitles", 0)),
        int(getattr(extraction_stats, "skipped_same_subtitle_frames", 0)),
        ocr_candidates,
        accepted,
        rejected,
        accepted_ratio,
        subtitle_blocks,
    )


def append_timing_to_report(report_path: Path, subtitle_changes: int, blocks: list[dict]) -> None:
    durations = [float(block["end"]) - float(block["start"]) for block in blocks if "start" in block and "end" in block]
    average_duration = sum(durations) / len(durations) if durations else 0.0
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\nTIMING\n")
        handle.write("=" * 40 + "\n")
        handle.write(f"subtitle_changes_or_ocr_crops: {subtitle_changes}\n")
        handle.write(f"subtitle_blocks: {len(blocks)}\n")
        handle.write(f"average_subtitle_duration: {average_duration:.3f}\n")
