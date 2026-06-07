from __future__ import annotations

import logging
import re
import threading
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from src.config import (
    CS_SRT,
    DEFAULT_BOX_HEIGHT,
    DEFAULT_DEEP_OCR,
    DEFAULT_MAX_SUBTITLE_GAP,
    DEFAULT_MERGE_SIMILARITY,
    DEFAULT_MIN_CYRILLIC_CHARS,
    DEFAULT_MIN_CYRILLIC_RATIO,
    DEFAULT_OCR_LANG,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SPEED_MODE,
    DEFAULT_SUBTITLE_AREA,
    DEFAULT_TARGET_LANG,
    DEFAULT_TESSERACT_PSM,
    DEFAULT_TRANSLATION_BATCH_SIZE,
    INPUT_DIR,
    OUTPUT_DIR,
    OUTPUT_VIDEO,
    SOURCE_VIDEO,
    TEMP_DIR,
)
from src.processor import ProcessOptions, process_video
from src.utils import ensure_dirs


app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
ensure_dirs(INPUT_DIR, OUTPUT_DIR, TEMP_DIR, OUTPUT_DIR / "jobs")

LOG_PATH = OUTPUT_DIR / "last_run.log"
ALLOWED_DOWNLOADS = {
    "subtitles_original_raw.srt",
    "subtitles_original_context_fixed.srt",
    "subtitles_cs.srt",
    "video_cz.mp4",
    "last_run.log",
    "ocr_compare.txt",
    "ocr_report.txt",
    "ocr_report.json",
    "ocr_samples.txt",
    "context_corrections.json",
    "translation_report.json",
    "translation_cache.json",
}
BATCH_DOWNLOADS = {
    "subtitles_original_raw.srt",
    "subtitles_original_context_fixed.srt",
    "subtitles_cs.srt",
    "video_cz.mp4",
    "log.txt",
    "ocr_report.txt",
    "ocr_report.json",
    "ocr_samples.txt",
    "context_corrections.json",
    "translation_report.json",
}

UI_DEFAULTS = {
    "ocr_engine": "paddle",
    "subtitle_detect_mode": "hybrid",
    "ocr_preset": "recall",
    "deep_ocr": DEFAULT_DEEP_OCR,
    "speed_mode": DEFAULT_SPEED_MODE,
    "quality_profile": "default",
    "ocr_preprocess": "auto",
    "subtitle_style": "auto",
    "translator": "none",
    "ai_cleanup": "auto",
    "ocr_min_gap": 2.0,
    "subtitle_mask_similarity": 0.90,
    "ocr_frame_timeout": 30.0,
    "paddle_init_timeout": 60.0,
    "max_actual_ocr_calls": "",
    "cleanup_temp": True,
    "crop_y1": 0.78,
    "crop_y2": 0.93,
    "crop_x1": 0.08,
    "crop_x2": 0.92,
}


@dataclass
class BatchJob:
    id: str
    url: str
    dir: Path
    options: ProcessOptions | None = None
    status: str = "queued"
    stage: str = "Čeká"
    progress: int = 0
    error: str | None = None
    outputs: dict[str, bool] = field(default_factory=lambda: {
        "srt_raw": False,
        "srt_context": False,
        "srt_cs": False,
        "ocr_report": False,
        "translation_report": False,
        "log": False,
        "video": False,
    })


technical_logs: deque[str] = deque(maxlen=1200)
human_logs: deque[str] = deque(maxlen=160)
job_lock = threading.Lock()

job_running = False
job_done = False
job_error: str | None = None
job_outputs: list[str] = []
job_progress = 0
job_stage = "Připraveno"
last_video_path: Path | None = None

batch_running = False
batch_stop_requested = False
batch_done = False
batch_current_job_id: str | None = None
batch_jobs: list[BatchJob] = []
batch_log_path: Path | None = None


class UiLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if record.name == "werkzeug":
            return
        line = self.format(record)
        technical_logs.append(line)
        message = record.getMessage()
        update_single_status_from_log(message)
        update_batch_status_from_log(message)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if batch_log_path is not None:
            batch_log_path.parent.mkdir(parents=True, exist_ok=True)
            with batch_log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = UiLogHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    root.handlers.clear()
    root.addHandler(handler)


configure_logging()


def form_bool(name: str) -> bool:
    return request.form.get(name) == "on"


def add_human_log(message: str) -> None:
    with job_lock:
        if not human_logs or human_logs[-1] != message:
            human_logs.append(message)


def set_single_status(progress: int, stage: str, human_message: str | None = None) -> None:
    global job_progress, job_stage
    with job_lock:
        job_progress = max(job_progress, max(0, min(100, int(progress))))
        job_stage = stage
        if human_message and (not human_logs or human_logs[-1] != human_message):
            human_logs.append(human_message)


def update_single_status_from_log(message: str) -> None:
    if batch_running:
        return
    progress, stage, human = progress_from_log(message)
    if progress is not None and stage is not None:
        set_single_status(progress, stage, human)


def update_batch_status_from_log(message: str) -> None:
    if not batch_running or batch_current_job_id is None:
        return
    progress, stage, human = progress_from_log(message)
    if progress is None or stage is None:
        return
    with job_lock:
        job = find_batch_job(batch_current_job_id)
        if job is None:
            return
        job.progress = max(job.progress, progress)
        job.stage = stage.rstrip(".")
        if human and (not human_logs or human_logs[-1] != human):
            human_logs.append(human)


def progress_from_log(message: str) -> tuple[int | None, str | None, str | None]:
    if "Zkouším" in message or "yt-dlp" in message:
        return 12, "Stahuji video...", "Stahuji video..."
    if "Video uloženo" in message:
        return 30, "Video staženo.", "Video staženo."
    if "Video:" in message or "Spouštím OCR" in message or "OCR crop" in message:
        return 35, "Čtu ruské titulky z obrazu...", "Čtu titulky..."
    if match := re.search(r"OCR frame\s+(\d+)/(\d+)", message):
        current, total = int(match.group(1)), max(1, int(match.group(2)))
        return 30 + round((current / total) * 35), "Čtu ruské titulky z obrazu...", "Čtu titulky..."
    if "Po OCR" in message or ("SRT uloženo" in message and "subtitles_original_raw" in message):
        return 65, "Překládám do češtiny...", "Překládám..."
    if "Překládám" in message or "Google fallback" in message or "Ollama" in message:
        return 68, "Překládám do češtiny...", "Překládám..."
    if match := re.search(r"Překlad průběh:\s+(\d+)/(\d+)", message):
        current, total = int(match.group(1)), max(1, int(match.group(2)))
        return 65 + round((current / total) * 20), "Překládám do češtiny...", "Překládám..."
    if "subtitles_cs.srt" in message:
        return 85, "Vytvářím hotové video...", "Vytvářím hotové video..."
    if "Vypaluji" in message:
        return 88, "Vkládám titulky do videa...", "Vytvářím hotové video..."
    if "Hotové video" in message:
        return 100, "Hotovo.", "Hotovo."
    return None, None, None


def find_batch_job(job_id: str) -> BatchJob | None:
    for job in batch_jobs:
        if job.id == job_id:
            return job
    return None


def save_upload() -> Path | None:
    upload = request.files.get("video_file")
    if upload is None or not upload.filename:
        return None
    filename = secure_filename(upload.filename)
    if not filename:
        raise ValueError("Název nahraného souboru není platný.")
    target = INPUT_DIR / f"web_upload_{filename}"
    upload.save(target)
    return target


def build_options_from_request() -> ProcessOptions:
    uploaded_path = save_upload()
    url = request.form.get("url", "").strip() or None
    if uploaded_path is None and url is None:
        raise ValueError("Vložte odkaz na video nebo vyberte video z počítače.")
    if uploaded_path is not None and url is not None:
        raise ValueError("Použijte buď odkaz, nebo video z počítače, ne oboje najednou.")

    burn = not form_bool("only_srt")
    return ProcessOptions(
        url=url,
        input_path=uploaded_path,
        burn=burn,
        only_srt=not burn,
        cookies_browser=cookies_browser_from_form(),
        **common_options_from_form(),
    )


def build_batch_options(job: BatchJob, only_srt: bool, cookies_browser: str | None, common_options: dict) -> ProcessOptions:
    return ProcessOptions(
        url=job.url,
        burn=not only_srt,
        only_srt=only_srt,
        cookies_browser=cookies_browser,
        input_dir=job.dir,
        output_dir=job.dir,
        temp_dir=job.dir / "temp",
        source_video=job.dir / "source.mp4",
        raw_ru_srt=job.dir / "subtitles_original_raw.srt",
        clean_ru_srt=job.dir / "subtitles_original_context_fixed.srt",
        cs_srt=job.dir / "subtitles_cs.srt",
        context_corrections=job.dir / "context_corrections.json",
        translation_report=job.dir / "translation_report.json",
        output_video=job.dir / "video_cz.mp4",
        ocr_samples=job.dir / "ocr_samples.txt",
        ocr_compare=job.dir / "ocr_compare.txt",
        ocr_report=job.dir / "ocr_report.txt",
        ocr_report_json=job.dir / "ocr_report.json",
        **common_options,
    )


def cookies_browser_from_form() -> str | None:
    if form_bool("cookies_chrome"):
        return "chrome"
    if form_bool("cookies_edge"):
        return "edge"
    return None


def common_options_from_form() -> dict:
    expert_paddle_enabled = form_bool("expert_enable_paddle")
    manual_crop_enabled = form_bool("manual_crop")
    requested_engine = request.form.get("ocr_engine", UI_DEFAULTS["ocr_engine"])
    if requested_engine == "compare" and not expert_paddle_enabled:
        requested_engine = "paddle"
    return {
        "ocr_lang": request.form.get("ocr_lang", DEFAULT_OCR_LANG).strip() or DEFAULT_OCR_LANG,
        "target_lang": request.form.get("target_lang", DEFAULT_TARGET_LANG).strip() or DEFAULT_TARGET_LANG,
        "sample_rate": float(request.form.get("sample_rate", DEFAULT_SAMPLE_RATE)),
        "subtitle_area": request.form.get("subtitle_area", DEFAULT_SUBTITLE_AREA).strip() or DEFAULT_SUBTITLE_AREA,
        "box_height": float(request.form.get("box_height", DEFAULT_BOX_HEIGHT)),
        "translator": request.form.get("translator", UI_DEFAULTS["translator"]),
        "translation_batch_size": int(request.form.get("translation_batch_size", DEFAULT_TRANSLATION_BATCH_SIZE)),
        "ocr_preset": request.form.get("ocr_preset", UI_DEFAULTS["ocr_preset"]),
        "ocr_engine": requested_engine,
        "deep_ocr": request.form.get("deep_ocr", UI_DEFAULTS["deep_ocr"]),
        "speed_mode": request.form.get("speed_mode", UI_DEFAULTS["speed_mode"]),
        "quality_profile": request.form.get("quality_profile", UI_DEFAULTS["quality_profile"]),
        "subtitle_style": request.form.get("subtitle_style", UI_DEFAULTS["subtitle_style"]),
        "ai_cleanup": request.form.get("ai_cleanup", UI_DEFAULTS["ai_cleanup"]),
        "detect_subtitle_changes": form_bool("detect_subtitle_changes"),
        "subtitle_detect_mode": request.form.get("subtitle_detect_mode", UI_DEFAULTS["subtitle_detect_mode"]),
        "ocr_min_gap": float(request.form.get("ocr_min_gap", UI_DEFAULTS["ocr_min_gap"])),
        "subtitle_mask_similarity": float(request.form.get("subtitle_mask_similarity", UI_DEFAULTS["subtitle_mask_similarity"])),
        "ocr_frame_timeout": float(request.form.get("ocr_frame_timeout", UI_DEFAULTS["ocr_frame_timeout"])),
        "paddle_init_timeout": float(request.form.get("paddle_init_timeout", UI_DEFAULTS["paddle_init_timeout"])),
        "max_actual_ocr_calls": optional_int("max_actual_ocr_calls"),
        "disable_paddle": form_bool("disable_paddle"),
        "cleanup_temp": request.form.get("cleanup_temp", "on") == "on",
        "crop_x1": optional_float("crop_x1") if manual_crop_enabled else None,
        "crop_x2": optional_float("crop_x2") if manual_crop_enabled else None,
        "crop_y1": optional_float("crop_y1") if manual_crop_enabled else None,
        "crop_y2": optional_float("crop_y2") if manual_crop_enabled else None,
        "ocr_preprocess": request.form.get("ocr_preprocess", UI_DEFAULTS["ocr_preprocess"]),
        "tesseract_psm": int(request.form.get("tesseract_psm", DEFAULT_TESSERACT_PSM)),
        "min_cyrillic_ratio": float(request.form.get("min_cyrillic_ratio", DEFAULT_MIN_CYRILLIC_RATIO)),
        "min_cyrillic_chars": int(request.form.get("min_cyrillic_chars", DEFAULT_MIN_CYRILLIC_CHARS)),
        "merge_similarity": float(request.form.get("merge_similarity", DEFAULT_MERGE_SIMILARITY)),
        "max_subtitle_gap": float(request.form.get("max_subtitle_gap", DEFAULT_MAX_SUBTITLE_GAP)),
        "debug_crops": form_bool("debug_crops"),
        "debug_subtitle_detection": form_bool("debug_subtitle_detection"),
        "defender_scan": form_bool("defender_scan"),
        "defender_strict": form_bool("defender_strict"),
        "start_time": request.form.get("start_time", "").strip() or None,
        "end_time": request.form.get("end_time", "").strip() or None,
        "max_ocr_frames": optional_int("max_ocr_frames"),
    }


def optional_float(name: str, default: float | None = None) -> float | None:
    value = request.form.get(name, "").strip()
    return float(value) if value else default


def optional_int(name: str) -> int | None:
    value = request.form.get(name, "").strip()
    return int(value) if value else None


def run_job(options: ProcessOptions) -> None:
    global job_done, job_error, job_outputs, job_running, last_video_path
    try:
        set_single_status(5, "Připravuji...", "Připravuji...")
        logging.info("Spouštím zpracování.")
        result = process_video(options, defender_continue_callback=lambda: True)
        last_video_path = SOURCE_VIDEO if options.url else options.input_path
        outputs = [
            result.raw_ru_srt.name,
            result.context_fixed_srt.name,
            result.cs_srt.name,
            result.ocr_report_json.name,
            result.translation_report.name,
            LOG_PATH.name,
        ]
        if result.output_video:
            outputs.append(result.output_video.name)
        job_outputs = [name for name in outputs if (OUTPUT_DIR / name).exists()]
        job_done = True
        set_single_status(100, "Hotovo.", "Hotovo.")
        logging.info("Zpracování dokončeno.")
    except Exception as exc:
        job_error = human_error_message(exc)
        job_done = True
        set_single_status(job_progress, "Chyba", job_error)
        logging.exception("Zpracování selhalo: %s", exc)
    finally:
        with job_lock:
            job_running = False


def run_burn_job() -> None:
    global job_done, job_error, job_outputs, job_running
    try:
        if last_video_path is None or not last_video_path.exists():
            raise FileNotFoundError("Původní video nebylo nalezeno.")
        if not CS_SRT.exists():
            raise FileNotFoundError("České titulky zatím nejsou hotové.")
        set_single_status(85, "Vkládám titulky do videa...", "Vytvářím hotové video...")
        result = process_video(
            ProcessOptions(input_path=last_video_path, burn_existing_srt=CS_SRT, box_height=DEFAULT_BOX_HEIGHT),
            defender_continue_callback=lambda: True,
        )
        outputs = [result.cs_srt.name]
        if result.output_video:
            outputs.append(result.output_video.name)
        job_outputs = [name for name in outputs if (OUTPUT_DIR / name).exists()]
        job_done = True
        set_single_status(100, "Hotovo.", "Hotovo.")
    except Exception as exc:
        job_error = str(exc) or "Vložení titulků do videa selhalo."
        job_done = True
        set_single_status(job_progress, "Chyba", job_error)
        logging.exception("Vložení titulků do videa selhalo: %s", exc)
    finally:
        with job_lock:
            job_running = False


def run_batch() -> None:
    global batch_running, batch_done, batch_current_job_id, batch_log_path
    try:
        for job in batch_jobs:
            with job_lock:
                if batch_stop_requested:
                    job.status = "stopped"
                    job.stage = "Zastaveno"
                    continue
                job.status = "running"
                job.stage = "Připravuji"
                job.progress = max(job.progress, 3)
                batch_current_job_id = job.id
                batch_log_path = job.dir / "log.txt"
                human_logs.append(f"{job.id}: Připravuji...")
            ensure_dirs(job.dir, job.dir / "temp")
            (job.dir / "log.txt").write_text("", encoding="utf-8")
            try:
                logging.info("Spouštím dávkovou úlohu %s.", job.id)
                if job.options is None:
                    raise RuntimeError("Dávková úloha nemá připravené nastavení.")
                result = process_video(job.options, defender_continue_callback=lambda: True)
                job.outputs = {
                    "srt_raw": result.raw_ru_srt.exists(),
                    "srt_context": result.context_fixed_srt.exists(),
                    "srt_cs": result.cs_srt.exists(),
                    "ocr_report": result.ocr_report_json.exists(),
                    "translation_report": result.translation_report.exists(),
                    "log": (job.dir / "log.txt").exists(),
                    "video": bool(result.output_video and result.output_video.exists()),
                }
                job.progress = 100
                job.stage = "Hotovo"
                job.status = "done"
                add_human_log(f"{job.id}: Hotovo.")
            except Exception as exc:
                job.error = human_error_message(exc)
                if str(exc).startswith("translation_failed:"):
                    job.status = "translation_failed"
                    job.stage = "Překlad selhal"
                else:
                    job.status = "error"
                    job.stage = "Chyba"
                job.progress = max(job.progress, 1)
                add_human_log(f"{job.id}: Chyba - {job.error}")
                logging.exception("Dávková úloha %s selhala: %s", job.id, exc)
            finally:
                with job_lock:
                    batch_current_job_id = None
                    batch_log_path = None
        with job_lock:
            batch_done = True
    finally:
        with job_lock:
            batch_running = False


def human_error_message(exc: Exception) -> str:
    text = str(exc)
    if text.startswith("translation_failed:"):
        return "Překlad selhal: " + text.split(":", 1)[1].strip()
    if "Stažení videa selhalo" in text:
        return "Video se nepodařilo stáhnout. Zkontrolujte odkaz nebo zkuste jiné video."
    return text or "Úlohu se nepodařilo dokončit."


def single_status() -> dict:
    return {
        "running": job_running,
        "done": job_done,
        "error": job_error,
        "progress": job_progress,
        "stage": job_stage,
        "human_log": list(human_logs),
        "technical_log": list(technical_logs),
        "outputs": {
            "srt_raw": "subtitles_original_raw.srt" in job_outputs and (OUTPUT_DIR / "subtitles_original_raw.srt").exists(),
            "srt_context": "subtitles_original_context_fixed.srt" in job_outputs and (OUTPUT_DIR / "subtitles_original_context_fixed.srt").exists(),
            "srt_cs": "subtitles_cs.srt" in job_outputs and CS_SRT.exists(),
            "ocr_report": "ocr_report.json" in job_outputs and (OUTPUT_DIR / "ocr_report.json").exists(),
            "translation_report": "translation_report.json" in job_outputs and (OUTPUT_DIR / "translation_report.json").exists(),
            "video": "video_cz.mp4" in job_outputs and OUTPUT_VIDEO.exists(),
            "can_burn": "subtitles_cs.srt" in job_outputs and CS_SRT.exists() and last_video_path is not None and last_video_path.exists(),
        },
    }


def batch_status_payload() -> dict:
    jobs_payload = [serialize_batch_job(job) for job in batch_jobs]
    overall_progress = round(sum(job.progress for job in batch_jobs) / len(batch_jobs)) if batch_jobs else 0
    current_job = find_batch_job(batch_current_job_id) if batch_current_job_id else None
    return {
        "running": batch_running,
        "overall_progress": overall_progress,
        "current_stage": current_job.stage if current_job else ("Hotovo." if batch_done else "Připraveno."),
        "jobs": jobs_payload,
        "human_log": list(human_logs),
        "technical_log": list(technical_logs),
    }


def serialize_batch_job(job: BatchJob) -> dict:
    ocr_report = read_json(job.dir / "ocr_report.json")
    translation_report = read_json(job.dir / "translation_report.json")
    job.outputs = {
        "srt_raw": (job.dir / "subtitles_original_raw.srt").exists(),
        "srt_context": (job.dir / "subtitles_original_context_fixed.srt").exists(),
        "srt_cs": (job.dir / "subtitles_cs.srt").exists(),
        "ocr_report": (job.dir / "ocr_report.json").exists(),
        "translation_report": (job.dir / "translation_report.json").exists(),
        "log": (job.dir / "log.txt").exists(),
        "video": (job.dir / "video_cz.mp4").exists(),
    }
    return {
        "id": job.id,
        "url": job.url,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "outputs": job.outputs,
        "metrics": {
            "used_crop": ocr_report.get("used_crop"),
            "total_frames_sampled": ocr_report.get("total_frames_sampled", 0),
            "frames_with_subtitles": ocr_report.get("frames_with_subtitles", 0),
            "skipped_without_subtitles": ocr_report.get("skipped_without_subtitles", 0),
            "skipped_same_subtitle": ocr_report.get("skipped_same_subtitle", 0),
            "actual_ocr_calls": ocr_report.get("actual_ocr_calls", 0),
            "subtitle_blocks": ocr_report.get("subtitle_blocks_generated", 0),
            "translated_blocks": translation_report.get("translated_blocks", 0),
            "translation_status": translation_report.get("status"),
            "translation_error": translation_report.get("error"),
        },
        "error": job.error,
    }


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def parse_urls_from_request() -> list[str]:
    raw = request.form.get("urls", "").strip()
    urls = [line.strip() for line in raw.splitlines() if line.strip()]
    if not urls:
        single = request.form.get("url", "").strip()
        if single:
            urls = [single]
    if not urls:
        raise ValueError("Vložte alespoň jeden odkaz na video.")
    return urls


@app.get("/")
def index():
    return render_template(
        "index.html",
        defaults={
            "ocr_lang": DEFAULT_OCR_LANG,
            "target_lang": DEFAULT_TARGET_LANG,
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "subtitle_area": DEFAULT_SUBTITLE_AREA,
            "box_height": DEFAULT_BOX_HEIGHT,
            "translator": UI_DEFAULTS["translator"],
            "ocr_preset": UI_DEFAULTS["ocr_preset"],
            "ocr_engine": UI_DEFAULTS["ocr_engine"],
            "deep_ocr": UI_DEFAULTS["deep_ocr"],
            "speed_mode": UI_DEFAULTS["speed_mode"],
            "quality_profile": UI_DEFAULTS["quality_profile"],
            "subtitle_style": UI_DEFAULTS["subtitle_style"],
            "ai_cleanup": UI_DEFAULTS["ai_cleanup"],
            "translation_batch_size": DEFAULT_TRANSLATION_BATCH_SIZE,
            "crop_x1": UI_DEFAULTS["crop_x1"],
            "crop_x2": UI_DEFAULTS["crop_x2"],
            "crop_y1": UI_DEFAULTS["crop_y1"],
            "crop_y2": UI_DEFAULTS["crop_y2"],
            "ocr_preprocess": UI_DEFAULTS["ocr_preprocess"],
            "tesseract_psm": DEFAULT_TESSERACT_PSM,
            "min_cyrillic_ratio": DEFAULT_MIN_CYRILLIC_RATIO,
            "min_cyrillic_chars": DEFAULT_MIN_CYRILLIC_CHARS,
            "merge_similarity": DEFAULT_MERGE_SIMILARITY,
            "max_subtitle_gap": DEFAULT_MAX_SUBTITLE_GAP,
            "start_time": "",
            "end_time": "",
            "max_ocr_frames": "",
            "subtitle_detect_mode": UI_DEFAULTS["subtitle_detect_mode"],
            "ocr_min_gap": UI_DEFAULTS["ocr_min_gap"],
            "subtitle_mask_similarity": UI_DEFAULTS["subtitle_mask_similarity"],
            "ocr_frame_timeout": UI_DEFAULTS["ocr_frame_timeout"],
            "paddle_init_timeout": UI_DEFAULTS["paddle_init_timeout"],
            "max_actual_ocr_calls": UI_DEFAULTS["max_actual_ocr_calls"],
            "cleanup_temp": UI_DEFAULTS["cleanup_temp"],
        },
    )


@app.post("/batch/start")
def batch_start():
    global batch_running, batch_stop_requested, batch_done, batch_jobs, batch_current_job_id, batch_log_path
    global job_done, job_error, job_outputs, job_running, job_progress, job_stage, last_video_path
    with job_lock:
        if batch_running or job_running:
            return jsonify({"ok": False, "error": "Zpracování už běží. Počkejte na dokončení nebo ho zastavte."}), 409
        try:
            urls = parse_urls_from_request()
            only_srt = form_bool("only_srt")
            cookies_browser = cookies_browser_from_form()
            common_options = common_options_from_form()
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        batch_jobs = []
        for index, url in enumerate(urls, start=1):
            job_id = f"job_{index:03d}"
            job = BatchJob(id=job_id, url=url, dir=OUTPUT_DIR / "jobs" / job_id)
            job.options = build_batch_options(job, only_srt=only_srt, cookies_browser=cookies_browser, common_options=common_options)
            batch_jobs.append(job)
        batch_running = True
        batch_stop_requested = False
        batch_done = False
        batch_current_job_id = None
        batch_log_path = None
        human_logs.clear()
        technical_logs.clear()
        LOG_PATH.write_text("", encoding="utf-8")
        job_done = False
        job_error = None
        job_outputs = []
        job_running = False
        job_progress = 0
        job_stage = "Připraveno"
        last_video_path = None
        human_logs.append("Připravuji...")

    thread = threading.Thread(target=run_batch, daemon=True)
    thread.start()
    return jsonify({"ok": True})


@app.get("/batch/status")
def batch_status():
    return jsonify(batch_status_payload())


@app.post("/batch/stop")
def batch_stop():
    global batch_stop_requested
    with job_lock:
        batch_stop_requested = True
        for job in batch_jobs:
            if job.status == "queued":
                job.status = "stopped"
                job.stage = "Zastaveno"
        human_logs.append("Zpracování bude zastaveno po aktuálním kroku.")
    return jsonify({"ok": True})


@app.get("/batch/download/<job_id>/<path:filename>")
def batch_download(job_id: str, filename: str):
    safe_job_id = secure_filename(job_id)
    safe_name = secure_filename(Path(filename).name)
    if safe_job_id != job_id or safe_name != filename or safe_name not in BATCH_DOWNLOADS:
        return jsonify({"error": "Soubor není povolený ke stažení."}), 404
    job_dir = (OUTPUT_DIR / "jobs" / safe_job_id).resolve()
    output_root = (OUTPUT_DIR / "jobs").resolve()
    if output_root not in job_dir.parents and job_dir != output_root:
        return jsonify({"error": "Neplatná cesta."}), 404
    target = job_dir / safe_name
    if not target.exists():
        return jsonify({"error": "Soubor zatím neexistuje."}), 404
    return send_from_directory(job_dir, safe_name, as_attachment=True)


@app.post("/start")
def start():
    global job_done, job_error, job_outputs, job_running, job_progress, job_stage, last_video_path
    with job_lock:
        if job_running or batch_running:
            return jsonify({"ok": False, "error": "Zpracování už běží. Počkejte na dokončení."}), 409
        job_running = True
        job_done = False
        job_error = None
        job_outputs = []
        job_progress = 0
        job_stage = "Připravuji..."
        last_video_path = None
        human_logs.clear()
        technical_logs.clear()
        LOG_PATH.write_text("", encoding="utf-8")
    try:
        options = build_options_from_request()
    except Exception as exc:
        message = human_error_message(exc)
        with job_lock:
            job_running = False
            job_done = True
            job_error = message
            human_logs.append(message)
        logging.error("%s", exc)
        return jsonify({"ok": False, "error": message}), 400
    thread = threading.Thread(target=run_job, args=(options,), daemon=True)
    thread.start()
    return jsonify({"ok": True})


@app.post("/burn")
def burn_existing():
    global job_done, job_error, job_outputs, job_running, job_progress, job_stage
    with job_lock:
        if job_running or batch_running:
            return jsonify({"ok": False, "error": "Zpracování už běží. Počkejte na dokončení."}), 409
        job_running = True
        job_done = False
        job_error = None
        job_outputs = []
        job_progress = 85
        job_stage = "Vkládám titulky do videa..."
        human_logs.append("Vytvářím hotové video...")
    thread = threading.Thread(target=run_burn_job, daemon=True)
    thread.start()
    return jsonify({"ok": True})


@app.get("/status")
def status():
    return jsonify(single_status())


@app.get("/logs")
def get_logs():
    status_data = single_status()
    return jsonify(
        {
            "running": status_data["running"],
            "done": status_data["done"],
            "error": status_data["error"],
            "logs": status_data["technical_log"],
            "outputs": job_outputs,
        }
    )


@app.get("/download/<path:filename>")
def download(filename: str):
    safe_name = secure_filename(Path(filename).name)
    if safe_name != filename or safe_name not in ALLOWED_DOWNLOADS:
        return jsonify({"error": "Soubor není povolený ke stažení."}), 404
    target = (OUTPUT_DIR / safe_name).resolve()
    output_root = OUTPUT_DIR.resolve()
    if output_root not in target.parents and target != output_root:
        return jsonify({"error": "Neplatná cesta."}), 404
    if not target.exists():
        return jsonify({"error": "Soubor zatím neexistuje."}), 404
    return send_from_directory(output_root, safe_name, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
