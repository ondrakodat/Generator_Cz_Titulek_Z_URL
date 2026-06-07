from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    width: int
    height: int
    duration: float


@dataclass(frozen=True)
class FrameCrop:
    time: float
    image_path: Path
    end_time: float | None = None
    changed: bool = True
    diff_score: float = 1.0
    has_subtitle: bool = True
    white_text_detected: bool = True
    text_component_count: int = 0
    text_bbox_area: float = 0.0
    text_bbox_y1: float = 0.0
    text_bbox_y2: float = 0.0
    mask_similarity: float = 0.0


@dataclass(frozen=True)
class ExtractionStats:
    sampled_frames_total: int = 0
    white_text_frames: int = 0
    skipped_without_subtitles: int = 0
    skipped_same_subtitle_frames: int = 0
    actual_ocr_calls: int = 0


@dataclass(frozen=True)
class WhiteTextDetection:
    detected: bool
    component_count: int
    bbox_area: float
    mask: np.ndarray
    boxes: tuple[tuple[int, int, int, int], ...]
    text_bbox: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class CropRegion:
    x1: float = 0.08
    x2: float = 0.92
    y1: float = 0.78
    y2: float = 0.93

    def validate(self) -> None:
        values = (self.x1, self.x2, self.y1, self.y2)
        if any(value < 0 or value > 1 for value in values):
            raise ValueError("Crop hodnoty musí být v rozsahu 0.0 až 1.0.")
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("Crop oblast musí mít x1 < x2 a y1 < y2.")


def get_video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video nelze otevřít: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = frame_count / fps if fps else 0.0
    return VideoInfo(fps=fps, width=width, height=height, duration=duration)


def detect_white_subtitle_text(crop: np.ndarray, legacy: bool = False) -> WhiteTextDetection:
    if crop.size == 0:
        empty = np.zeros((1, 1), dtype=np.uint8)
        return WhiteTextDetection(False, 0, 0.0, empty, ())

    height, width = crop.shape[:2]
    white_mask = cv2.inRange(crop, np.array([180, 180, 180]), np.array([255, 255, 255]))
    white_mask[: int(height * 0.18), :] = 0
    raw_bright_ratio = float(cv2.countNonZero(white_mask)) / max(1, width * height)
    if raw_bright_ratio > (0.42 if legacy else 0.32):
        return WhiteTextDetection(False, 0, 0.0, white_mask, ())

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8), iterations=1)

    label_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    total_area = 0.0
    min_y_center = height * (0.33 if legacy else 0.50)
    max_component_area = width * height * 0.18

    for label in range(1, label_count):
        x, y, box_w, box_h, area = (int(value) for value in stats[label])
        if box_h < 10 or box_h > 80:
            continue
        if box_w < 18:
            continue
        if area < 35:
            continue
        if area > max_component_area:
            continue
        density = area / max(1, box_w * box_h)
        if density < 0.06 or density > 0.82:
            continue
        aspect = box_w / max(1, box_h)
        if aspect < 0.35:
            continue
        if box_w > width * 0.92 and box_h > height * 0.45:
            continue
        if y + box_h / 2 < min_y_center:
            continue
        boxes.append((x, y, box_w, box_h))
        total_area += float(area)

    if not boxes:
        return WhiteTextDetection(False, 0, 0.0, mask, ())

    x_min = min(x for x, _, _, _ in boxes)
    x_max = max(x + box_w for x, _, box_w, _ in boxes)
    y_min = min(y for _, y, _, _ in boxes)
    y_max = max(y + box_h for _, y, _, box_h in boxes)
    bbox_area = float((x_max - x_min) * (y_max - y_min))
    bright_ratio = float(cv2.countNonZero(white_mask)) / max(1, width * height)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    centered = width * 0.18 <= center_x <= width * 0.82
    lower_crop = center_y >= height * (0.38 if legacy else 0.50)
    enough_components = len(boxes) >= 2 or total_area >= 220
    enough_width = (x_max - x_min) >= max(50, width * 0.12)
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    text_like_height = True if legacy else 10 <= bbox_height <= min(80, height * 0.70)
    multiline_text_like = True if legacy else (bbox_height <= 45 or (len(boxes) >= 7 and bbox_width >= width * 0.50))
    not_bright_background = bright_ratio <= 0.24 or (len(boxes) >= 3 and raw_bright_ratio <= 0.32)
    detected = bool(enough_components and enough_width and centered and lower_crop and text_like_height and multiline_text_like and not_bright_background)
    return WhiteTextDetection(detected, len(boxes), bbox_area, mask, tuple(boxes), (x_min, y_min, x_max, y_max))


def crop_to_detected_text(crop: np.ndarray, detection: WhiteTextDetection) -> tuple[np.ndarray, tuple[int, int, int, int] | None]:
    if not detection.text_bbox:
        return crop, None
    height, width = crop.shape[:2]
    x_min, y_min, x_max, y_max = detection.text_bbox
    pad_x = max(12, int(width * 0.025))
    pad_y = max(8, int(height * 0.06))
    x1 = max(0, x_min - pad_x)
    x2 = min(width, x_max + pad_x)
    y1 = max(0, y_min - pad_y)
    y2 = min(height, y_max + pad_y)
    if x2 <= x1 or y2 <= y1:
        return crop, None
    return crop[y1:y2, x1:x2], (x1, y1, x2, y2)


def save_detection_debug(debug_dir: Path, index: int, crop: np.ndarray, detection: WhiteTextDetection) -> None:
    crop_dir = debug_dir / "crop"
    mask_dir = debug_dir / "mask"
    boxes_dir = debug_dir / "boxes"
    crop_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    boxes_dir.mkdir(parents=True, exist_ok=True)
    name = f"detection_{index:06d}.png"
    cv2.imwrite(str(crop_dir / name), crop)
    cv2.imwrite(str(mask_dir / name), detection.mask)
    boxed = crop.copy()
    color = (0, 255, 0) if detection.detected else (0, 0, 255)
    for x, y, width, height in detection.boxes:
        cv2.rectangle(boxed, (x, y), (x + width, y + height), color, 2)
    cv2.imwrite(str(boxes_dir / name), boxed)


def compact_text_mask(mask: np.ndarray | None, size: tuple[int, int] = (96, 32)) -> np.ndarray | None:
    if mask is None or mask.size == 0:
        return None
    small = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
    return small > 0


def compare_text_masks(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    intersection = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return intersection / union


def extract_subtitle_crops(
    video_path: Path,
    temp_dir: Path,
    sample_rate: float,
    subtitle_area_ratio: float,
    crop_region: CropRegion | None = None,
    detect_subtitle_changes: bool = False,
    change_threshold: float = 7.5,
    start_time: float | None = None,
    end_time: float | None = None,
    subtitle_detect_mode: str = "hybrid",
    debug_subtitle_detection: bool = False,
    ocr_min_gap: float = 2.0,
    subtitle_mask_similarity: float = 0.90,
    use_text_bbox_crop: bool = True,
    legacy_white_text_detection: bool = False,
) -> tuple[VideoInfo, list[FrameCrop], ExtractionStats]:
    if sample_rate <= 0:
        raise ValueError("sample_rate musí být větší než 0.")
    if subtitle_detect_mode not in {"diff", "white-text", "hybrid"}:
        raise ValueError("--subtitle-detect-mode musí být diff, white-text nebo hybrid.")

    if ocr_min_gap < 0:
        raise ValueError("--ocr-min-gap nesmi byt zaporny.")
    if not 0 <= subtitle_mask_similarity <= 1:
        raise ValueError("--subtitle-mask-similarity musi byt v rozsahu 0.0 az 1.0.")

    temp_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = temp_dir / "debug_detection"
    if debug_subtitle_detection:
        shutil.rmtree(debug_dir, ignore_errors=True)
        debug_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Video nelze otevřít: {video_path}")

    info = get_video_info(video_path)
    if crop_region is None:
        crop_region = CropRegion(x1=0.0, x2=1.0, y1=1.0 - subtitle_area_ratio, y2=1.0)
    crop_region.validate()
    start = start_time if start_time is not None else 0.0
    end = end_time if end_time is not None else info.duration
    if start < 0:
        raise ValueError("--start-time nesmí být záporný.")
    if end < 0:
        raise ValueError("--end-time nesmí být záporný.")
    if start >= info.duration:
        raise ValueError("--start-time je mimo délku videa.")
    if end <= start:
        raise ValueError("--end-time musí být větší než --start-time.")
    end = min(end, info.duration)
    x1 = int(info.width * crop_region.x1)
    x2 = int(info.width * crop_region.x2)
    y1 = int(info.height * crop_region.y1)
    y2 = int(info.height * crop_region.y2)
    crops: list[FrameCrop] = []
    current_time = start
    index = 0
    previous_small = None
    skipped = 0
    sampled_frames_total = 0
    skipped_without_subtitles = 0
    white_text_frames = 0
    skipped_same_subtitle_frames = 0
    previous_image_path: Path | None = None
    active_white_subtitle = False
    last_ocr_time: float | None = None
    last_ocr_mask: np.ndarray | None = None

    logging.info(
        "Video: %.2f FPS, %dx%d, délka %.1f s. Extrahuji snímky po %.2f s.",
        info.fps,
        info.width,
        info.height,
        info.duration,
        sample_rate,
    )
    if start > 0 or end < info.duration:
        logging.info("Testovací rozsah videa: %.1f s až %.1f s.", start, end)
    logging.info("OCR crop oblast: x %.2f-%.2f, y %.2f-%.2f.", crop_region.x1, crop_region.x2, crop_region.y1, crop_region.y2)

    while current_time <= end:
        sampled_frames_total += 1
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        crop = frame[y1:y2, x1:x2]
        diff_score = 1.0
        changed = True
        current_small = None
        if subtitle_detect_mode in {"diff", "hybrid"} or detect_subtitle_changes:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            current_small = cv2.resize(gray, (160, 40), interpolation=cv2.INTER_AREA)
            if previous_small is not None:
                diff_score = float(cv2.absdiff(current_small, previous_small).mean())
                if subtitle_detect_mode == "diff":
                    changed = diff_score >= change_threshold
        detection = WhiteTextDetection(False, 0, 0.0, np.zeros(crop.shape[:2], dtype=np.uint8), ())
        if subtitle_detect_mode in {"white-text", "hybrid"}:
            detection = detect_white_subtitle_text(crop, legacy=legacy_white_text_detection)
            if detection.detected:
                white_text_frames += 1
            if debug_subtitle_detection:
                save_detection_debug(debug_dir, index, crop, detection)
            if not detection.detected:
                skipped_without_subtitles += 1
                active_white_subtitle = False
                last_ocr_mask = None
                previous_small = None
                previous_image_path = None
                current_time += sample_rate
                index += 1
                continue
            current_mask = compact_text_mask(detection.mask)
            mask_similarity = compare_text_masks(last_ocr_mask, current_mask)
            strong_image_change = active_white_subtitle and diff_score >= change_threshold
            same_subtitle = active_white_subtitle and mask_similarity > subtitle_mask_similarity and not strong_image_change
            strong_mask_change = active_white_subtitle and mask_similarity < subtitle_mask_similarity * 0.75
            min_gap_elapsed = last_ocr_time is None or current_time - last_ocr_time >= ocr_min_gap
            if same_subtitle or (active_white_subtitle and not strong_mask_change and not strong_image_change and not min_gap_elapsed and previous_image_path is not None):
                skipped_same_subtitle_frames += 1
                if crops:
                    crops[-1] = replace(crops[-1], end_time=current_time + sample_rate)
                current_time += sample_rate
                index += 1
                continue
            changed = True
            active_white_subtitle = True
            last_ocr_time = current_time
            last_ocr_mask = current_mask
        if subtitle_detect_mode in {"diff", "hybrid"} or detect_subtitle_changes:
            previous_small = current_small
            if not changed and previous_image_path is not None:
                skipped += 1
                if crops:
                    crops[-1] = replace(crops[-1], end_time=current_time + sample_rate)
                current_time += sample_rate
                index += 1
                continue
        image_path = temp_dir / f"crop_{index:06d}.png"
        ocr_crop, text_crop_bbox = crop_to_detected_text(crop, detection) if detection.detected and use_text_bbox_crop else (crop, None)
        cv2.imwrite(str(image_path), ocr_crop)
        bbox_y1 = bbox_y2 = 0.0
        if text_crop_bbox:
            _, local_y1, _, local_y2 = text_crop_bbox
            bbox_y1 = (y1 + local_y1) / max(1, info.height)
            bbox_y2 = (y1 + local_y2) / max(1, info.height)
        elif detection.boxes:
            local_y1 = min(box_y for _, box_y, _, _ in detection.boxes)
            local_y2 = max(box_y + box_h for _, box_y, _, box_h in detection.boxes)
            bbox_y1 = (y1 + local_y1) / max(1, info.height)
            bbox_y2 = (y1 + local_y2) / max(1, info.height)
        crops.append(FrameCrop(
            time=current_time,
            image_path=image_path,
            end_time=current_time + sample_rate,
            changed=changed,
            diff_score=diff_score,
            has_subtitle=True,
            white_text_detected=detection.detected,
            text_component_count=detection.component_count,
            text_bbox_area=detection.bbox_area,
            text_bbox_y1=bbox_y1,
            text_bbox_y2=bbox_y2,
            mask_similarity=compare_text_masks(last_ocr_mask, compact_text_mask(detection.mask)) if detection.detected else 0.0,
        ))
        previous_image_path = image_path
        current_time += sample_rate
        index += 1

    cap.release()
    if subtitle_detect_mode in {"diff", "hybrid"} or detect_subtitle_changes:
        logging.info("Subtitle change detection přeskočil %d téměř stejných výřezů.", skipped)
    if subtitle_detect_mode in {"white-text", "hybrid"}:
        logging.info(
            "White-text subtitle detection: %d framů s titulky, %d framů přeskočeno bez titulků.",
            white_text_frames,
            skipped_without_subtitles,
        )
    logging.info("Vyextrahováno %d OCR výřezů.", len(crops))
    if subtitle_detect_mode in {"white-text", "hybrid"}:
        logging.info("OCR timing optimization skipped %d same-subtitle frames.", skipped_same_subtitle_frames)
    stats = ExtractionStats(
        sampled_frames_total=sampled_frames_total,
        white_text_frames=white_text_frames,
        skipped_without_subtitles=skipped_without_subtitles,
        skipped_same_subtitle_frames=skipped_same_subtitle_frames,
        actual_ocr_calls=sum(1 for crop in crops if crop.has_subtitle),
    )
    logging.info("Vyextrahovano %d framu celkem.", stats.sampled_frames_total)
    logging.info("White-text frames: %d.", stats.white_text_frames)
    logging.info("Skipped without subtitles: %d.", stats.skipped_without_subtitles)
    logging.info("Skipped same subtitle frames: %d.", stats.skipped_same_subtitle_frames)
    logging.info("Actual OCR calls: %d.", stats.actual_ocr_calls)
    if stats.skipped_same_subtitle_frames > 0 and stats.actual_ocr_calls >= stats.white_text_frames:
        logging.error(
            "OCR timing sanity check failed: skipped_same_subtitle_frames=%d, actual_ocr_calls=%d, white_text_frames=%d.",
            stats.skipped_same_subtitle_frames,
            stats.actual_ocr_calls,
            stats.white_text_frames,
        )
    return info, crops, stats
