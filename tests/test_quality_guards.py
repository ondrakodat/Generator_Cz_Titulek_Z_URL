from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("requests")

from src.ocr_reader import confidence_histogram
from src.processor import parse_report_int, write_benchmark_html
from src.subtitle_builder import build_subtitle_blocks


def test_subtitle_blocks_keep_confidence():
    blocks = build_subtitle_blocks(
        [
            {"time": 1.0, "end": 2.0, "text": "Привет", "score": 10.0, "confidence": 0.42},
            {"time": 3.0, "end": 4.0, "text": "Пойдем домой", "score": 12.0, "confidence": 0.81},
        ],
        sample_rate=0.5,
    )

    assert blocks[0]["confidence"] == 0.42
    assert blocks[1]["confidence"] == 0.81


def test_confidence_histogram_bins_values():
    assert confidence_histogram([0.1, 0.25, 0.45, 0.65, 0.95]) == {
        "0.0-0.2": 1,
        "0.2-0.4": 1,
        "0.4-0.6": 1,
        "0.6-0.8": 1,
        "0.8-1.0": 1,
    }


def test_parse_report_int_and_write_html(tmp_path: Path):
    report_path = tmp_path / "ocr_report.txt"
    report_path.write_text("timeout_frames: 3\nfailed_frames: 2\n", encoding="utf-8")

    assert parse_report_int(report_path, "timeout_frames") == 3
    assert parse_report_int(report_path, "failed_frames") == 2
    assert parse_report_int(report_path, "missing") == 0

    html_path = tmp_path / "benchmark_report.html"
    write_benchmark_html(
        {
            "recall": {
                "ocr_candidates": 100,
                "ocr_hits": 90,
                "subtitle_blocks": 80,
                "average_confidence": 0.7,
                "timeout_frames": 1,
                "failed_frames": 2,
                "processing_speed_candidates_per_second": 3.5,
            }
        },
        html_path,
    )

    html = html_path.read_text(encoding="utf-8")
    assert "OCR benchmark" in html
    assert "recall" in html
    assert "100" in html
