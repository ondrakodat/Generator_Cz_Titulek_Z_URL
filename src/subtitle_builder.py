from __future__ import annotations

import logging
from datetime import timedelta
from difflib import SequenceMatcher
from pathlib import Path

import srt

from src.ocr_reader import analyze_ocr_noise, normalized_for_merge, score_ocr_candidate


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
