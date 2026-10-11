"""Local OCR and subtitle extraction with source timestamps."""

from __future__ import annotations

import re
from pathlib import Path


class OCRError(RuntimeError):
    pass


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def extract_ocr(samples: list[dict], min_score: float = 0.50) -> list[dict]:
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise OCRError("未安装 PaddleOCR；请运行 pip install -e '.[ocr]' 后重试") from exc
    engine = PaddleOCR(lang="ch", use_doc_orientation_classify=False,
                       use_doc_unwarping=False, use_textline_orientation=False)
    tracks: list[dict] = []
    active: dict[tuple[str, str], int] = {}
    for sample_index, sample in enumerate(samples, start=1):
        try:
            predictions = engine.predict(sample["path"])
        except Exception as exc:
            raise OCRError(f"PaddleOCR 无法处理 {sample['path']}: {type(exc).__name__}") from exc
        present: set[tuple[str, str]] = set()
        for prediction in predictions:
            raw = prediction.json
            data = raw.get("res", raw) if isinstance(raw, dict) else {}
            texts = data.get("rec_texts", [])
            scores = data.get("rec_scores", [])
            boxes = data.get("rec_boxes", [])
            for index, text in enumerate(texts):
                text = str(text).strip()
                score = float(scores[index]) if index < len(scores) else 0.0
                if not text or score < min_score:
                    continue
                box = [int(value) for value in boxes[index]] if index < len(boxes) else []
                image_height = data.get("input_img_shape", [0, 0])[0] or 0
                zone = "bottom" if box and image_height and box[1] / image_height >= 0.65 else "other"
                key = (_normalize(text), zone)
                present.add(key)
                previous_index = active.get(key)
                if previous_index is not None and sample["timestamp_ms"] - tracks[previous_index]["end_ms"] <= 1000:
                    tracks[previous_index]["end_ms"] = sample["timestamp_ms"] + 500
                    tracks[previous_index]["confidence"] = max(score, tracks[previous_index]["confidence"])
                else:
                    tracks.append({"id": f"ocr_{len(tracks):04d}", "start_ms": sample["timestamp_ms"],
                                   "end_ms": sample["timestamp_ms"] + 500, "text": text,
                                   "bbox": box, "zone": zone, "confidence": score,
                                   "source_frame": sample["path"], "source": "paddleocr"})
                    active[key] = len(tracks) - 1
        active = {key: value for key, value in active.items() if key in present}
        if sample_index % 10 == 0 or sample_index == len(samples):
            print(f"OCR 进度：{sample_index}/{len(samples)} 帧", flush=True)
    return tracks


def _srt_ms(value: str) -> int:
    hours, minutes, seconds, milliseconds = [int(part) for part in re.split(r"[:,]", value)]
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds


def parse_srt(paths: list[str]) -> list[dict]:
    entries: list[dict] = []
    timing = re.compile(r"(\d\d:\d\d:\d\d,\d{3})\s*-->\s*(\d\d:\d\d:\d\d,\d{3})")
    for path in paths:
        for block in re.split(r"\n\s*\n", Path(path).read_text(encoding="utf-8-sig", errors="replace")):
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            match_index = next((i for i, line in enumerate(lines) if timing.search(line)), None)
            if match_index is None:
                continue
            match = timing.search(lines[match_index])
            assert match
            text = " ".join(lines[match_index + 1:]).strip()
            if text:
                entries.append({"id": f"subtitle_{len(entries):04d}",
                                "start_ms": _srt_ms(match.group(1)), "end_ms": _srt_ms(match.group(2)),
                                "text": text, "source": "subtitle_track"})
    return entries
