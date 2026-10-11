"""Publish a small, portable result bundle from one completed graph run."""

from __future__ import annotations

import copy
import shutil
import tempfile
from pathlib import Path

from .report import render_report
from .storage import read_json, write_json


def export_results(analysis_path: str | Path, result_dir: str | Path) -> Path:
    """Write report, self-contained JSON and keyframes to a fresh directory.

    Existing nonempty directories are left alone, so a new video never replaces
    a prior result or an unrelated user file.
    """
    source = read_json(analysis_path)
    destination = Path(result_dir).resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"结果目录已存在且非空：{destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        report_data = copy.deepcopy(source)
        frame_dir = temporary / "keyframes"
        frame_dir.mkdir()
        for index, frame in enumerate(report_data.get("keyframes", []), start=1):
            original = Path(frame["path"])
            if not original.is_file():
                raise FileNotFoundError(original)
            copied = frame_dir / f"{index:02d}_{original.name}"
            shutil.copy2(original, copied)
            frame["path"] = str(copied)

        render_report(report_data, temporary / "report.md", compact=True)

        public = copy.deepcopy(report_data)
        paths = public.pop("artifact_paths")
        public["transcript"] = read_json(paths["asr"])["segments"]
        ocr = read_json(paths["ocr"])
        public["ocr_spans"] = ocr["spans"]
        public["subtitles"] = ocr["subtitles"]
        visual = read_json(paths["visual"])
        public["visual_observations"] = visual["observations"]
        public["motion_clips"] = visual["motion_clips"]
        public["evidence"] = read_json(paths["evidence"])["evidence"]
        # OCR sample images stay in the internal cache; their absolute paths
        # would be broken if this result bundle is moved elsewhere.
        for item in public["ocr_spans"] + public["evidence"]:
            item.pop("source_frame", None)
        for frame in public.get("keyframes", []):
            frame["path"] = str(Path("keyframes") / Path(frame["path"]).name)
        write_json(temporary / "analysis.json", public)

        if destination.exists():
            destination.rmdir()  # Only an empty directory can reach this point.
        temporary.rename(destination)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
