"""LangGraph orchestration for one video. Nodes exchange artifact paths."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import TypedDict

import httpx
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from .media import (detect_scenes, download_video, extract_subtitles, make_audio_chunks,
                    probe, sample_video, select_keyframes)
from .models import Bailian
from .ocr import extract_ocr, parse_srt
from .sources import resolve_video_source
from .storage import read_json, write_json


class VideoState(TypedDict, total=False):
    raw_input: str
    source_url: str
    download_url: str
    platform: str
    local_file: str
    run_dir: str
    source_manifest: str
    media_manifest: str
    probe_manifest: str
    asr_manifest: str
    scene_manifest: str
    ocr_manifest: str
    visual_manifest: str
    evidence_manifest: str
    analysis_manifest: str
    report_path: str


def run_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def _cached(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _transient_model_error(exc: Exception) -> bool:
    return isinstance(exc, (httpx.NetworkError, httpx.TimeoutException,
                            APIConnectionError, APITimeoutError, InternalServerError, RateLimitError))


def _verify_last_asr(analysis: dict, evidence: list[dict]) -> dict:
    """Reject claims that an earlier chunk boundary is the end of recognized speech."""
    asr = sorted((item for item in evidence if item.get("source") == "asr"),
                 key=lambda item: item["end_ms"])
    if not asr or not asr[-1]["text"].endswith(("。", "！", "？", "!", "?")):
        return analysis
    final = asr[-1]
    result = {**analysis, "structure": [dict(part) for part in analysis["structure"]]}
    result["uncertainty"] = [item for item in analysis.get("uncertainty", [])
                             if not ("ASR" in item and "截断" in item)]
    last_part = result["structure"][-1] if result["structure"] else None
    if last_part and "ASR" in last_part["description"] and "截断" in last_part["description"]:
        text = last_part["description"]
        position = text.find("ASR")
        last_part["description"] = (text[:position].rstrip("；，。 ") +
                                    f"；ASR 末段 {final['start_ms']}–{final['end_ms']}ms 为“{final['text']}”。")
        last_part["evidence_ids"] = list(dict.fromkeys(last_part.get("evidence_ids", []) + [final["id"]]))
        result["evidence_ids"] = list(dict.fromkeys(result.get("evidence_ids", []) + [final["id"]]))
    return result


def _needs_summary_polish(analysis: dict) -> bool:
    return (len(analysis.get("topic", "").strip()) < 12
            or len(analysis.get("summary", "").strip()) < 30
            or any("”“" in analysis.get(key, "") for key in ("topic", "story_or_viewpoint", "summary"))
            or not analysis.get("summary", "").rstrip().endswith(("。", "！", "？", "!", "?")))


def _info_json(media_path: Path) -> dict:
    path = media_path.with_suffix(".info.json")
    if not path.is_file():
        return {}
    data = read_json(path)
    return {key: data.get(key) for key in ("id", "title", "description", "uploader", "duration", "webpage_url") if data.get(key) is not None}


def _one_per_scene(candidates: list[dict], limit: int = 24) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for item in candidates:
        groups.setdefault(item["scene_id"], []).append(item)
    frames = sorted((max(group, key=lambda item: item["quality"]) for group in groups.values()),
                    key=lambda item: item["timestamp_ms"])
    if len(frames) <= limit:
        return frames
    indexes = {round(i * (len(frames) - 1) / (limit - 1)) for i in range(limit)}
    return [frame for index, frame in enumerate(frames) if index in indexes]


def _opening_frame(candidates: list[dict], selected: list[dict]) -> dict | None:
    if not candidates:
        return None
    opening = min(candidates, key=lambda item: item["timestamp_ms"])
    if opening["timestamp_ms"] <= 1500 and all(item["id"] != opening["id"] for item in selected):
        return opening
    return None


def _motion_clips(media_path: str, scene_data: dict, run_dir: Path) -> list[dict]:
    """Check longer action shots and dense-cut windows with valid video duration."""
    import imagehash
    from PIL import Image
    from .media import command

    grouped: dict[str, list[dict]] = {}
    for item in scene_data["candidates"]:
        grouped.setdefault(item["scene_id"], []).append(item)
    intervals: list[tuple[int, int, int]] = []
    for scene in scene_data["scenes"]:
        frames = grouped.get(scene["id"], [])
        length = scene["end_ms"] - scene["start_ms"]
        if len(frames) < 2 or length < 4000:
            continue
        hashes = [imagehash.phash(Image.open(frame["path"])) for frame in frames]
        if length <= 10000:
            distance = hashes[0] - hashes[-1]
            if distance >= 14:
                intervals.append((100 + distance, scene["start_ms"], scene["end_ms"]))
        else:
            # A continuous action shot can exceed the model's useful clip length.
            # Rank short windows by actual frame change instead of skipping it.
            for index in range(len(frames) - 1):
                distance = hashes[index] - hashes[index + 1]
                if distance < 14:
                    continue
                midpoint = (frames[index]["timestamp_ms"] + frames[index + 1]["timestamp_ms"]) // 2
                start = max(scene["start_ms"], min(midpoint - 2250, scene["end_ms"] - 4500))
                intervals.append((100 + distance, start, start + 4500))
    duration_ms = scene_data["scenes"][-1]["end_ms"]
    if len(scene_data["scenes"]) >= 6:
        for start in range(0, max(1, duration_ms - 4500), 2000):
            end = min(duration_ms, start + 4500)
            if end - start < 4000:
                continue
            cuts = sum(start < scene["start_ms"] < end for scene in scene_data["scenes"])
            if cuts >= 2:
                intervals.append((cuts, start, end))
    clips: list[dict] = []
    for _, start, end in sorted(intervals, reverse=True):
        if any(start < item["end_ms"] and end > item["start_ms"] for item in clips):
            continue
        path = run_dir / "clips" / f"{start}_{end}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        command(["ffmpeg", "-y", "-v", "error", "-ss", str(start / 1000),
                 "-i", media_path, "-t", str((end - start) / 1000),
                 "-an", "-vf", "fps=2,scale=640:-2", "-c:v", "libx264", "-b:v", "500k", str(path)], timeout=120)
        clips.append({"path": str(path), "start_ms": start, "end_ms": end})
        if len(clips) == 3:
            break
    return clips


def build_graph(model: Bailian, chrome_profile: str | None = None, interactive_browser: bool = False):
    def resolve_source(state: VideoState) -> dict:
        run_dir = Path(state["run_dir"])
        manifest = run_dir / "source.json"
        if _cached(manifest):
            saved = read_json(manifest)
            if "platform" not in saved:
                saved["platform"] = "local" if state.get("local_file") else "douyin"
            return {"source_url": saved["source_url"],
                    "download_url": saved.get("download_url", saved["source_url"]),
                    "platform": saved["platform"], "source_manifest": str(manifest)}
        if state.get("local_file"):
            source_url = str(Path(state["local_file"]).resolve())
            download_url = source_url
            platform = "local"
            kind = "local-file"
        else:
            raw = state.get("raw_input") or state.get("source_url", "")
            resolved = resolve_video_source(raw, chrome_profile, interactive_browser)
            source_url, download_url = resolved.source_url, resolved.download_url
            platform, kind = resolved.platform, resolved.input_kind
        record = {"source_url": source_url, "download_url": download_url,
                  "platform": platform, "input_kind": kind}
        return {"source_url": source_url, "download_url": download_url,
                "platform": platform, "source_manifest": write_json(manifest, record)}

    def download(state: VideoState) -> dict:
        run_dir = Path(state["run_dir"])
        manifest = run_dir / "media.json"
        if _cached(manifest) and Path(read_json(manifest)["media_path"]).is_file():
            return {"media_manifest": str(manifest)}
        if state.get("local_file"):
            original = Path(state["local_file"]).resolve()
            if not original.is_file():
                raise FileNotFoundError(original)
            destination = run_dir / "media" / original.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(original, destination)
            record = {"source_url": str(original), "media_path": str(destination),
                      "download_method": "local-file", "platform": "local"}
        else:
            record = download_video(state.get("download_url", state["source_url"]), run_dir,
                                    chrome_profile, interactive_browser,
                                    platform=state.get("platform", "douyin"), source_url=state["source_url"])
        return {"media_manifest": write_json(manifest, record)}

    def inspect(state: VideoState) -> dict:
        run_dir = Path(state["run_dir"])
        manifest = run_dir / "probe.json"
        if _cached(manifest):
            return {"probe_manifest": str(manifest)}
        media_record = read_json(state["media_manifest"])
        media_path = Path(media_record["media_path"])
        details = probe(media_path)
        if media_record.get("download_method") == "chrome-media-url+yt-dlp":
            details["metadata"] = {"id": media_record["source_url"].rstrip("/").split("/")[-1],
                                   "title": media_record.get("page_title", ""),
                                   "description": media_record.get("page_description", ""),
                                   "webpage_url": media_record["source_url"]}
        else:
            details["metadata"] = _info_json(media_path)
            if details["metadata"]:
                details["metadata"]["webpage_url"] = media_record["source_url"]
        details["subtitle_paths"] = extract_subtitles(str(media_path), run_dir, details["subtitle_streams"])
        return {"probe_manifest": write_json(manifest, details)}

    def audio(state: VideoState) -> dict:
        run_dir = Path(state["run_dir"])
        manifest = run_dir / "asr.json"
        if _cached(manifest):
            return {"asr_manifest": str(manifest)}
        media = read_json(state["media_manifest"])
        info = read_json(state["probe_manifest"])
        chunks = make_audio_chunks(media["media_path"], run_dir, info["duration_ms"], info["has_audio"])
        segments = model.transcribe(chunks) if chunks else []
        return {"asr_manifest": write_json(manifest, {"segments": segments, "chunks": chunks})}

    def scenes(state: VideoState) -> dict:
        run_dir = Path(state["run_dir"])
        manifest = run_dir / "scenes.json"
        if _cached(manifest):
            return {"scene_manifest": str(manifest)}
        media = read_json(state["media_manifest"])
        info = read_json(state["probe_manifest"])
        boundaries = detect_scenes(media["media_path"], info["duration_ms"])
        return {"scene_manifest": write_json(manifest, sample_video(media["media_path"], run_dir, boundaries, info["duration_ms"]))}

    def ocr(state: VideoState) -> dict:
        run_dir = Path(state["run_dir"])
        manifest = run_dir / "ocr.json"
        if _cached(manifest):
            return {"ocr_manifest": str(manifest)}
        scene_data = read_json(state["scene_manifest"])
        info = read_json(state["probe_manifest"])
        spans = extract_ocr(scene_data["ocr_samples"])
        subtitles = parse_srt(info["subtitle_paths"])
        return {"ocr_manifest": write_json(manifest, {"spans": spans, "subtitles": subtitles})}

    def vision(state: VideoState) -> dict:
        run_dir = Path(state["run_dir"])
        manifest = run_dir / "visual.json"
        if _cached(manifest):
            return {"visual_manifest": str(manifest)}
        scene_data = read_json(state["scene_manifest"])
        frames = _one_per_scene(scene_data["candidates"])
        observations = model.describe_frames(frames)
        opening = _opening_frame(scene_data["candidates"], frames)
        if opening:
            observations.extend(model.describe_frames([opening]))
            observations.sort(key=lambda item: item["timestamp_ms"])
        media_path = read_json(state["media_manifest"])["media_path"]
        clips = []
        for item in _motion_clips(media_path, scene_data, run_dir):
            clip_cache = run_dir / "clip_descriptions" / (Path(item["path"]).stem + ".json")
            if _cached(clip_cache):
                clips.append(read_json(clip_cache))
            else:
                described = model.describe_clip(item["path"], item["start_ms"], item["end_ms"])
                write_json(clip_cache, described)
                clips.append(described)
        return {"visual_manifest": write_json(manifest, {"observations": observations, "motion_clips": clips})}

    def fuse(state: VideoState) -> dict:
        run_dir = Path(state["run_dir"])
        manifest = run_dir / "evidence.json"
        if _cached(manifest):
            return {"evidence_manifest": str(manifest)}
        asr = read_json(state["asr_manifest"])["segments"]
        ocr_data = read_json(state["ocr_manifest"])
        visual_data = read_json(state["visual_manifest"])
        scene_data = read_json(state["scene_manifest"])
        info = read_json(state["probe_manifest"])
        evidence = list(asr) + ocr_data["spans"] + ocr_data["subtitles"]
        scenes_by_id = {item["id"]: item for item in scene_data["scenes"]}
        for item in visual_data["observations"]:
            scene = scenes_by_id[item["scene_id"]]
            evidence.append({"id": item["id"], "start_ms": scene["start_ms"], "end_ms": scene["end_ms"],
                             "text": item["description"], "style": item["style"],
                             "visible_text": item["visible_text"], "source": "visual"})
        for item in visual_data["motion_clips"]:
            evidence.append({"id": item["id"], "start_ms": item["start_ms"], "end_ms": item["end_ms"],
                             "text": item["description"], "source": "video_clip"})
        for field in ("title", "description"):
            value = info["metadata"].get(field)
            if value:
                evidence.append({"id": f"meta_{field}", "start_ms": 0, "end_ms": 0,
                                 "text": str(value), "source": "publisher_metadata"})
        evidence.sort(key=lambda item: (item["start_ms"], item["id"]))
        keyframes = select_keyframes(scene_data["candidates"], ocr_data["spans"], asr)
        return {"evidence_manifest": write_json(manifest, {"evidence": evidence, "keyframes": keyframes})}

    def understand(state: VideoState) -> dict:
        run_dir = Path(state["run_dir"])
        manifest = run_dir / "analysis.json"
        if _cached(manifest):
            return {"analysis_manifest": str(manifest)}
        info = read_json(state["probe_manifest"])
        evidence_data = read_json(state["evidence_manifest"])
        analysis = model.summarize(evidence_data["evidence"], info["duration_ms"], info["metadata"])
        analysis = _verify_last_asr(analysis, evidence_data["evidence"])
        conflict = any("ASR" in text and ("OCR" in text or "字幕" in text or "画面" in text)
                       for text in analysis.get("uncertainty", []))
        if conflict and hasattr(model, "audit_analysis"):
            try:
                analysis = model.audit_analysis(analysis, evidence_data["evidence"], info["duration_ms"])
                analysis = _verify_last_asr(analysis, evidence_data["evidence"])
                analysis["audit_status"] = "accepted"
            except Exception as exc:
                analysis["audit_status"] = f"rejected:{type(exc).__name__}"
        if _needs_summary_polish(analysis) and hasattr(model, "polish_summary"):
            try:
                analysis = model.polish_summary(analysis, evidence_data["evidence"], info["metadata"])
                analysis["polish_status"] = "accepted"
            except Exception as exc:
                analysis["polish_status"] = f"rejected:{type(exc).__name__}"
        analysis.update({"source_url": read_json(state["media_manifest"])["source_url"],
                         "platform": read_json(state["media_manifest"]).get("platform", "douyin"),
                         "duration_ms": info["duration_ms"], "metadata": info["metadata"],
                         "keyframes": evidence_data["keyframes"],
                         "artifact_paths": {"asr": state["asr_manifest"], "ocr": state["ocr_manifest"],
                                            "scenes": state["scene_manifest"], "visual": state["visual_manifest"],
                                            "evidence": state["evidence_manifest"]}})
        return {"analysis_manifest": write_json(manifest, analysis)}

    def report(state: VideoState) -> dict:
        from .report import render_report

        path = Path(state["run_dir"]) / "report.md"
        analysis = read_json(state["analysis_manifest"])
        render_report(analysis, path)
        return {"report_path": str(path)}

    builder = StateGraph(VideoState)
    for name, node in (("resolve_source", resolve_source), ("download", download), ("inspect", inspect), ("audio", audio),
                       ("scenes", scenes), ("ocr", ocr), ("vision", vision),
                       ("fuse", fuse), ("understand", understand), ("report", report)):
        policy = RetryPolicy(max_attempts=2, retry_on=_transient_model_error) if name in {"vision", "understand"} else None
        builder.add_node(name, node, retry_policy=policy)
    builder.add_edge(START, "resolve_source")
    builder.add_edge("resolve_source", "download")
    builder.add_edge("download", "inspect")
    builder.add_edge("inspect", "audio")
    builder.add_edge("inspect", "scenes")
    builder.add_edge("scenes", "ocr")
    builder.add_edge("scenes", "vision")
    builder.add_edge(["audio", "ocr", "vision"], "fuse")
    builder.add_edge("fuse", "understand")
    builder.add_edge("understand", "report")
    builder.add_edge("report", END)
    return builder
