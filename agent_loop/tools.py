"""Closed, local video tool registry used by the model-directed loop.

The existing fixed workflow remains untouched. These adapters reuse its media,
OCR, model and report functions while keeping each action independently callable.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from douyin_review.export import export_results
from douyin_review.media import (command, detect_scenes, download_video, extract_subtitles,
                                 make_audio_chunks, probe, sample_video, select_keyframes)
from douyin_review.models import Bailian
from douyin_review.ocr import extract_ocr, parse_srt
from douyin_review.report import render_report
from douyin_review.sources import resolve_video_source
from douyin_review.storage import read_json, write_json
from douyin_review.workflow import (_info_json, _motion_clips, _needs_summary_polish,
                                    _one_per_scene, _opening_frame, _verify_last_asr)

from .state import AgentState
from .quality import incomplete_structure_indices, repair_structure_descriptions


ARTIFACTS = {
    "source": "source.json", "media": "media.json", "probe": "probe.json",
    "asr": "asr.json", "scenes": "scenes.json", "ocr": "ocr.json",
    "visual": "visual.json", "evidence": "evidence.json",
    "agent_evidence": "agent_evidence.json", "analysis": "analysis.json",
    "report": "report.md", "review": "review.json",
}


class AgentToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolOutcome:
    observation: str
    review_dirty: bool | None = None


class AgentTools:
    def __init__(self, model: Bailian):
        self.model = model

    @staticmethod
    def artifact_paths(run_dir: str | Path) -> dict[str, str]:
        root = Path(run_dir)
        return {name: str(root / filename) for name, filename in ARTIFACTS.items()
                if (root / filename).is_file() and (root / filename).stat().st_size > 0}

    @staticmethod
    def _path(state: AgentState, name: str) -> Path:
        return Path(state["run_dir"]) / ARTIFACTS[name]

    def _read(self, state: AgentState, name: str) -> dict:
        path = self._path(state, name)
        if not path.is_file():
            raise AgentToolError(f"缺少 {name} 阶段产物；请先运行其前置工具")
        return read_json(path)

    def execute(self, action: str, state: AgentState, args: dict[str, Any]) -> ToolOutcome:
        handler = getattr(self, f"_do_{action}", None)
        if handler is None:
            raise AgentToolError(f"未注册的工具：{action}")
        return handler(state, args)

    def _do_import_workflow_run(self, state: AgentState, args: dict) -> ToolOutcome:
        source = Path(state.get("seed_run_dir", "")).expanduser().resolve()
        if not source.is_dir():
            raise AgentToolError("没有可导入的旧工作流运行目录")
        required = ("source", "media", "probe", "asr", "scenes", "ocr", "visual", "evidence", "analysis")
        for name in required:
            if not (source / ARTIFACTS[name]).is_file():
                raise AgentToolError(f"旧运行缺少 {ARTIFACTS[name]}")
        media = read_json(source / "media.json")
        if not Path(media["media_path"]).is_file():
            raise AgentToolError("旧运行中的原视频已不存在")
        if state.get("raw_input"):
            requested = resolve_video_source(state["raw_input"])
            if requested.source_url != media["source_url"]:
                raise AgentToolError("导入目录对应的视频与当前输入链接不一致")
        target = Path(state["run_dir"])
        target.mkdir(parents=True, exist_ok=True)
        if (target / "source.json").exists() and (target / "report.md").is_file():
            return ToolOutcome("旧工作流产物已经导入；可检查现有分析或进一步复核")
        for name in required:
            shutil.copy2(source / ARTIFACTS[name], target / ARTIFACTS[name])

        def copy_asset(original: str, folder: str) -> str:
            src = Path(original)
            if not src.is_file():
                raise AgentToolError(f"旧运行引用的文件已不存在：{src.name}")
            destination = target / folder / src.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(src, destination)
            return str(destination)

        media = read_json(target / "media.json")
        original_media = Path(media["media_path"])
        media["media_path"] = copy_asset(media["media_path"], "media")
        info_json = original_media.with_suffix(".info.json")
        if info_json.is_file():
            copy_asset(str(info_json), "media")
        write_json(target / "media.json", media)

        probe_data = read_json(target / "probe.json")
        probe_data["subtitle_paths"] = [copy_asset(path, "subtitles")
                                        for path in probe_data.get("subtitle_paths", [])]
        write_json(target / "probe.json", probe_data)

        asr_data = read_json(target / "asr.json")
        for chunk in asr_data.get("chunks", []):
            chunk["path"] = copy_asset(chunk["path"], "audio")
        write_json(target / "asr.json", asr_data)

        mapped_images: dict[str, str] = {}
        scene_data = read_json(target / "scenes.json")
        for field, folder in (("candidates", "candidates"), ("ocr_samples", "ocr_samples")):
            for item in scene_data.get(field, []):
                original = item["path"]
                item["path"] = mapped_images.setdefault(original, copy_asset(original, folder))
        write_json(target / "scenes.json", scene_data)

        for name in ("ocr", "evidence"):
            record = read_json(target / ARTIFACTS[name])
            spans = record.get("spans", []) if name == "ocr" else record.get("evidence", [])
            for item in spans:
                if item.get("source_frame") in mapped_images:
                    item["source_frame"] = mapped_images[item["source_frame"]]
            if name == "evidence":
                for frame in record.get("keyframes", []):
                    original = frame["path"]
                    frame["path"] = mapped_images.get(original) or copy_asset(original, "candidates")
            write_json(target / ARTIFACTS[name], record)

        analysis = read_json(target / "analysis.json")
        for frame in analysis.get("keyframes", []):
            original = frame["path"]
            frame["path"] = mapped_images.get(original) or copy_asset(original, "candidates")
        analysis["artifact_paths"] = {key: str(target / ARTIFACTS[key])
                                      for key in ("asr", "ocr", "scenes", "visual", "evidence")}
        write_json(target / "analysis.json", analysis)
        render_report(analysis, target / "report.md")
        return ToolOutcome(f"已导入 {media.get('platform', 'unknown')} 视频的完整旧工作流产物；"
                           "可直接检查证据、补看片段或完成交付")

    def _do_resolve_source(self, state: AgentState, args: dict) -> ToolOutcome:
        path = self._path(state, "source")
        if path.is_file():
            saved = read_json(path)
            return ToolOutcome(f"来源已解析：{saved['platform']}，{saved['source_url']}")
        if state.get("local_file"):
            local = Path(state["local_file"]).expanduser().resolve()
            if not local.is_file():
                raise AgentToolError("提供的本地视频文件不存在，请向人请求正确路径")
            record = {"source_url": str(local), "download_url": str(local),
                      "platform": "local", "input_kind": "local-file"}
        elif state.get("raw_input"):
            source = resolve_video_source(state["raw_input"], state.get("chrome_profile"),
                                          state.get("interactive_browser", False))
            record = {"source_url": source.source_url, "download_url": source.download_url,
                      "platform": source.platform, "input_kind": source.input_kind}
        else:
            raise AgentToolError("缺少视频链接或本地文件路径，请向人请求输入")
        write_json(path, record)
        return ToolOutcome(f"已解析 {record['platform']} 视频：{record['source_url']}")

    def _do_download(self, state: AgentState, args: dict) -> ToolOutcome:
        path = self._path(state, "media")
        if path.is_file() and Path(read_json(path)["media_path"]).is_file():
            return ToolOutcome("本地原视频已存在，复用 media.json")
        source = self._read(state, "source")
        run_dir = Path(state["run_dir"])
        if source["platform"] == "local":
            original = Path(source["source_url"])
            if not original.is_file():
                raise AgentToolError("本地视频文件已不存在")
            destination = run_dir / "media" / original.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(original, destination)
            record = {"source_url": str(original), "media_path": str(destination),
                      "download_method": "local-file", "platform": "local"}
        else:
            record = download_video(source["download_url"], run_dir,
                                    state.get("chrome_profile"), state.get("interactive_browser", False),
                                    platform=source["platform"], source_url=source["source_url"])
        write_json(path, record)
        return ToolOutcome(f"视频已下载：平台 {record['platform']}；方式 {record['download_method']}")

    def _do_probe(self, state: AgentState, args: dict) -> ToolOutcome:
        path = self._path(state, "probe")
        if path.is_file():
            info = read_json(path)
            return ToolOutcome(f"媒体信息已存在：{info['duration_ms']}ms，有音轨={info['has_audio']}")
        media = self._read(state, "media")
        details = probe(media["media_path"])
        if media.get("download_method") == "chrome-media-url+yt-dlp":
            details["metadata"] = {"id": media["source_url"].rstrip("/").split("/")[-1],
                                   "title": media.get("page_title", ""),
                                   "description": media.get("page_description", ""),
                                   "webpage_url": media["source_url"]}
        else:
            details["metadata"] = _info_json(Path(media["media_path"]))
            if details["metadata"]:
                details["metadata"]["webpage_url"] = media["source_url"]
        details["subtitle_paths"] = extract_subtitles(media["media_path"], Path(state["run_dir"]),
                                                       details["subtitle_streams"])
        write_json(path, details)
        return ToolOutcome(f"媒体检查完成：{details['duration_ms']}ms，"
                           f"{details['width']}×{details['height']}，有音轨={details['has_audio']}")

    def _do_transcribe(self, state: AgentState, args: dict) -> ToolOutcome:
        path = self._path(state, "asr")
        if path.is_file():
            return ToolOutcome(f"语音结果已存在：{len(read_json(path)['segments'])} 条片段")
        media, info = self._read(state, "media"), self._read(state, "probe")
        chunks = make_audio_chunks(media["media_path"], Path(state["run_dir"]),
                                   info["duration_ms"], info["has_audio"])
        segments = self.model.transcribe(chunks) if chunks else []
        write_json(path, {"chunks": chunks, "segments": segments})
        return ToolOutcome(f"ASR 完成：{len(chunks)} 个音频块、{len(segments)} 条转写片段")

    def _do_sample_frames(self, state: AgentState, args: dict) -> ToolOutcome:
        path = self._path(state, "scenes")
        if path.is_file():
            saved = read_json(path)
            return ToolOutcome(f"抽帧结果已存在：{len(saved['scenes'])} 个镜头")
        media, info = self._read(state, "media"), self._read(state, "probe")
        boundaries = detect_scenes(media["media_path"], info["duration_ms"])
        data = sample_video(media["media_path"], Path(state["run_dir"]),
                            boundaries, info["duration_ms"])
        write_json(path, data)
        return ToolOutcome(f"视频抽帧完成：{len(data['scenes'])} 个镜头、"
                           f"{len(data['candidates'])} 张候选帧、{len(data['ocr_samples'])} 张 OCR 采样帧")

    def _do_read_screen_text(self, state: AgentState, args: dict) -> ToolOutcome:
        path = self._path(state, "ocr")
        if path.is_file():
            return ToolOutcome(f"OCR 已存在：{len(read_json(path)['spans'])} 条画面文字")
        scenes, info = self._read(state, "scenes"), self._read(state, "probe")
        spans = extract_ocr(scenes["ocr_samples"])
        subtitles = parse_srt(info["subtitle_paths"])
        write_json(path, {"spans": spans, "subtitles": subtitles})
        return ToolOutcome(f"本地 OCR 完成：{len(spans)} 条文字、{len(subtitles)} 条独立字幕")

    def _do_describe_visuals(self, state: AgentState, args: dict) -> ToolOutcome:
        path = self._path(state, "visual")
        if path.is_file():
            saved = read_json(path)
            return ToolOutcome(f"视觉结果已存在：{len(saved['observations'])} 张代表帧，"
                               f"{len(saved['motion_clips'])} 段动作")
        scenes, media = self._read(state, "scenes"), self._read(state, "media")
        frames = _one_per_scene(scenes["candidates"])
        observations = self.model.describe_frames(frames)
        opening = _opening_frame(scenes["candidates"], frames)
        if opening:
            observations.extend(self.model.describe_frames([opening]))
            observations.sort(key=lambda item: item["timestamp_ms"])
        clips = []
        for item in _motion_clips(media["media_path"], scenes, Path(state["run_dir"])):
            cache = Path(state["run_dir"]) / "clip_descriptions" / (Path(item["path"]).stem + ".json")
            if cache.is_file():
                clips.append(read_json(cache))
            else:
                described = self.model.describe_clip(item["path"], item["start_ms"], item["end_ms"])
                write_json(cache, described)
                clips.append(described)
        write_json(path, {"observations": observations, "motion_clips": clips})
        return ToolOutcome(f"视觉理解完成：{len(observations)} 条图片观察、{len(clips)} 条短片描述")

    def _do_fuse_evidence(self, state: AgentState, args: dict) -> ToolOutcome:
        path = self._path(state, "evidence")
        if path.is_file():
            saved = read_json(path)
            return ToolOutcome(f"证据已汇合：{len(saved['evidence'])} 条证据、"
                               f"{len(saved['keyframes'])} 张关键帧")
        asr = self._read(state, "asr")["segments"]
        ocr = self._read(state, "ocr")
        visual = self._read(state, "visual")
        scenes = self._read(state, "scenes")
        info = self._read(state, "probe")
        evidence = list(asr) + ocr["spans"] + ocr["subtitles"]
        by_scene = {item["id"]: item for item in scenes["scenes"]}
        for item in visual["observations"]:
            scene = by_scene[item["scene_id"]]
            evidence.append({"id": item["id"], "start_ms": scene["start_ms"],
                             "end_ms": scene["end_ms"], "text": item["description"],
                             "style": item.get("style", ""),
                             "visible_text": item.get("visible_text", ""), "source": "visual"})
        for item in visual["motion_clips"]:
            evidence.append({"id": item["id"], "start_ms": item["start_ms"],
                             "end_ms": item["end_ms"], "text": item["description"],
                             "source": "video_clip"})
        for field in ("title", "description"):
            value = info["metadata"].get(field)
            if value:
                evidence.append({"id": f"meta_{field}", "start_ms": 0, "end_ms": 0,
                                 "text": str(value), "source": "publisher_metadata"})
        evidence.sort(key=lambda item: (item["start_ms"], item["id"]))
        keyframes = select_keyframes(scenes["candidates"], ocr["spans"], asr)
        write_json(path, {"evidence": evidence, "keyframes": keyframes})
        return ToolOutcome(f"证据汇合完成：{len(evidence)} 条时间轴证据、"
                           f"{len(keyframes)} 张关键帧")

    def _review_items(self, state: AgentState) -> list[dict]:
        path = self._path(state, "review")
        return read_json(path)["items"] if path.is_file() else []

    def _save_review(self, state: AgentState, item: dict) -> bool:
        path = self._path(state, "review")
        items = self._review_items(state)
        if any(existing["id"] == item["id"] for existing in items):
            return False
        write_json(path, {"items": items + [item]})
        return True

    def _do_write_analysis(self, state: AgentState, args: dict) -> ToolOutcome:
        path = self._path(state, "analysis")
        if path.is_file() and not state.get("review_dirty", False) and not args.get("force", False):
            saved = read_json(path)
            return ToolOutcome(f"分析结果已存在：{saved.get('topic', '')[:100]}", review_dirty=False)
        base = self._read(state, "evidence")
        info = self._read(state, "probe")
        media = self._read(state, "media")
        reviews = self._review_items(state)
        evidence = list(base["evidence"]) + reviews
        evidence.sort(key=lambda item: (item["start_ms"], item["id"]))
        keyframes = list(base["keyframes"])
        for item in reviews:
            if not item["id"].startswith("agent_visual_"):
                continue
            timestamp_ms = item["start_ms"]
            image = Path(state["run_dir"]) / "review_frames" / f"{timestamp_ms}.jpg"
            if not image.is_file() or any(frame["id"] == item["id"] for frame in keyframes):
                continue
            if len(keyframes) >= 24:
                keyframes.remove(min(keyframes, key=lambda frame: frame.get("quality", 0)))
            keyframes.append({"id": item["id"], "scene_id": "agent_review",
                              "timestamp_ms": timestamp_ms, "path": str(image),
                              "quality": 0, "reason": "智能体补充检查"})
        keyframes.sort(key=lambda frame: frame["timestamp_ms"])
        merged_path = self._path(state, "agent_evidence")
        write_json(merged_path, {"evidence": evidence, "keyframes": keyframes})
        analysis = self.model.summarize(evidence, info["duration_ms"], info["metadata"])
        analysis = _verify_last_asr(analysis, evidence)
        conflict = any("ASR" in text and ("OCR" in text or "字幕" in text or "画面" in text)
                       for text in analysis.get("uncertainty", []))
        if conflict and hasattr(self.model, "audit_analysis"):
            try:
                analysis = self.model.audit_analysis(analysis, evidence, info["duration_ms"])
                analysis = _verify_last_asr(analysis, evidence)
                analysis["audit_status"] = "accepted"
            except Exception as exc:
                analysis["audit_status"] = f"rejected:{type(exc).__name__}"
        if _needs_summary_polish(analysis) and hasattr(self.model, "polish_summary"):
            try:
                analysis = self.model.polish_summary(analysis, evidence, info["metadata"])
                analysis["polish_status"] = "accepted"
            except Exception as exc:
                analysis["polish_status"] = f"rejected:{type(exc).__name__}"
        analysis.update({"source_url": media["source_url"], "platform": media["platform"],
                         "duration_ms": info["duration_ms"], "metadata": info["metadata"],
                         "keyframes": keyframes, "agent_reviews": reviews,
                         "artifact_paths": {key: str(self._path(state, key))
                                            for key in ("asr", "ocr", "scenes", "visual")}
                         | {"evidence": str(merged_path)}})
        if path.is_file():
            revisions = Path(state["run_dir"]) / "revisions"
            revisions.mkdir(parents=True, exist_ok=True)
            number = len(list(revisions.glob("analysis_*.json")))
            shutil.copy2(path, revisions / f"analysis_{number:03d}.json")
        write_json(path, analysis)
        return ToolOutcome(f"结构化分析完成：{analysis.get('topic', '')[:120]}；"
                           f"不确定项 {len(analysis.get('uncertainty', []))} 条",
                           review_dirty=False)

    def _do_repair_summary(self, state: AgentState, args: dict) -> ToolOutcome:
        path = self._path(state, "analysis")
        analysis = self._read(state, "analysis")
        if not _needs_summary_polish(analysis):
            return ToolOutcome("摘要已完整，无需修复")
        if not hasattr(self.model, "polish_summary"):
            raise AgentToolError("当前模型适配器不支持摘要修复")
        info = self._read(state, "probe")
        evidence = read_json(analysis["artifact_paths"]["evidence"])["evidence"]
        last_error: Exception | None = None
        for _ in range(2):
            try:
                repaired = self.model.polish_summary(analysis, evidence, info["metadata"])
                if _needs_summary_polish(repaired):
                    raise AgentToolError("摘要修复后仍有未完成字段")
                repaired["polish_status"] = "accepted:agent_repair"
                revisions = Path(state["run_dir"]) / "revisions"
                revisions.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, revisions / f"summary_before_repair_{len(list(revisions.glob('summary_before_repair_*.json'))):03d}.json")
                write_json(path, repaired)
                return ToolOutcome("摘要修复完成；请重新生成报告")
            except Exception as exc:
                last_error = exc
        raise AgentToolError(f"摘要修复连续失败：{type(last_error).__name__}")

    def _do_repair_structure(self, state: AgentState, args: dict) -> ToolOutcome:
        path = self._path(state, "analysis")
        analysis = self._read(state, "analysis")
        indices = incomplete_structure_indices(analysis)
        if not indices:
            return ToolOutcome("分段描述均为完整句子，无需修复")
        evidence = read_json(analysis["artifact_paths"]["evidence"])["evidence"]
        repaired = repair_structure_descriptions(self.model, analysis, evidence)
        repaired["structure_repair_status"] = "accepted:agent_repair"
        revisions = Path(state["run_dir"]) / "revisions"
        revisions.mkdir(parents=True, exist_ok=True)
        number = len(list(revisions.glob("structure_before_repair_*.json")))
        shutil.copy2(path, revisions / f"structure_before_repair_{number:03d}.json")
        write_json(path, repaired)
        return ToolOutcome(f"已修复 {len(indices)} 个未完成的时间轴段落；请重新生成报告")

    @staticmethod
    def _review_lines(items: list[dict]) -> list[str]:
        if not items:
            return []
        lines = ["", "## 智能体追加检查", ""]
        for item in items:
            lines.append(f"- {item['start_ms']}–{item['end_ms']}ms "
                         f"[{item['source']}] {str(item['text']).replace(chr(10), ' ')[:700]} "
                         f"（{item['id']}）")
        return lines

    def _do_write_report(self, state: AgentState, args: dict) -> ToolOutcome:
        analysis = self._read(state, "analysis")
        if state.get("review_dirty", False):
            raise AgentToolError("新增复核证据后分析已过期，请先运行 write_analysis")
        if incomplete_structure_indices(analysis):
            raise AgentToolError("时间轴段落未写完整；请先运行 repair_structure")
        path = self._path(state, "report")
        render_report(analysis, path)
        extra = self._review_lines(analysis.get("agent_reviews", []))
        if extra:
            with path.open("a", encoding="utf-8") as file:
                file.write("\n".join(extra) + "\n")
        return ToolOutcome(f"报告已写入本地：{path}")

    def _do_inspect_evidence(self, state: AgentState, args: dict) -> ToolOutcome:
        path = self._path(state, "analysis")
        if not path.is_file():
            missing = [name for name in ("source", "media", "probe", "asr", "scenes",
                                         "ocr", "visual", "evidence", "analysis")
                       if not self._path(state, name).is_file()]
            return ToolOutcome("分析未完成；仍缺少阶段：" + ", ".join(missing))
        analysis = read_json(path)
        evidence_path = Path(analysis["artifact_paths"]["evidence"])
        evidence = read_json(evidence_path)["evidence"]
        counts: dict[str, int] = {}
        for item in evidence:
            counts[item["source"]] = counts.get(item["source"], 0) + 1
        uncertainties = [str(value)[:220] for value in analysis.get("uncertainty", [])[:5]]
        return ToolOutcome(json.dumps({"topic": analysis.get("topic"),
                                       "summary": str(analysis.get("summary", ""))[:450],
                                       "sources": counts, "uncertainty": uncertainties,
                                       "keyframes": len(analysis.get("keyframes", []))}, ensure_ascii=False))

    def _do_review_frame(self, state: AgentState, args: dict) -> ToolOutcome:
        import cv2
        from douyin_review.media import _shrink_frame

        media, info, scenes = (self._read(state, name) for name in ("media", "probe", "scenes"))
        self._read(state, "evidence")
        try:
            timestamp_ms = int(args["timestamp_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AgentToolError("review_frame 需要整数 timestamp_ms") from exc
        if not 0 <= timestamp_ms < info["duration_ms"]:
            raise AgentToolError("review_frame 时间超出视频范围")
        evidence_id = f"agent_visual_{timestamp_ms}"
        if any(item["id"] == evidence_id for item in self._review_items(state)):
            return ToolOutcome("该时间点已复核，复用已有观察")
        capture = cv2.VideoCapture(media["media_path"])
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
            success, frame = capture.read()
        finally:
            capture.release()
        if not success:
            raise AgentToolError("无法解码指定时间点的视频帧")
        scene = next((s for s in scenes["scenes"]
                      if s["start_ms"] <= timestamp_ms < s["end_ms"]), scenes["scenes"][-1])
        image = Path(state["run_dir"]) / "review_frames" / f"{timestamp_ms}.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image), _shrink_frame(frame))
        observation = self.model.describe_frames([{"id": f"agent_frame_{timestamp_ms}",
                                                   "scene_id": scene["id"],
                                                   "timestamp_ms": timestamp_ms,
                                                   "path": str(image)}])[0]
        item = {"id": evidence_id, "source": "visual", "start_ms": timestamp_ms,
                "end_ms": min(info["duration_ms"], timestamp_ms + 500),
                "text": observation["description"], "style": observation.get("style", ""),
                "visible_text": observation.get("visible_text", "")}
        self._save_review(state, item)
        return ToolOutcome(f"已补看 {timestamp_ms}ms 画面：{item['text'][:180]}", review_dirty=True)

    def _do_review_clip(self, state: AgentState, args: dict) -> ToolOutcome:
        media, info = self._read(state, "media"), self._read(state, "probe")
        self._read(state, "evidence")
        try:
            start_ms, end_ms = int(args["start_ms"]), int(args["end_ms"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AgentToolError("review_clip 需要整数 start_ms 和 end_ms") from exc
        if not (0 <= start_ms < end_ms <= info["duration_ms"]
                and 4000 <= end_ms - start_ms <= 10000):
            raise AgentToolError("review_clip 必须在视频范围内且持续 4–10 秒")
        evidence_id = f"agent_clip_{start_ms}_{end_ms}"
        if any(item["id"] == evidence_id for item in self._review_items(state)):
            return ToolOutcome("该视频片段已复核，复用已有观察")
        clip = Path(state["run_dir"]) / "review_clips" / f"{start_ms}_{end_ms}.mp4"
        clip.parent.mkdir(parents=True, exist_ok=True)
        if not clip.is_file():
            command(["ffmpeg", "-y", "-v", "error", "-ss", str(start_ms / 1000),
                     "-i", media["media_path"], "-t", str((end_ms - start_ms) / 1000),
                     "-an", "-vf", "fps=2,scale=640:-2", "-c:v", "libx264",
                     "-b:v", "500k", str(clip)], timeout=120)
        described = self.model.describe_clip(str(clip), start_ms, end_ms)
        item = {"id": evidence_id, "source": "video_clip", "start_ms": start_ms,
                "end_ms": end_ms, "text": described["description"]}
        self._save_review(state, item)
        return ToolOutcome(f"已补看片段 {start_ms}–{end_ms}ms：{item['text'][:180]}",
                           review_dirty=True)

    def record_human_observation(self, state: AgentState, question: str, answer: str,
                                 args: dict) -> bool:
        info_path = self._path(state, "probe")
        duration = read_json(info_path)["duration_ms"] if info_path.is_file() else 0
        try:
            start_ms = max(0, int(args.get("start_ms", 0)))
            end_ms = max(start_ms, int(args.get("end_ms", duration)))
        except (TypeError, ValueError) as exc:
            raise AgentToolError("人工观察的时间范围必须为整数毫秒") from exc
        if duration:
            start_ms, end_ms = min(start_ms, duration), min(end_ms, duration)
        items = self._review_items(state)
        item = {"id": f"human_note_{len(items):03d}", "source": "human_note",
                "start_ms": start_ms, "end_ms": end_ms,
                "text": f"问题：{question}；人工补充：{answer}"}
        return self._save_review(state, item)

    def publish(self, state: AgentState) -> str:
        analysis = self._read(state, "analysis")
        if _needs_summary_polish(analysis):
            raise AgentToolError("摘要未写完整；请先运行 repair_summary，再运行 write_report")
        if incomplete_structure_indices(analysis):
            raise AgentToolError("时间轴段落未写完整；请先运行 repair_structure，再运行 write_report")
        report = self._path(state, "report")
        if not report.is_file() or state.get("review_dirty", False):
            raise AgentToolError("报告缺失或补充证据尚未重新总结，不能完成")
        if report.stat().st_mtime_ns < self._path(state, "analysis").stat().st_mtime_ns:
            raise AgentToolError("报告早于最新分析，请先运行 write_report")
        evidence = read_json(analysis["artifact_paths"]["evidence"])["evidence"]
        valid_ids = {item["id"] for item in evidence}
        used_ids = set(analysis.get("evidence_ids", []))
        for part in analysis.get("structure", []):
            used_ids.update(part.get("evidence_ids", []))
        if not used_ids <= valid_ids:
            raise AgentToolError("结论引用了不存在的证据 ID")
        destination = state.get("result_dir")
        if not destination:
            return str(report)
        output = export_results(self._path(state, "analysis"), destination)
        extra = self._review_lines(analysis.get("agent_reviews", []))
        if extra:
            with (output / "report.md").open("a", encoding="utf-8") as file:
                file.write("\n".join(extra) + "\n")
        return str(output)
