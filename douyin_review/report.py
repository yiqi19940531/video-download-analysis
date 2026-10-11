"""Human-readable report from structured analysis artifacts."""

from __future__ import annotations

from pathlib import Path

from .storage import read_json


def _time(ms: int) -> str:
    seconds = max(0, ms // 1000)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _safe(text: object) -> str:
    return str(text or "").replace("\n", " ").strip()


def render_report(analysis: dict, path: Path, *, compact: bool = False) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    asr = read_json(analysis["artifact_paths"]["asr"])["segments"]
    ocr = read_json(analysis["artifact_paths"]["ocr"])
    visual = read_json(analysis["artifact_paths"]["visual"])
    lines = [
        "# 视频解析报告", "",
        f"- 平台：{analysis.get('platform', '未知')}",
        f"- 视频：{analysis['source_url']}",
        f"- 时长：{_time(analysis['duration_ms'])}",
        f"- 标题：{_safe(analysis.get('metadata', {}).get('title')) or '未获取'}", "",
        "## 内容理解", "",
        f"**主题：** {_safe(analysis.get('topic'))}", "",
        f"**故事或观点：** {_safe(analysis.get('story_or_viewpoint'))}", "",
        f"**摘要：** {_safe(analysis.get('summary'))}", "",
        "## 视频结构", "",
    ]
    for part in analysis.get("structure", []):
        refs = ", ".join(part.get("evidence_ids", [])) or "无"
        lines += [f"- **{_time(part['start_ms'])}–{_time(part['end_ms'])}｜{_safe(part['role'])}：** {_safe(part['description'])}（证据：{refs}）"]
    lines += ["", "## 画面场景与视觉风格", ""]
    for item in visual["observations"]:
        lines += [f"- [{_time(item['timestamp_ms'])}] {_safe(item['description'])}；风格：{_safe(item['style']) or '未说明'}（{item['id']}）"]
    if visual["motion_clips"]:
        lines += ["", "## 动作与转场", ""]
        for item in visual["motion_clips"]:
            lines += [f"- [{_time(item['start_ms'])}–{_time(item['end_ms'])}] {_safe(item['description'])}（{item['id']}）"]
    lines += ["", "## 关键帧", ""]
    for frame in analysis.get("keyframes", []):
        relative = Path(frame["path"]).relative_to(path.parent)
        lines += [f"### {_time(frame['timestamp_ms'])} · {frame['scene_id']}", "",
                  f"原因：{_safe(frame['reason'])}", "",
                  f"![{frame['id']}]({relative.as_posix()})", ""]
    lines += ["## 完整语音转写", ""]
    if asr:
        lines += [f"- [{_time(item['start_ms'])}–{_time(item['end_ms'])}] {_safe(item['text'])}（{item['id']}）" for item in asr]
    else:
        lines += ["未检测到可转写语音。"]
    lines += ["", "## 画面文字与字幕", ""]
    text_items = sorted(ocr["spans"] + ocr["subtitles"], key=lambda item: item["start_ms"])
    if text_items:
        lines += [f"- [{_time(item['start_ms'])}–{_time(item['end_ms'])}] {_safe(item['text'])}（{item['id']}）" for item in text_items]
    else:
        lines += ["未提取到可确认的画面文字或字幕。"]
    if analysis.get("uncertainty"):
        lines += ["", "## 不确定信息", ""]
        lines += [f"- {_safe(item)}" for item in analysis["uncertainty"]]
    if compact:
        lines += ["", "## 数据文件", "", "- `analysis.json`：结构化结论、时间轴证据、完整转写与画面文字", "- `keyframes/`：入选画面", ""]
    else:
        lines += ["", "## 数据文件", "", "- `analysis.json`：结构化结论与关键帧索引", "- `evidence.json`：按时间轴排列的原始证据", "- `asr.json`、`ocr.json`、`scenes.json`、`visual.json`：各步骤完整结果", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
