"""Publish a small, reviewed subset of local Agent Loop results.

Run from the repository root after build_record.py has checked the local runs:
python scripts/build_agent_loop_examples.py --record agent_loop/validation/2026-10-12/results.json
The generated files contain reports and three still images per video, never media files.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


NAMES = {
    "bilibili": "B站｜学以乱用",
    "douyin": "抖音｜中国人能飞",
    "kuaishou": "快手｜哪个状元更有含金量",
    "xiaohongshu": "小红书｜酱板鸭反转短剧",
}


def _clock(milliseconds: int) -> str:
    seconds = milliseconds // 1000
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _safe_text(value: str) -> str:
    if any(token in value for token in ("xsec_token=", "DASHSCOPE_API_KEY=", "sk-ws-")):
        raise ValueError("公开示例包含访问令牌或密钥")
    return value


def _curate_report(raw: str, frames: list[tuple[str, int]]) -> str:
    prefix, remainder = raw.split("## 关键帧\n", 1)
    _, suffix = remainder.split("## 完整语音转写", 1)
    suffix = suffix.split("## 数据文件", 1)[0]
    lines = [prefix.rstrip(), "", "## 公开的关键帧", "",
             "完整运行选出的其他关键帧和原视频保存在本地，此处只展示三张截图。", ""]
    for filename, timestamp in frames:
        lines += [f"### {_clock(timestamp)}", "", f"![{filename}](frames/{filename})", ""]
    lines += ["## 完整语音转写" + suffix.rstrip(), ""]
    result = "\n".join(lines)
    if "keyframes/" in result:
        raise ValueError("报告仍引用了未公开的关键帧")
    return _safe_text(result)


def build(record: dict, target: Path) -> None:
    platform = record["platform"]
    if platform not in NAMES or record["status"] != "complete":
        raise ValueError(f"示例未验证完成：{platform}")
    analysis = json.loads(Path(record["analysis_path"]).read_text(encoding="utf-8"))
    if record["missing_evidence_ids"] or len(record["screenshots"]) != 3:
        raise ValueError(f"证据或截图未通过验证：{platform}")
    target.mkdir(parents=True, exist_ok=True)
    frames = []
    for position, shot in zip(("opening", "middle", "ending"), record["screenshots"], strict=True):
        original = Path(shot["path"])
        if not original.is_file() or original.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise ValueError(f"截图不存在：{original}")
        filename = position + original.suffix.lower()
        destination = target / "frames" / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
        frames.append((filename, shot["timestamp_ms"]))

    trace = " → ".join(record["decision_trace"])
    lines = [f"# Agent Loop 示例：{NAMES[platform]}", "",
             "[返回项目主页](../../../README.md) · [完整模型报告](report.md)", "",
             "这份结果来自 2026-10-12 对原始链接的一次**全新下载和完整分析**。"
             "报告由模型辅助生成，时间戳、证据 ID 和截图供人工核对。原视频不在仓库中。", "",
             f"- 原始页面：{analysis['source_url']}",
             f"- 时长：{record['duration_ms'] / 1000:.2f} 秒；下载方式：`{record['download_method']}`。",
             f"- 镜头 {record['scene_count']} 个；候选帧 {record['candidate_frames']} 张；"
             f"OCR 采样帧 {record['ocr_sample_frames']} 张。",
             f"- ASR {record['asr_segments']} 条；画面文字 {record['ocr_spans']} 条；"
             f"图片观察 {record['visual_observations']} 条；时间轴证据 {record['evidence_count']} 条。",
             f"- 本地最终结果有 {record['keyframe_count']} 张关键帧；本页只公开 3 张。", "",
             "## Agent 实际做了什么", "",
             "`qwen3.8-flash` 每轮依据检查点中的阶段状态选择一个受控动作；本次路径为：", "",
             f"`{trace}`", "",
             "| 环节 | 本次作用 |",
             "| --- | --- |",
             f"| 获取视频 | `{record['download_method']}` 下载，`ffprobe` 确认音视频和时长。 |",
             "| 声音 | FFmpeg 制作音频块；百炼 `qwen-audio-3.0-asr-flash` 返回带时间戳的转写。 |",
             "| 画面 | PySceneDetect/OpenCV 切镜头和抽帧；PaddleOCR 在本地读取文字。 |",
             "| 视觉 | 百炼 `qwen3.8-flash` 描述镜头代表帧和最多三段无声短片。 |",
             "| 结论 | 合并来源证据后，用 LangChain/Pydantic 调 `qwen3.8-flash` 生成结构、摘要和不确定项。 |",
             "| 交付 | LangGraph 检查报告和证据引用；公开报告保留时间戳，截图只取三张。 |", ""]
    if record.get("quality_repair"):
        lines += ["**本次质量复核**：" + record["quality_repair"] + "。"
                  f"复核路径：`{' → '.join(record['repair_decision_trace'])}`。"
                  "下面展示的是最终版本。", ""]
    if analysis.get("audit_status"):
        lines += [f"条件审校状态：`{analysis['audit_status']}`。", ""]
    lines += ["## 这条视频讲了什么", "", analysis["summary"], "",
              "## 分段与证据", "",
              "| 原视频时间 | 情节或观点 | 证据 ID |",
              "| --- | --- | --- |"]
    for part in analysis["structure"]:
        interval = f"{_clock(part['start_ms'])}–{_clock(part['end_ms'])}"
        description = str(part["description"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {interval} | {description} | {', '.join(part['evidence_ids'])} |")
    lines += ["", "## 选出的三张截图", ""]
    for filename, timestamp in frames:
        lines += [f"**{_clock(timestamp)}**", "", f"![{platform} {filename}](frames/{filename})", ""]
    lines += ["## 不确定与人工复核", ""]
    for item in analysis.get("uncertainty", []):
        lines.append(f"- {item}")
    lines += ["", "[查看完整模型报告，包括视觉描述、语音和画面文字](report.md)。"
              "证据 ID 可定位模型所用记录，但不能独立证明其解释必然正确。", ""]
    (target / "README.md").write_text(_safe_text("\n".join(lines)), encoding="utf-8")
    report = Path(record["report_path"]).read_text(encoding="utf-8")
    (target / "report.md").write_text(_curate_report(report, frames), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build small public Agent Loop examples from checked local results")
    parser.add_argument("--record", type=Path, required=True, help="Local results.json from build_record.py")
    parser.add_argument("--output", type=Path, default=Path("examples/agent-loop"))
    args = parser.parse_args()
    records = json.loads(args.record.read_text(encoding="utf-8"))
    if {item["platform"] for item in records} != set(NAMES):
        raise ValueError("必须提供四个平台的验证记录")
    for item in records:
        build(item, args.output / item["platform"])


if __name__ == "__main__":
    main()
