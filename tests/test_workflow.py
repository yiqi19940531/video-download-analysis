from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from douyin_review.media import MediaError, _shrink_frame, detect_scenes, douyin_url, make_audio_chunks, probe, sample_video, select_keyframes
from douyin_review.export import export_results
from douyin_review.models import Bailian, ModelError, _analysis_text_incomplete, _incomplete_uncertainties, _model_evidence
from douyin_review.storage import read_json
from douyin_review.workflow import _motion_clips, _needs_summary_polish, _opening_frame, _verify_last_asr, build_graph


def test_modal_url_is_normalized() -> None:
    source = "https://www.douyin.com/jingxuan/search/%E9%85%B1%E6%9D%BF%E9%B8%AD?modal_id=7621111268732095786&type=general"
    assert douyin_url(source) == "https://www.douyin.com/video/7621111268732095786"
    with pytest.raises(MediaError):
        douyin_url("https://evil.example/video/123")


def test_copied_share_text_resolves_short_link(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []
    monkeypatch.setattr("douyin_review.media._resolve_short_http", lambda url: seen.append(url) or "https://www.douyin.com/video/7691519615142277608")
    copied = "0.05 复制打开抖音，看看【乌鲁木齐市实验学校的作品】祝祖国繁荣昌盛！ 🇨🇳国庆和孩子们一起拍的合照太暖... https://v.douyin.com/9XfqXkqnB00/ MwS:/ q@R.xS 02/13 :5pm"
    assert douyin_url(copied) == "https://www.douyin.com/video/7691519615142277608"
    assert seen == ["https://v.douyin.com/9XfqXkqnB00/"]
    with pytest.raises(MediaError, match="多个"):
        douyin_url("https://www.douyin.com/video/1 https://www.douyin.com/video/2")


def test_analysis_frames_are_downscaled() -> None:
    import numpy as np

    frame = np.zeros((384, 216, 3), dtype=np.uint8)
    assert _shrink_frame(frame, max_width=108).shape[:2] == (192, 108)
    assert _shrink_frame(frame, max_width=300) is frame


def test_long_motion_scene_creates_clip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np
    from PIL import Image

    first = tmp_path / "first.jpg"
    last = tmp_path / "last.jpg"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(first)
    Image.fromarray(np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(last)

    def fake_ffmpeg(args, *, timeout=120):
        Path(args[-1]).write_bytes(b"clip")
        return ""

    monkeypatch.setattr("douyin_review.media.command", fake_ffmpeg)
    scene_data = {"scenes": [{"id": "scene_000", "start_ms": 0, "end_ms": 5000}],
                  "candidates": [{"scene_id": "scene_000", "path": str(first)},
                                 {"scene_id": "scene_000", "path": str(last)}]}
    clips = _motion_clips("source.mp4", scene_data, tmp_path)
    assert len(clips) == 1
    assert Path(clips[0]["path"]).is_file()


def test_action_in_scene_over_ten_seconds_creates_short_clips(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np
    from PIL import Image

    frames = []
    for index, ms in enumerate((0, 3000, 6000, 9000, 12000, 14900)):
        path = tmp_path / f"frame_{index}.jpg"
        Image.fromarray(np.random.default_rng(index).integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(path)
        frames.append({"scene_id": "scene_000", "timestamp_ms": ms, "path": str(path)})

    def fake_ffmpeg(args, *, timeout=120):
        Path(args[-1]).write_bytes(b"clip")
        return ""

    monkeypatch.setattr("douyin_review.media.command", fake_ffmpeg)
    data = {"scenes": [{"id": "scene_000", "start_ms": 0, "end_ms": 15000}],
            "candidates": frames}
    clips = _motion_clips("source.mp4", data, tmp_path)
    assert 1 <= len(clips) <= 3
    assert all(item["end_ms"] - item["start_ms"] == 4500 for item in clips)
    assert all(Path(item["path"]).is_file() for item in clips)


def test_opening_frame_is_added_when_best_scene_frame_is_later() -> None:
    early = {"id": "scene_000_0", "timestamp_ms": 150}
    chosen = {"id": "scene_000_2", "timestamp_ms": 3900}
    assert _opening_frame([early, chosen], [chosen]) == early
    assert _opening_frame([early, chosen], [early]) is None


def test_model_endpoints_are_separated() -> None:
    client = Bailian("dummy", "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
    assert client.native_asr_url.endswith("/api/v1/services/aigc/multimodal-generation/generation")
    with pytest.raises(ModelError):
        Bailian("dummy", "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")


def test_asr_nested_response_keeps_word_timestamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "short.mp3"
    audio.write_bytes(b"test")
    client = Bailian("dummy", "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setattr(client, "_asr_request", lambda body: {"output": {"output": {"sentence": {
        "text": "你好。再见。", "begin_time": 100, "end_time": 1200,
        "words": [{"text": "你", "punctuation": "", "begin_time": 100, "end_time": 200},
                  {"text": "好", "punctuation": "。", "begin_time": 200, "end_time": 400},
                  {"text": "再见", "punctuation": "。", "begin_time": 800, "end_time": 1200}],
    }}}})
    result = client.transcribe([{"path": str(audio), "offset_ms": 2000, "duration_ms": 1500}])
    assert [(part["start_ms"], part["end_ms"], part["text"]) for part in result] == [
        (2100, 2400, "你好。"), (2800, 3200, "再见。")]


def test_last_asr_segment_prevents_false_truncation_claim() -> None:
    draft = {"structure": [{"description": "末尾有食物；ASR在2000ms截断。", "evidence_ids": ["asr_0"]}],
             "uncertainty": ["ASR在2000ms截断", "画面食物种类不确定"], "evidence_ids": ["asr_0"]}
    evidence = [{"id": "asr_0", "source": "asr", "start_ms": 0, "end_ms": 2000, "text": "云 南"},
                {"id": "asr_1", "source": "asr", "start_ms": 2000, "end_ms": 2600, "text": "米线。"}]
    fixed = _verify_last_asr(draft, evidence)
    assert "ASR在2000ms截断" not in fixed["structure"][-1]["description"]
    assert "米线。" in fixed["structure"][-1]["description"]
    assert fixed["uncertainty"] == ["画面食物种类不确定"]
    assert "asr_1" in fixed["structure"][-1]["evidence_ids"]


def test_short_or_unfinished_overview_is_repaired() -> None:
    assert _needs_summary_polish({"topic": "以喜剧短剧形式对比", "summary": "这是一条完整的摘要。"})
    assert _needs_summary_polish({"topic": "海滩上的空翻与口号", "summary": "两名男子在海边反复高喊中国人能飞"})
    assert _needs_summary_polish({"topic": "海滩上的空翻与口号",
                                  "summary": "两名男子在海边反复高喊“中国人能”“飞”，同时完成侧手翻。"})
    assert not _needs_summary_polish({"topic": "体育考试状元与高考状元的对比短剧",
                                      "summary": "这条短剧通过夸张的奖励桥段，把体育考试状元与高考状元的待遇放在一起讨论。"})


def test_incomplete_model_text_is_rejected_before_report() -> None:
    complete = {"topic": "海滩翻转表演", "story_or_viewpoint": "两名男子在海滩上展示翻转动作。",
                "summary": "视频展示两名男子在海滩上连续翻转，语音反复说出中国人能飞。",
                "structure": [{"description": "两人依次完成翻转动作。"}],
                "uncertainty": ["画面无法证实发布者提到的观众。"]}
    assert not _analysis_text_incomplete(complete)
    assert _analysis_text_incomplete({**complete, "summary": ""})
    assert _incomplete_uncertainties({**complete, "uncertainty": ["画面不能证明标题中的"]})


def test_model_prompt_keeps_evidence_without_internal_image_paths() -> None:
    item = {"id": "ocr_1", "source": "paddleocr", "start_ms": 100, "end_ms": 600,
            "text": "画面文字", "source_frame": "/private/ocr_samples/000000100.jpg",
            "bbox": [1, 2, 3, 4], "confidence": 0.91}
    assert _model_evidence(item) == {"id": "ocr_1", "source": "paddleocr",
                                     "start_ms": 100, "end_ms": 600, "text": "画面文字"}


def test_summary_repair_preserves_evidence_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    client = Bailian("dummy", "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
    payload = ('{"topic":"海滩上的翻转动作和重复口号",'
               '"story_or_viewpoint":"两名男子在海滩表演侧手翻，画外音反复高喊中国人能飞。",'
               '"summary":"视频展示两名男子在阳光明亮的海滩上连续表演侧手翻和空翻动作，画外音多次反复说出中国人能飞，整体呈现轻松的特技展示。"}')
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=payload))]))))
    monkeypatch.setattr(client, "_openai", lambda: fake)
    draft = {"topic": "海滩表演", "story_or_viewpoint": "两人反复高喊",
             "summary": "两人正在海滩上", "structure": [{"description": "动作"}],
             "evidence_ids": ["asr_1"]}
    result = client.polish_summary(draft, [{"source": "asr", "text": "中国人能飞。"}], {})
    assert result["summary"].endswith("。")
    assert result["structure"] == draft["structure"]
    assert result["evidence_ids"] == ["asr_1"]


def test_long_audio_is_split_and_no_audio_is_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert make_audio_chunks("unused.mp4", tmp_path, 361_000, False) == []

    def fake_ffmpeg(args, *, timeout=600):
        Path(args[-1]).write_bytes(b"mp3")
        return ""

    monkeypatch.setattr("douyin_review.media.command", fake_ffmpeg)
    chunks = make_audio_chunks("unused.mp4", tmp_path, 361_000, True)
    assert [item["offset_ms"] for item in chunks] == [0, 179_000, 358_000]
    assert all(item["duration_ms"] <= 180_000 for item in chunks)


@pytest.fixture
def sample_media_file(tmp_path: Path) -> Path:
    destination = tmp_path / "sample.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=red:s=320x240:d=2:r=10",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:d=2:r=10",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=4",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(destination),
    ], check=True)
    return destination


def test_media_preprocessing(sample_media_file: Path, tmp_path: Path) -> None:
    details = probe(sample_media_file)
    assert 3900 <= details["duration_ms"] <= 4100
    assert details["has_audio"]
    chunks = make_audio_chunks(str(sample_media_file), tmp_path, details["duration_ms"], True)
    assert len(chunks) == 1 and Path(chunks[0]["path"]).is_file()
    scenes = detect_scenes(str(sample_media_file), details["duration_ms"])
    sampled = sample_video(str(sample_media_file), tmp_path, scenes, details["duration_ms"])
    assert sampled["candidates"] and sampled["ocr_samples"]
    keyframes = select_keyframes(sampled["candidates"], [], [])
    assert keyframes and all(Path(item["path"]).is_file() for item in keyframes)


def test_graph_writes_report(sample_media_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("douyin_review.workflow.extract_ocr", lambda samples: [])

    class FakeModel:
        def transcribe(self, chunks):
            return [{"id": "asr_0", "start_ms": 0, "end_ms": 1000, "text": "测试声音", "source": "asr"}]

        def describe_frames(self, frames):
            return [{"id": f"visual_{item['id']}", "frame_id": item["id"],
                     "scene_id": item["scene_id"], "timestamp_ms": item["timestamp_ms"],
                     "description": "彩色画面", "style": "简洁", "visible_text": "", "source": "fake"}
                    for item in frames]

        def describe_clip(self, clip_path, start_ms, end_ms):
            return {"id": "clip_0", "start_ms": start_ms, "end_ms": end_ms, "description": "颜色变化"}

        def summarize(self, evidence, duration_ms, metadata):
            return {"topic": "颜色测试", "story_or_viewpoint": "红色变蓝色", "summary": "展示两种颜色。",
                    "structure": [{"start_ms": 0, "end_ms": duration_ms, "role": "展示", "description": "颜色变化", "evidence_ids": [evidence[0]["id"]]}],
                    "uncertainty": [], "evidence_ids": [evidence[0]["id"]]}

    run_dir = tmp_path / "run"
    graph = build_graph(FakeModel()).compile()
    state = graph.invoke({"source_url": str(sample_media_file), "local_file": str(sample_media_file), "run_dir": str(run_dir)})
    assert Path(state["report_path"]).is_file()
    assert read_json(run_dir / "source.json")["input_kind"] == "local-file"
    analysis = read_json(state["analysis_manifest"])
    assert analysis["summary"] == "展示两种颜色。"
    assert analysis["keyframes"]
    report = Path(state["report_path"]).read_text()
    assert "测试声音" in report
    assert "画面场景与视觉风格" in report

    result_dir = tmp_path / "final-result"
    export_results(state["analysis_manifest"], result_dir)
    assert {item.name for item in result_dir.iterdir()} == {"report.md", "analysis.json", "keyframes"}
    delivered = read_json(result_dir / "analysis.json")
    assert delivered["transcript"][0]["text"] == "测试声音"
    assert "artifact_paths" not in delivered
    assert all((result_dir / frame["path"]).is_file() for frame in delivered["keyframes"])
    assert "keyframes/" in (result_dir / "report.md").read_text()
    with pytest.raises(FileExistsError):
        export_results(state["analysis_manifest"], result_dir)
