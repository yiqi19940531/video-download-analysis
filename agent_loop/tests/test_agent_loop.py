from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent_loop.graph import build_agent_graph
from agent_loop.planner import _snapshot
from agent_loop.quality import incomplete_structure_indices
from agent_loop.state import Decision
from agent_loop.tools import AgentToolError, AgentTools, ToolOutcome
from douyin_review.storage import read_json, write_json


class SequencePlanner:
    def __init__(self, *actions: Decision):
        self.actions = list(actions)

    def choose(self, state):
        assert self.actions, "planner should not be called after completion"
        return self.actions.pop(0)


class HumanTestTools:
    def __init__(self, root: Path):
        self.root = root

    def _path(self, state, name):
        return self.root / f"{name}.json"

    def artifact_paths(self, run_dir):
        path = self.root / "source.json"
        return {"source": str(path)} if path.is_file() else {}

    def execute(self, action, state, args):
        assert action == "resolve_source"
        assert state["raw_input"] == "https://www.bilibili.com/video/BV1yJaH6DEV1/"
        (self.root / "source.json").write_text("{}")
        return ToolOutcome("source resolved")

    def publish(self, state):
        return str(self.root / "result")


def test_human_interrupt_resumes_same_agent_loop(tmp_path: Path) -> None:
    planner = SequencePlanner(
        Decision(action="ask_human", reason="No video link was supplied", question="请提供视频链接",
                 args={"field": "input"}),
        Decision(action="resolve_source", reason="The human supplied a link"),
        Decision(action="finish", reason="The fake tool reports completion"),
    )
    graph = build_agent_graph(planner, HumanTestTools(tmp_path)).compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "human-test"}, "recursion_limit": 30}
    initial = {"raw_input": "", "run_dir": str(tmp_path), "history": [], "human_notes": [],
               "artifacts": {}, "steps": 0, "max_steps": 5, "review_dirty": False}
    interrupted = graph.invoke(initial, config)
    assert interrupted["__interrupt__"][0].value["question"] == "请提供视频链接"
    completed = graph.invoke(Command(resume="https://www.bilibili.com/video/BV1yJaH6DEV1/"), config)
    assert completed["status"] == "complete"
    assert completed["steps"] == 3
    assert completed["raw_input"].startswith("https://www.bilibili.com/")


def test_human_interrupt_survives_sqlite_reopen(tmp_path: Path) -> None:
    database = tmp_path / "agent-checkpoints.sqlite"
    tools = HumanTestTools(tmp_path)
    config = {"configurable": {"thread_id": "persisted-human"}, "recursion_limit": 30}
    initial = {"raw_input": "", "run_dir": str(tmp_path), "history": [], "human_notes": [],
               "artifacts": {}, "steps": 0, "max_steps": 5, "review_dirty": False}
    first = SequencePlanner(Decision(action="ask_human", reason="Need a video link",
                                     question="请提供视频链接", args={"field": "input"}))
    with SqliteSaver.from_conn_string(str(database)) as saver:
        graph = build_agent_graph(first, tools).compile(checkpointer=saver)
        assert graph.invoke(initial, config)["__interrupt__"]
    later = SequencePlanner(Decision(action="resolve_source", reason="Human supplied the link"),
                            Decision(action="finish", reason="Ready to finish"))
    with SqliteSaver.from_conn_string(str(database)) as saver:
        graph = build_agent_graph(later, tools).compile(checkpointer=saver)
        state = graph.invoke(Command(resume="https://www.bilibili.com/video/BV1yJaH6DEV1/"), config)
    assert state["status"] == "complete"
    assert state["steps"] == 3


def test_three_incomplete_human_replies_stop_without_fourth_question(tmp_path: Path) -> None:
    question = Decision(action="ask_human", reason="Still need a valid video link",
                        question="请提供可访问的视频链接", args={"field": "input"})
    planner = SequencePlanner(question, question, question, question)
    database = tmp_path / "human-limit.sqlite"
    config = {"configurable": {"thread_id": "three-questions"}, "recursion_limit": 30}
    initial = {"raw_input": "", "run_dir": str(tmp_path), "history": [],
               "human_notes": [], "artifacts": {}, "steps": 0, "max_steps": 10,
               "human_attempts": 0, "review_dirty": False}
    with SqliteSaver.from_conn_string(str(database)) as saver:
        graph = build_agent_graph(planner, HumanTestTools(tmp_path)).compile(checkpointer=saver)
        state = graph.invoke(initial, config)
        for attempt in range(3):
            assert state["__interrupt__"][0].value["question"] == "请提供可访问的视频链接"
            assert graph.get_state(config).values["human_attempts"] == attempt + 1
            state = graph.invoke(Command(resume="这仍然不是视频链接"), config)
        assert "__interrupt__" not in state
        assert state["status"] == "human_input_exhausted"
        assert state["human_attempts"] == 3
        assert sum(item["kind"] == "human_limit" for item in state["history"]) == 1
        assert not planner.actions


def test_successful_tool_resets_human_question_count(tmp_path: Path) -> None:
    question = Decision(action="ask_human", reason="Need a valid link",
                        question="请提供视频链接", args={"field": "input"})
    planner = SequencePlanner(question,
                              Decision(action="resolve_source", reason="Try the new URL"),
                              question)
    graph = build_agent_graph(planner, HumanTestTools(tmp_path)).compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "reset-count"}, "recursion_limit": 30}
    initial = {"raw_input": "", "run_dir": str(tmp_path), "history": [],
               "human_notes": [], "artifacts": {}, "steps": 0, "max_steps": 5,
               "human_attempts": 0, "review_dirty": False}
    assert graph.invoke(initial, config)["__interrupt__"]
    state = graph.invoke(Command(resume="https://www.bilibili.com/video/BV1yJaH6DEV1/"), config)
    assert state["__interrupt__"]
    assert state["human_attempts"] == 1
    assert any(item["kind"] == "tool" for item in state["history"])


def test_three_invalid_sources_stop_before_another_model_call(tmp_path: Path) -> None:
    class InvalidSourceTools(HumanTestTools):
        def execute(self, action, state, args):
            raise AgentToolError("输入中没有找到受支持的视频链接（B站、抖音、快手、小红书）")

    question = Decision(action="ask_human", reason="Need a real video link",
                        question="请提供视频链接", args={"field": "input"})
    resolve = Decision(action="resolve_source", reason="Check the latest answer")
    planner = SequencePlanner(question, resolve, question, resolve, question, resolve)
    graph = build_agent_graph(planner, InvalidSourceTools(tmp_path)).compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "invalid-three"}, "recursion_limit": 35}
    state = graph.invoke({"raw_input": "", "run_dir": str(tmp_path), "history": [],
                          "human_notes": [], "artifacts": {}, "steps": 0, "max_steps": 10,
                          "human_attempts": 0, "review_dirty": False}, config)
    for _ in range(3):
        assert state["__interrupt__"]
        state = graph.invoke(Command(resume="视频在网上，请继续分析"), config)
    assert state["status"] == "human_input_exhausted"
    assert state["human_attempts"] == 3
    assert not planner.actions


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    path = tmp_path / "sample.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=red:s=320x240:d=2:r=10", "-f", "lavfi", "-i",
                    "color=c=blue:s=320x240:d=2:r=10", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=4", "-filter_complex",
                    "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-map", "2:a",
                    "-c:v", "libx264", "-c:a", "aac", "-shortest", str(path)], check=True)
    return path


class FakeModel:
    def transcribe(self, chunks):
        return [{"id": "asr_000_0000", "source": "asr", "start_ms": 0,
                 "end_ms": 1000, "text": "颜色从红色变成蓝色。"}]

    def describe_frames(self, frames):
        return [{"id": f"visual_{item['id']}", "scene_id": item["scene_id"],
                 "frame_id": item["id"], "timestamp_ms": item["timestamp_ms"],
                 "description": "画面有色块。", "style": "简单", "visible_text": ""}
                for item in frames]

    def describe_clip(self, clip_path, start_ms, end_ms):
        return {"id": f"clip_{start_ms}_{end_ms}", "start_ms": start_ms,
                "end_ms": end_ms, "description": "画面颜色发生变化。"}

    def summarize(self, evidence, duration_ms, metadata):
        return {"topic": "红色蓝色交替变化的测试视频", "story_or_viewpoint": "视频依次展示红色和蓝色。",
                "summary": "这段测试视频先展示红色画面，然后展示蓝色画面，并伴随持续的测试声音，最后结束。",
                "structure": [{"start_ms": 0, "end_ms": duration_ms, "role": "展示",
                               "description": "红色和蓝色画面依次出现，形成明显的颜色变化。",
                               "evidence_ids": [evidence[0]["id"]]}],
                "uncertainty": [], "evidence_ids": [evidence[0]["id"]]}

    def polish_summary(self, draft, evidence, metadata):
        return {**draft, "summary": "这段测试视频先展示红色画面，然后展示蓝色画面，并伴随持续的测试声音，最后结束。"}


def test_stage_tools_form_complete_loop_and_review_can_reopen_it(
        sample_video: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_loop.tools.extract_ocr", lambda samples: [])
    model = FakeModel()
    tools = AgentTools(model)
    actions = ["resolve_source", "download", "probe", "transcribe", "sample_frames",
               "read_screen_text", "describe_visuals", "fuse_evidence", "write_analysis",
               "write_report", "finish"]
    planner = SequencePlanner(*(Decision(action=action, reason="Complete the next useful stage")
                                for action in actions))
    graph = build_agent_graph(planner, tools).compile(checkpointer=InMemorySaver())
    run_dir, result_dir = tmp_path / "agent-run", tmp_path / "delivery"
    initial = {"local_file": str(sample_video), "run_dir": str(run_dir),
               "result_dir": str(result_dir), "history": [], "human_notes": [],
               "artifacts": {}, "steps": 0, "max_steps": 14, "review_dirty": False}
    result = graph.invoke(initial, {"configurable": {"thread_id": "video-test"},
                                    "recursion_limit": 60})
    assert result["status"] == "complete"
    assert result["steps"] == len(actions)
    assert (result_dir / "report.md").is_file()
    assert read_json(result_dir / "analysis.json")["evidence"]

    imported = tmp_path / "imported"
    imported_state = {"seed_run_dir": str(run_dir), "run_dir": str(imported), "raw_input": ""}
    tools.execute("import_workflow_run", imported_state, {})
    imported_analysis = read_json(imported / "analysis.json")
    imported_media = read_json(imported / "media.json")
    imported_evidence = read_json(imported / "evidence.json")
    assert str(imported) in imported_media["media_path"]
    assert all(str(imported) in frame["path"] for frame in imported_analysis["keyframes"])
    assert all(str(imported) in frame["path"] and Path(frame["path"]).is_file()
               for frame in imported_evidence["keyframes"])
    assert (imported / "report.md").is_file()

    review = tools.execute("review_frame", result, {"timestamp_ms": 1000})
    assert review.review_dirty is True
    clip_review = tools.execute("review_clip", result, {"start_ms": 0, "end_ms": 4000})
    assert clip_review.review_dirty is True
    assert tools.record_human_observation(result, "画面是什么颜色？", "先红后蓝",
                                          {"start_ms": 0, "end_ms": 4000})
    dirty = {**result, "review_dirty": True, "result_dir": str(tmp_path / "delivery-2")}
    with pytest.raises(AgentToolError, match="尚未重新总结"):
        tools.publish(dirty)
    refreshed = tools.execute("write_analysis", dirty, {})
    assert refreshed.review_dirty is False
    clean = {**dirty, "review_dirty": False}
    tools.execute("write_report", clean, {})
    output = tools.publish(clean)
    delivered = read_json(Path(output) / "analysis.json")
    assert any(item["id"] == "agent_visual_1000" for item in delivered["evidence"])
    assert any(item["id"] == "agent_clip_0_4000" for item in delivered["evidence"])
    assert any(item["source"] == "human_note" for item in delivered["evidence"])
    assert any(frame["id"] == "agent_visual_1000" for frame in delivered["keyframes"])
    assert all((Path(output) / frame["path"]).is_file() for frame in delivered["keyframes"])
    assert "智能体追加检查" in (Path(output) / "report.md").read_text()

    broken = read_json(run_dir / "analysis.json")
    broken["summary"] = "视频先展示"
    broken["polish_status"] = "rejected:ModelError"
    write_json(run_dir / "analysis.json", broken)
    with pytest.raises(AgentToolError, match="摘要未写完整"):
        tools.publish({**clean, "result_dir": str(tmp_path / "delivery-3")})
    assert tools.execute("repair_summary", clean, {}).observation == "摘要修复完成；请重新生成报告"
    tools.execute("write_report", clean, {})
    repaired_output = tools.publish({**clean, "result_dir": str(tmp_path / "delivery-3")})
    repaired = read_json(Path(repaired_output) / "analysis.json")
    assert repaired["polish_status"] == "accepted:agent_repair"
    assert repaired["summary"].endswith("。")

    broken = read_json(run_dir / "analysis.json")
    broken["structure"][0]["description"] = "红蓝颜色"
    write_json(run_dir / "analysis.json", broken)
    assert incomplete_structure_indices(broken) == [0]
    with pytest.raises(AgentToolError, match="时间轴段落未写完整"):
        tools.publish({**clean, "result_dir": str(tmp_path / "delivery-4")})
    monkeypatch.setattr("agent_loop.tools.repair_structure_descriptions",
                        lambda model, analysis, evidence: {
                            **analysis, "structure": [{**analysis["structure"][0],
                                                       "description": "红色和蓝色画面依次出现。"}]})
    assert "已修复 1 个" in tools.execute("repair_structure", clean, {}).observation
    tools.execute("write_report", clean, {})
    structure_output = tools.publish({**clean, "result_dir": str(tmp_path / "delivery-4")})
    structure = read_json(Path(structure_output) / "analysis.json")
    assert structure["structure_repair_status"] == "accepted:agent_repair"


def test_planner_snapshot_hides_human_supplied_link_token(tmp_path: Path) -> None:
    state = {"run_dir": str(tmp_path), "artifacts": {}, "steps": 1,
             "human_notes": [{"field": "input", "question": "视频地址？",
                              "answer": "https://www.xiaohongshu.com/explore/id?xsec_token=private"}]}
    snapshot = _snapshot(state)
    assert snapshot["human_notes"][0]["answer"] == "[已提供视频输入]"


@pytest.mark.parametrize("platform,url", [
    ("bilibili", "https://www.bilibili.com/video/BV1yJaH6DEV1/"),
    ("douyin", "https://www.douyin.com/video/7684591599804937466"),
    ("kuaishou", "https://www.kuaishou.com/short-video/3x56qx9f9beqtmm"),
    ("xiaohongshu", "https://www.xiaohongshu.com/explore/69e09cb0000000001f0035b3?xsec_token=sample"),
])
def test_agent_source_tool_accepts_four_platforms(tmp_path: Path, platform: str, url: str) -> None:
    tools = AgentTools(object())
    state = {"raw_input": url, "run_dir": str(tmp_path)}
    tools.execute("resolve_source", state, {})
    source = read_json(tmp_path / "source.json")
    assert source["platform"] == platform
    assert "xsec_token" not in source["source_url"]


def test_step_limit_stops_repeated_failed_tool_calls(tmp_path: Path) -> None:
    class FailingTools(HumanTestTools):
        def execute(self, action, state, args):
            raise AgentToolError("download unavailable")

    planner = SequencePlanner(Decision(action="download", reason="Try the download tool"),
                              Decision(action="download", reason="Retry after failure"))
    graph = build_agent_graph(planner, FailingTools(tmp_path)).compile()
    state = graph.invoke({"run_dir": str(tmp_path), "history": [], "artifacts": {},
                          "steps": 0, "max_steps": 2, "review_dirty": False},
                         {"recursion_limit": 15})
    assert state["status"] == "step_limit"
    assert sum(item["kind"] == "tool_error" for item in state["history"]) == 2
