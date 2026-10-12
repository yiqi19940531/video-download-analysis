"""Decide -> tool/human -> observe -> decide, with durable human interrupts."""

from __future__ import annotations

import re
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from douyin_review.storage import read_json

from .planner import Planner
from .state import AgentState, Decision
from .tools import AgentTools


MAX_CONSECUTIVE_HUMAN_QUESTIONS = 3


def _safe_error(exc: Exception, state: AgentState, tools: AgentTools) -> str:
    message = f"{type(exc).__name__}: {str(exc)}"
    source = tools._path(state, "source")
    if source.is_file():
        download_url = read_json(source).get("download_url", "")
        if download_url:
            message = message.replace(download_url, "[内部下载地址]")
    message = re.sub(r"xsec_token=[^&\s]+", "xsec_token=[已隐藏]", message)
    return message[:700]


def build_agent_graph(planner: Planner, tools: AgentTools):
    """Build a true tool-selection loop; media actions are a closed registry."""

    def decide(state: AgentState) -> dict:
        steps = state.get("steps", 0)
        recent = state.get("history", [])[-1:]
        if (state.get("human_attempts", 0) >= MAX_CONSECUTIVE_HUMAN_QUESTIONS
                and recent and recent[0].get("kind") == "tool_error"
                and recent[0].get("action") == "resolve_source"
                and "输入中没有找到受支持的视频链接" in recent[0].get("observation", "")):
            observation = "连续三次人工补充都未提供可解析的视频来源，已停止反问；请用完整信息新建运行"
            return {"status": "human_input_exhausted", "last_observation": observation,
                    "history": state.get("history", []) + [{"step": steps,
                        "kind": "human_limit", "observation": observation}]}
        if steps >= state.get("max_steps", 16):
            return {"status": "step_limit", "last_observation": "已达到本次智能体步数上限"}
        choice = planner.choose(state)
        attempts = state.get("human_attempts", 0)
        if choice.action == "ask_human" and attempts >= MAX_CONSECUTIVE_HUMAN_QUESTIONS:
            observation = "连续三次人工补充仍未满足任务条件，已停止反问；请用完整信息新建运行"
            return {"status": "human_input_exhausted", "last_observation": observation,
                    "history": state.get("history", []) + [{"step": steps + 1,
                        "kind": "human_limit", "observation": observation}]}
        return {"decision": choice.model_dump(), "steps": steps + 1,
                "human_attempts": attempts + 1 if choice.action == "ask_human" else attempts,
                "status": "running",
                "history": state.get("history", []) + [{"step": steps + 1,
                    "kind": "decision", "action": choice.action, "reason": choice.reason}]}

    def route(state: AgentState) -> str:
        if state.get("status") in {"step_limit", "human_input_exhausted"}:
            return END
        action = state["decision"]["action"]
        if action == "ask_human":
            return "human"
        if action == "finish":
            return "finish"
        return "execute"

    def execute(state: AgentState) -> dict:
        choice = Decision.model_validate(state["decision"])
        try:
            outcome = tools.execute(choice.action, state, choice.args)
            update: dict = {"artifacts": tools.artifact_paths(state["run_dir"]),
                            "last_observation": outcome.observation,
                            "history": state.get("history", []) + [{"step": state["steps"],
                                "kind": "tool", "action": choice.action,
                                "observation": outcome.observation[:700]}]}
            if outcome.review_dirty is not None:
                update["review_dirty"] = outcome.review_dirty
            # A completed tool breaks a chain of unanswered or insufficient human replies.
            # A failed tool leaves the count intact, so invalid links cannot loop forever.
            if choice.action != "inspect_evidence":
                update["human_attempts"] = 0
            return update
        except Exception as exc:
            observation = _safe_error(exc, state, tools)
            return {"artifacts": tools.artifact_paths(state["run_dir"]),
                    "last_observation": observation,
                    "history": state.get("history", []) + [{"step": state["steps"],
                        "kind": "tool_error", "action": choice.action,
                        "observation": observation}]}

    def human(state: AgentState) -> dict:
        choice = Decision.model_validate(state["decision"])
        # This is deliberately the first side-effectful operation in the node:
        # LangGraph re-enters the node on Command(resume=...).
        answer = interrupt({"question": choice.question, "field": choice.args.get("field", "observation"),
                            "run_dir": state["run_dir"]})
        answer = str(answer).strip()
        if not answer:
            observation = "人工没有提供内容；请提出更具体的问题或安全停止"
            return {"last_observation": observation,
                    "history": state.get("history", []) + [{"step": state["steps"],
                        "kind": "human_empty", "observation": observation}]}
        field = choice.args.get("field", "observation")
        update: dict = {"human_notes": state.get("human_notes", []) + [
            {"question": choice.question, "answer": answer[:1000], "field": field}],
            "last_observation": f"人工已补充 {field} 信息",
            "history": state.get("history", []) + [{"step": state["steps"],
                "kind": "human", "field": field, "observation": "已收到人工补充"}]}
        if field == "input":
            if tools._path(state, "media").is_file():
                update["last_observation"] = "视频已下载，不能在同一运行中替换来源；请建立新的运行"
            else:
                tools._path(state, "source").unlink(missing_ok=True)
                supplied = Path(answer).expanduser()
                if supplied.is_file():
                    update.update({"local_file": str(supplied.resolve()), "raw_input": ""})
                else:
                    update.update({"raw_input": answer, "local_file": ""})
        elif field == "result_dir":
            update["result_dir"] = str(Path(answer).expanduser().resolve())
        else:
            try:
                added = tools.record_human_observation(state, choice.question or "", answer, choice.args)
                if added:
                    update["review_dirty"] = True
            except Exception as exc:
                update["last_observation"] = _safe_error(exc, state, tools)
        update["artifacts"] = tools.artifact_paths(state["run_dir"])
        return update

    def finish(state: AgentState) -> dict:
        try:
            published = tools.publish(state)
            return {"status": "complete", "published_path": published,
                    "report_path": str(tools._path(state, "report")),
                    "last_observation": "分析已完成并保存交付物"}
        except Exception as exc:
            observation = _safe_error(exc, state, tools)
            return {"status": "running", "last_observation": observation,
                    "history": state.get("history", []) + [{"step": state["steps"],
                        "kind": "finish_rejected", "observation": observation}]}

    builder = StateGraph(AgentState)
    builder.add_node("decide", decide)
    builder.add_node("execute", execute)
    builder.add_node("human", human)
    builder.add_node("finish", finish)
    builder.add_edge(START, "decide")
    builder.add_conditional_edges("decide", route,
                                  {"execute": "execute", "human": "human",
                                   "finish": "finish", END: END})
    builder.add_edge("execute", "decide")
    builder.add_edge("human", "decide")
    builder.add_conditional_edges("finish", lambda state: END if state.get("status") == "complete" else "decide",
                                  {END: END, "decide": "decide"})
    return builder
