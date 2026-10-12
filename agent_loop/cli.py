"""Local CLI for the model-directed loop; no foreground browser by default."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from douyin_review.models import Bailian

from .graph import build_agent_graph
from .planner import BailianPlanner
from .tools import AgentTools


def _load_local_env() -> None:
    """Read only the two supported variables from the ignored project .env file."""
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in {"DASHSCOPE_API_KEY", "BAILIAN_BASE_URL"} and key not in os.environ:
            os.environ[key] = value.strip().strip("\"'")


def _identifier(args: argparse.Namespace) -> str:
    if args.thread_id:
        return args.thread_id
    if args.input_text:
        value = args.input_text.strip()
    elif args.file:
        value = str(Path(args.file).expanduser().resolve())
    elif args.seed_workflow_run:
        value = str(Path(args.seed_workflow_run).expanduser().resolve())
    else:
        raise ValueError("请提供 --input、--file 或 --seed-workflow-run")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _pending_question(result: dict) -> str | None:
    pending = result.get("__interrupt__", ())
    if not pending:
        return None
    value = pending[0].value
    return str(value.get("question", value)) if isinstance(value, dict) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="模型决定工具调用、支持人工补充的本地视频分析循环")
    parser.add_argument("--input", dest="input_text", help="单条四平台视频 URL 或包含一个链接的分享文字")
    parser.add_argument("--file", help="本地视频文件")
    parser.add_argument("--seed-workflow-run", help="导入已完成的固定工作流运行目录")
    parser.add_argument("--output-dir", default="agent_loop/runs", help="智能体检查点和运行数据根目录")
    parser.add_argument("--result-dir", help="完成后导出的报告、JSON 和关键帧目录")
    parser.add_argument("--chrome-profile", default="runs/chrome-profile", help="无头 Chrome 的专用本地配置")
    parser.add_argument("--interactive-browser", action="store_true", help="明确要求可见浏览器时使用")
    parser.add_argument("--max-steps", type=int, default=16, help="最多决策轮数，默认 16")
    parser.add_argument("--thread-id", help="显式运行 ID；续跑时必须提供")
    parser.add_argument("--resume", action="store_true", help="从同一 thread-id 的检查点继续")
    parser.add_argument("--answer-file", help="续跑时从本地 UTF-8 文件读取人工回答，避免写入命令参数")
    parser.add_argument("--no-prompt", action="store_true", help="需要人工补充时暂停并退出，不读取终端输入")
    args = parser.parse_args()
    if args.input_text and args.file:
        parser.error("--input 和 --file 只能选一个")
    if args.resume and not args.thread_id:
        parser.error("--resume 必须同时提供原来的 --thread-id")
    if args.answer_file and not args.resume:
        parser.error("--answer-file 只能和 --resume 一起使用")
    if not args.resume and not (args.input_text or args.file or args.seed_workflow_run):
        parser.error("请提供 --input、--file 或 --seed-workflow-run")
    if args.max_steps < 2:
        parser.error("--max-steps 至少为 2")
    try:
        _load_local_env()
        model = Bailian.from_env()
        planner = BailianPlanner(model)
        tools = AgentTools(model)
        identifier = _identifier(args)
        root = Path(args.output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / identifier
        run_dir.mkdir(parents=True, exist_ok=True)
        result_dir = (Path(args.result_dir).expanduser().resolve() if args.result_dir else
                      Path("agent_loop/results").resolve() / identifier)
        config = {"configurable": {"thread_id": identifier},
                  "recursion_limit": args.max_steps * 4 + 10}
        with SqliteSaver.from_conn_string(str(root / "checkpoints.sqlite")) as saver:
            graph = build_agent_graph(planner, tools).compile(checkpointer=saver)
            if args.resume:
                snapshot = graph.get_state(config)
                if not snapshot.values:
                    raise RuntimeError("没有找到该 thread-id 的检查点")
                if snapshot.values.get("status") == "complete":
                    print("该运行已经完成：", snapshot.values.get("published_path"))
                    return
                if any(task.interrupts for task in snapshot.tasks):
                    payload = snapshot.tasks[0].interrupts[0].value
                    question = payload.get("question", "请补充信息") if isinstance(payload, dict) else str(payload)
                    if args.answer_file:
                        answer = Path(args.answer_file).expanduser().read_text(encoding="utf-8").strip()
                        if not answer:
                            raise ValueError("人工回答文件为空")
                        value: dict | Command | None = Command(resume=answer)
                    elif args.no_prompt or not sys.stdin.isatty():
                        print(f"等待人工补充：{question}\n用相同的 --thread-id 执行 --resume，在终端回答。")
                        raise SystemExit(3)
                    else:
                        value = Command(resume=input(f"{question}\n> "))
                elif snapshot.next and snapshot.values.get("status") == "running":
                    if args.answer_file:
                        raise ValueError("当前没有待回答的问题，不需要 --answer-file")
                    value = None  # Retry the node that failed after the last checkpoint.
                else:
                    raise RuntimeError("该检查点没有可继续的节点；请检查上次停止状态并新建运行")
            else:
                value = {"raw_input": args.input_text or "", "local_file": args.file or "",
                         "seed_run_dir": args.seed_workflow_run or "", "run_dir": str(run_dir),
                         "result_dir": str(result_dir),
                         "chrome_profile": str(Path(args.chrome_profile).expanduser().resolve()),
                         "interactive_browser": args.interactive_browser,
                         "artifacts": tools.artifact_paths(run_dir), "history": [],
                         "human_notes": [], "steps": 0, "max_steps": args.max_steps,
                         "human_attempts": 0,
                         "review_dirty": False, "status": "running"}
            while True:
                result = graph.invoke(value, config)
                question = _pending_question(result)
                if not question:
                    break
                if args.no_prompt or not sys.stdin.isatty():
                    print(f"等待人工补充：{question}\n运行 ID：{identifier}。"
                          "请在终端用 --resume --thread-id 继续。")
                    raise SystemExit(3)
                value = Command(resume=input(f"{question}\n> "))
            for entry in result.get("history", []):
                if entry.get("kind") in {"tool", "tool_error", "finish_rejected"}:
                    print(f"[{entry.get('step')}] {entry.get('action', entry['kind'])}: "
                          f"{entry.get('observation', '')}")
            if result.get("status") == "complete":
                print("报告：", result["report_path"])
                print("交付：", result["published_path"])
                print("运行 ID：", identifier)
            else:
                print("智能体停止：", result.get("status"), result.get("last_observation", ""))
                raise SystemExit(2)
    except Exception as exc:
        print(f"失败：{type(exc).__name__}: {str(exc)[:700]}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
