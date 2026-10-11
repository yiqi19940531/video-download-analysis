"""Command-line entry point."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from .export import export_results
from .models import Bailian
from .workflow import build_graph, run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="下载并解析单条 B站、抖音、快手或小红书视频")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", "--share-text", dest="input_text", help="包含一条受支持视频链接的分享文字，或单条视频 URL")
    source.add_argument("--url", help="直接提供单条视频 URL；推荐使用 --input")
    source.add_argument("--file", help="本地视频文件；供验证或重新处理使用")
    parser.add_argument("--output-dir", default="runs", help="内部运行数据根目录，默认 runs")
    parser.add_argument("--result-dir", help="最终交付目录；只放报告、结构化结果与关键帧，须为空或不存在")
    parser.add_argument("--chrome-profile", help="专用 Chrome 配置目录，用于下载回退")
    parser.add_argument("--interactive-browser", action="store_true", help="回退时打开可见 Chrome，允许手动登录")
    parser.add_argument("--resume", action="store_true", help="从现有 LangGraph 检查点继续")
    args = parser.parse_args()
    try:
        model = Bailian.from_env()
        if args.input_text or args.url:
            raw_input = args.input_text or args.url
            identifier = run_id(raw_input.strip())
            input_state = {"raw_input": raw_input}
        else:
            file_path = Path(args.file).resolve()
            if not file_path.is_file():
                raise FileNotFoundError(file_path)
            identifier = hashlib.sha256(file_path.read_bytes()).hexdigest()[:16]
            input_state = {"source_url": str(file_path), "local_file": str(file_path)}
        root = Path(args.output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / identifier
        run_dir.mkdir(parents=True, exist_ok=True)
        input_state["run_dir"] = str(run_dir)
        chrome_profile = args.chrome_profile or str(root / "chrome-profile")
        builder = build_graph(model, chrome_profile=chrome_profile, interactive_browser=args.interactive_browser)
        config = {"configurable": {"thread_id": identifier}}
        with SqliteSaver.from_conn_string(str(root / "checkpoints.sqlite")) as checkpointer:
            graph = builder.compile(checkpointer=checkpointer)
            for update in graph.stream(None if args.resume else input_state, config, stream_mode="updates"):
                for node in update:
                    print(f"完成：{node}", flush=True)
            snapshot = graph.get_state(config)
            report = snapshot.values.get("report_path")
            if not report:
                raise RuntimeError("工作流未生成报告")
            if args.result_dir:
                published = export_results(snapshot.values["analysis_manifest"], args.result_dir)
                print(f"交付目录：{published}")
                print(f"报告：{published / 'report.md'}")
                print(f"结构化结果：{published / 'analysis.json'}")
            else:
                print(f"报告：{report}")
                print(f"结构化结果：{snapshot.values['analysis_manifest']}")
    except Exception as exc:
        print(f"失败：{type(exc).__name__}: {str(exc)[:800]}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
