"""Small checkpoint state and validated decisions for the agent loop."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator


Action = Literal[
    "import_workflow_run", "resolve_source", "download", "probe", "transcribe",
    "sample_frames", "read_screen_text", "describe_visuals", "fuse_evidence",
    "write_analysis", "repair_summary", "repair_structure", "write_report", "inspect_evidence", "review_frame",
    "review_clip", "ask_human", "finish",
]


class Decision(BaseModel):
    action: Action
    reason: str = Field(min_length=3, max_length=300)
    args: dict[str, Any] = Field(default_factory=dict)
    question: str | None = None

    @model_validator(mode="after")
    def require_question(self) -> "Decision":
        if self.action == "ask_human" and not (self.question or "").strip():
            raise ValueError("ask_human requires a concrete question")
        return self


class AgentState(TypedDict, total=False):
    raw_input: str
    local_file: str
    seed_run_dir: str
    run_dir: str
    result_dir: str
    chrome_profile: str
    interactive_browser: bool
    artifacts: dict[str, str]
    decision: dict[str, Any]
    history: list[dict[str, Any]]
    human_notes: list[dict[str, Any]]
    steps: int
    max_steps: int
    human_attempts: int
    review_dirty: bool
    status: str
    last_observation: str
    report_path: str
    published_path: str
