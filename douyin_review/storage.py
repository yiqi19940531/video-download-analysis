"""Small, atomic artifact helpers. Graph checkpoints contain paths, not media bytes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_json(path: str | Path, data: Any) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    return str(destination)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
