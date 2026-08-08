"""Task format for browser-agent RL."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class BrowserTask:
    task_id: str
    split: Literal["train", "held_out"]
    instruction: str
    start_url: str
    grader: str


def load_tasks(path: Path) -> list[BrowserTask]:
    """Load and validate a JSONL task manifest."""
    tasks: list[BrowserTask] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            task = BrowserTask(**json.loads(raw_line))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid task at {path}:{line_number}: {exc}") from exc
        if not all((task.task_id, task.instruction, task.start_url, task.grader)):
            raise ValueError(f"Empty required field at {path}:{line_number}")
        if task.split not in {"train", "held_out"}:
            raise ValueError(f"Invalid split at {path}:{line_number}: {task.split}")
        if task.task_id in seen:
            raise ValueError(f"Duplicate task_id at {path}:{line_number}: {task.task_id}")
        seen.add(task.task_id)
        tasks.append(task)
    if not tasks:
        raise ValueError(f"No tasks found in {path}")
    return tasks

