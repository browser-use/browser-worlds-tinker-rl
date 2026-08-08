import json
from pathlib import Path

import pytest

from browser_rl.task import load_tasks


def test_load_tasks_preserves_train_and_held_out(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    rows = [
        {
            "task_id": "train-1",
            "split": "train",
            "instruction": "Do the train task",
            "start_url": "https://example.invalid",
            "grader": "world-v1",
        },
        {
            "task_id": "eval-1",
            "split": "held_out",
            "instruction": "Do the held-out task",
            "start_url": "https://example.invalid",
            "grader": "world-v1",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    tasks = load_tasks(path)
    assert [task.split for task in tasks] == ["train", "held_out"]


def test_load_tasks_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    row = {
        "task_id": "same",
        "split": "train",
        "instruction": "Do the task",
        "start_url": "https://example.invalid",
        "grader": "world-v1",
    }
    path.write_text(f"{json.dumps(row)}\n{json.dumps(row)}\n")
    with pytest.raises(ValueError, match="Duplicate task_id"):
        load_tasks(path)

