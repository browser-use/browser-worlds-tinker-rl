"""Validate the example task manifest without creating browsers or model calls."""

from pathlib import Path

from browser_rl.task import load_tasks

tasks = load_tasks(Path("tasks/example.jsonl"))
print(f"Loaded {len(tasks)} tasks without provider calls")

