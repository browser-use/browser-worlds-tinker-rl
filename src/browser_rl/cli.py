"""Small local CLI for validating experiment inputs before provider calls."""

from __future__ import annotations

import argparse
from pathlib import Path

from browser_rl.task import load_tasks


def main() -> None:
    parser = argparse.ArgumentParser(prog="browser-rl")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate a JSONL task manifest")
    validate.add_argument("manifest", type=Path)
    args = parser.parse_args()
    if args.command == "validate":
        tasks = load_tasks(args.manifest)
        train = sum(task.split == "train" for task in tasks)
        held_out = sum(task.split == "held_out" for task in tasks)
        print(f"valid tasks={len(tasks)} train={train} held_out={held_out}")


if __name__ == "__main__":
    main()

