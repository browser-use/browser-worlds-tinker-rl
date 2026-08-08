"""Run one real Harbor RL training step on one Terminal-Bench task with Inkling."""

import asyncio

from tinker_cookbook.recipes.harbor_rl.harbor_env import (
    default_sandbox_factory,
    load_harbor_tasks,
)
from tinker_cookbook.recipes.harbor_rl.train import CLIConfig, cli_main

DATASET = "terminal-bench-2.0/terminal-bench"


async def main() -> None:
    tasks = load_harbor_tasks(DATASET)[:1]
    config = CLIConfig(
        model_name="thinkingmachines/Inkling",
        max_steps=1,
        group_size=2,
        groups_per_batch=1,
        learning_rate=1e-5,
        eval_every=0,
        save_every=0,
        log_path="/tmp/tinker-examples/harbor-inkling-one-task",
    )
    await cli_main(config, tasks, sandbox_factory=default_sandbox_factory)


if __name__ == "__main__":
    asyncio.run(main())
