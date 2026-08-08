# Browser Worlds × Tinker RL

This repository will demonstrate specialist browser-agent improvement with reinforcement
learning on deterministic Browser Use Worlds. Before changing the environment, it reproduces
Tinker Cookbook's default Harbor RL recipe on Terminal-Bench using Inkling.

No browser-agent training code is present yet.

## Pinned baseline

Tinker Cookbook is pinned in `pyproject.toml`; see [UPSTREAM.md](UPSTREAM.md). The baseline uses:

- the upstream Harbor task loader and bash-tool agent;
- Terminal-Bench 2.0;
- Modal sandboxes;
- each task's `tests/test.sh` reward;
- `thinkingmachines/Inkling` instead of the recipe's original model default.

## Setup

Place `TINKER_API_KEY` in `.env` and authenticate Modal. Then:

```bash
uv sync
uvx harbor datasets download terminal-bench@2.0 \
  -o ~/.cache/harbor/tasks/terminal-bench-2.0
```

Load `.env` without passing credentials as command arguments:

```bash
set -a
source .env
set +a
```

## One-step baseline

Run the smallest real training step first:

```bash
uv run python scripts/train_terminal_bench_smoke.py
```

This runs one Terminal-Bench task with two Inkling rollouts and disables the upstream iteration-zero
evaluation over all 89 tasks. The Harbor agent, sandbox, grader, and training loop are unchanged.

Validated on August 8, 2026: one batch and two rollouts completed, the optimizer step succeeded,
and final Tinker state and sampler checkpoints were produced. Both rollouts reached their token
limit and received `-0.1`, so this validates the baseline pipeline rather than task performance.

## Full upstream example

After the one-step run succeeds:

```bash
uv run python -m tinker_cookbook.recipes.harbor_rl.scripts.train_terminal_bench \
  model_name=thinkingmachines/Inkling \
  learning_rate=1e-5
```

The learning rate is an explicit initial experiment value; Inkling does not publish a universal
recommended RL learning rate.
