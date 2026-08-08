# Browser Worlds × Tinker RL

A focused, reproducible demonstration of improving a small open-source browser agent with
reinforcement learning on deterministic Browser Use Worlds.

The first experiment targets a narrow synthetic-commerce domain. We will publish held-out
before/after scores, traces, and short browser recordings. The model improvement is the demo;
the core result is that high-quality Worlds, tasks, and rewards make browser-agent RL practical.

## Architecture

This project preserves Tinker Cookbook's Harbor RL training loop and tool-calling agent. It
changes only the domain boundary:

- the Harbor-shaped bash tool runs `browser-harness` against a leased Browser Use Cloud browser;
- a task supplies an instruction, starting URL, and deterministic World grader;
- one sandbox and one browser lease are created and cleaned up per rollout;
- the upstream group-relative training loop consumes the resulting scalar reward.

Upstream is pinned in `pyproject.toml`; see [UPSTREAM.md](UPSTREAM.md) for provenance.

## Status

The Harbor-derived agent scaffold, typed task format, browser-aware bash tool, lifecycle
interfaces, and training configuration are initialized. Live browser provisioning and the first
World reward adapter are the next implementation step.

## Quick start

```bash
uv sync --extra dev
uv run browser-rl validate tasks/example.jsonl
uv run pytest
```

Training credentials belong in a local `.env` and must never be committed. A live run will be
added only after the browser lease and grader adapters are connected.

## Layout

```text
configs/                 Experiment configuration
src/browser_rl/          Agent, environment, grader, task, and training code
tasks/                   Train/held-out task manifests
scripts/                 Small operational entry points
tests/                   Focused contract tests
```

