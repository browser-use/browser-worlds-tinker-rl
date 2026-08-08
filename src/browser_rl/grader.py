"""Reward boundary for deterministic Browser Use Worlds."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from tinker_cookbook.renderers.base import Message

from browser_rl.task import BrowserTask

WorldScore = Callable[[BrowserTask], Awaitable[tuple[float, dict[str, float]]]]


@dataclass
class BrowserWorldReward:
    """Adapt a deterministic World scorer to Tinker's reward-function contract."""

    task: BrowserTask
    score_world: WorldScore

    async def __call__(self, history: list[Message]) -> tuple[float, dict[str, float]]:
        del history
        reward, metrics = await self.score_world(self.task)
        if not 0.0 <= reward <= 1.0:
            raise ValueError(f"World reward must be in [0, 1], got {reward}")
        return reward, {"reward": reward, **metrics}

