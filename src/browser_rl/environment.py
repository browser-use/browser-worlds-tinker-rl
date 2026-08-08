"""Browser lifecycle adaptation of Tinker Cookbook's Harbor environment builder."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.rl.types import Env, EnvGroupBuilder
from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.tool_use import build_agent_tool_env

from browser_rl.agent import BROWSER_SYSTEM_PROMPT, BrowserHarnessBashTool
from browser_rl.grader import BrowserWorldReward, WorldScore
from browser_rl.task import BrowserTask

logger = logging.getLogger(__name__)


@dataclass
class BrowserLease:
    """One isolated cloud browser attached to one rollout."""

    environment: Mapping[str, str]
    cleanup: Callable[[], Awaitable[None]]


SandboxFactory = Callable[[int], Awaitable[SandboxInterface]]
BrowserFactory = Callable[[BrowserTask], Awaitable[BrowserLease]]


class BrowserEnvGroupBuilder(EnvGroupBuilder):
    """Create one sandbox and one browser lease for every rollout in a group."""

    def __init__(
        self,
        *,
        task: BrowserTask,
        model_name: str,
        group_size: int,
        sandbox_factory: SandboxFactory,
        browser_factory: BrowserFactory,
        score_world: WorldScore,
        renderer_name: str | None = None,
        max_turns: int = 40,
        sandbox_timeout: int = 3600,
        command_timeout: int = 300,
        max_trajectory_tokens: int = 32 * 1024,
        max_generation_tokens: int = 8192,
    ) -> None:
        self.task = task
        self.model_name = model_name
        self.group_size = group_size
        self.sandbox_factory = sandbox_factory
        self.browser_factory = browser_factory
        self.score_world = score_world
        self.renderer_name = renderer_name
        self.max_turns = max_turns
        self.sandbox_timeout = sandbox_timeout
        self.command_timeout = command_timeout
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens
        self._resources: list[tuple[SandboxInterface, BrowserLease]] = []

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = tokenizer_utils.get_tokenizer(self.model_name)
        renderer_name = self.renderer_name or model_info.get_recommended_renderer_name(
            self.model_name
        )
        renderer = get_renderer(renderer_name, tokenizer)
        envs: list[Env] = []
        for _ in range(self.group_size):
            sandbox = await self.sandbox_factory(self.sandbox_timeout)
            try:
                browser = await self.browser_factory(self.task)
            except Exception:
                await sandbox.cleanup()
                raise
            self._resources.append((sandbox, browser))
            bash_tool = BrowserHarnessBashTool(
                sandbox,
                browser.environment,
                command_timeout=self.command_timeout,
            )
            tool_specs = [bash_tool.bash.to_spec()]
            prefix = renderer.create_conversation_prefix_with_tools(
                tools=tool_specs,
                system_prompt=BROWSER_SYSTEM_PROMPT,
            )
            messages = prefix + [
                {
                    "role": "user",
                    "content": f"Start at {self.task.start_url}\n\n{self.task.instruction}",
                }
            ]
            envs.append(
                build_agent_tool_env(
                    renderer=renderer,
                    tools=[bash_tool.bash],
                    initial_messages=messages,
                    reward_fn=BrowserWorldReward(self.task, self.score_world),
                    max_turns=self.max_turns,
                    max_trajectory_tokens=self.max_trajectory_tokens,
                    max_generation_tokens=self.max_generation_tokens,
                )
            )
        return envs

    async def cleanup(self) -> None:
        for sandbox, browser in reversed(self._resources):
            try:
                await browser.cleanup()
            except Exception as exc:  # noqa: BLE001 - cleanup must continue for remaining leases
                logger.warning("Browser cleanup failed: %s", exc)
            try:
                await sandbox.cleanup()
            except Exception as exc:  # noqa: BLE001 - cleanup must continue for remaining sandboxes
                logger.warning("Sandbox cleanup failed: %s", exc)
        self._resources.clear()

    def logging_tags(self) -> list[str]:
        return ["browser-world", self.task.task_id]
