"""Harbor-shaped bash tool configured for browser-harness."""

from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from typing import Annotated

from tinker_cookbook.sandbox import SandboxInterface
from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool

MAX_OUTPUT_CHARS = 16_384

BROWSER_SYSTEM_PROMPT = (
    "You are a browser agent working in an isolated sandbox. Use the bash tool and "
    "browser-harness to operate the website described by the user. Complete only the requested "
    "task and report the result."
)


class BrowserHarnessBashTool:
    """The Harbor bash schema with per-rollout browser connection variables injected."""

    def __init__(
        self,
        sandbox: SandboxInterface,
        browser_environment: Mapping[str, str],
        command_timeout: int = 300,
    ) -> None:
        self._sandbox = sandbox
        self._browser_environment = dict(browser_environment)
        self._command_timeout = command_timeout

    @tool
    async def bash(
        self,
        command: Annotated[str, "The bash command to execute."],
    ) -> ToolResult:
        """Execute a command with this rollout's browser connection available."""
        exports = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in self._browser_environment.items()
        )
        wrapped = f"export {exports}; {command}" if exports else command
        result = await self._sandbox.run_command(
            wrapped,
            workdir="/workspace",
            timeout=self._command_timeout,
            max_output_bytes=MAX_OUTPUT_CHARS,
        )
        output = json.dumps(
            {
                "exit_code": result.exit_code,
                "stdout": result.stdout[:MAX_OUTPUT_CHARS],
                "stderr": result.stderr[:MAX_OUTPUT_CHARS],
            }
        )
        return simple_tool_result(output)

