"""Generate and execute four concurrent Inkling Zenith Browser Harness rollouts."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import tinker
from tinker_cookbook.renderers import Message, get_text_content
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = "thinkingmachines/Inkling"
ROLLOUTS = 4

BROWSER_HARNESS_SKILL = (
    Path(__file__).resolve().parents[1] / "skills/browser-harness/SKILL.md"
).read_text()
SYSTEM_PROMPT = (
    "You are a browser agent. "
    "Use the browser_harness tool to achieve your goal in the already-connected local Chromium browser.\n\n"
    + BROWSER_HARNESS_SKILL
)

PROMPT = """You are controlling a local Chromium browser through Browser Harness Python.
Complete this browser task independently: From the provided Zenith UK search-results page,
extract at least three smart-lighting products and return each product's exact title, numeric
GBP price, and actual seller name from its product detail page. Set delivery to the United
Kingdom through the website UI before extracting. Do not mistake a brand or badge for seller.

Your program receives ENTRY, the exact task URL. Browser Harness helpers are already global;
do not import browser_harness. Exact synchronous calls include new_tab(url), goto_url(url),
wait_for_load(), js(expression), cdp(method, **params), click_at_xy(x,y), type_text(text),
page_info(), and capture_screenshot(path). There is no click(selector) helper. Prefer js(...)
for DOM inspection; use a DOM form's requestSubmit() to change delivery market through the UI.
Use browser actions only; do not use requests or access the World control plane. Write the final raw JSON array to
/tmp/outputs/final-answer.json and print the same JSON. Each array item must have exactly title,
price, and seller. The surrounding runner records the final screenshot and page metadata.

Call browser_harness exactly once with executable Python. Import json and os.
"""

BROWSER_TOOL = {
    "name": "browser_harness",
    "description": "Execute synchronous Browser Harness Python against the task browser.",
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Complete Python program using Browser Harness calls.",
            }
        },
        "required": ["code"],
        "additionalProperties": False,
    },
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_program(text: str) -> str:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1]).strip()
    if value.startswith("python\n"):
        value = value[7:]
    compile(value, "<inkling-program>", "exec")
    return value + "\n"


async def sample_program(client, renderer, ordinal: int, root: Path) -> Path:
    messages = renderer.create_conversation_prefix_with_tools(
        tools=[BROWSER_TOOL], system_prompt=SYSTEM_PROMPT
    ) + [Message(role="user", content=PROMPT)]
    prompt = renderer.build_generation_prompt(messages, effort=0.7)
    response = await client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=8192,
            temperature=1.0,
            seed=880800 + ordinal,
            stop=renderer.get_stop_sequences(),
        ),
    )
    sequence = response.sequences[0]
    message, termination = renderer.parse_response(sequence.tokens)
    raw = get_text_content(message)
    tool_calls = list(message.get("tool_calls") or [])
    if len(tool_calls) != 1 or tool_calls[0].function.name != "browser_harness":
        raise RuntimeError(
            f"rollout {ordinal} did not produce one browser_harness call: "
            f"content={raw!r}, calls={[tc.function.name for tc in tool_calls]}"
        )
    arguments = json.loads(tool_calls[0].function.arguments)
    program = clean_program(str(arguments["code"]))
    rollout = root / f"rollout-{ordinal:02d}"
    rollout.mkdir()
    program_path = rollout / "inkling-program.py"
    program_path.write_text(program)
    receipt = {
        "ordinal": ordinal,
        "model": MODEL,
        "seed": 880800 + ordinal,
        "effort": 0.7,
        "max_tokens": 8192,
        "generated_tokens": len(sequence.tokens),
        "termination": str(getattr(termination, "value", termination)),
        "prompt_sha256": sha(PROMPT.encode()),
        "response_sha256": sha(raw.encode()),
        "program_sha256": sha(program.encode()),
        "raw_response": raw,
        "tool_name": tool_calls[0].function.name,
        "tool_arguments_sha256": sha(tool_calls[0].function.arguments.encode()),
    }
    (rollout / "generation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return program_path


async def execute_rollout(
    ordinal: int,
    program: Path,
    root: Path,
    args: argparse.Namespace,
    barrier: asyncio.Event,
) -> dict:
    await barrier.wait()
    rollout = root / f"rollout-{ordinal:02d}"
    result_dir = rollout / "execution"
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("verify_zenith_daytona_sandbox.py")),
        "--world-binary", str(args.world_binary),
        "--customer-repo", str(args.customer_repo),
        "--output", str(result_dir),
        "--program", str(program),
        "--rollout-id", f"r{ordinal:02d}",
        "--seed", "411001",
    ]
    env = os.environ.copy()
    for key in (
        "BROWSER_USE_API_KEY",
        "BROWSER_USE_CLOUD_API_KEY",
        "BROWSER_USE_CLOUD_URL",
        "BROWSER_USE_API_URL",
    ):
        env.pop(key, None)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    stdout, _ = await proc.communicate()
    (rollout / "runner.log").write_bytes(stdout)
    record = {"ordinal": ordinal, "exit_code": proc.returncode}
    result_path = result_dir / "result.json"
    cleanup_path = result_dir / "cleanup.json"
    if result_path.exists():
        record["result"] = json.loads(result_path.read_text())
    if cleanup_path.exists():
        record["cleanup"] = json.loads(cleanup_path.read_text())
    if proc.returncode and stdout:
        record["error_tail"] = stdout.decode(errors="replace")[-4000:]
    return record


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-binary", type=Path, required=True)
    parser.add_argument("--customer-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("TINKER_API_KEY", "").strip():
        raise RuntimeError("TINKER_API_KEY is required")
    if not os.environ.get("DAYTONA_KEY", "").strip():
        raise RuntimeError("DAYTONA_KEY is required")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)

    renderer = TmlV0Renderer(get_tokenizer(MODEL))
    service = tinker.ServiceClient()
    client = await service.create_sampling_client_async(base_model=MODEL)
    programs = await asyncio.gather(
        *(sample_program(client, renderer, ordinal, root) for ordinal in range(1, ROLLOUTS + 1))
    )
    barrier = asyncio.Event()
    tasks = [
        asyncio.create_task(execute_rollout(i, programs[i - 1], root, args, barrier))
        for i in range(1, ROLLOUTS + 1)
    ]
    barrier_time = datetime.now(UTC).isoformat()
    (root / "barrier.json").write_text(json.dumps({
        "released_at": barrier_time,
        "rollouts": ROLLOUTS,
        "concurrency": "all",
    }, indent=2) + "\n")
    barrier.set()
    records = await asyncio.gather(*tasks)
    complete = sum(int(bool(r.get("result", {}).get("complete"))) for r in records)
    passed = sum(int(bool(r.get("result", {}).get("deterministic_passed"))) for r in records)
    rewards = [float(r.get("result", {}).get("deterministic_reward", 0)) for r in records]
    summary = {
        "model": MODEL,
        "task": "zenith-zslr01",
        "requested": ROLLOUTS,
        "accounted": len(records),
        "complete": complete,
        "passed": passed,
        "pass_rate": passed / ROLLOUTS,
        "mean_deterministic_reward": sum(rewards) / ROLLOUTS,
        "barrier_released_at": barrier_time,
        "browser_use_cloud": False,
        "records": records,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
