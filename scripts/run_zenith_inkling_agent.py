"""Run one grounded multi-turn Inkling agent against one local Zenith sandbox."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

import tinker
from tinker_cookbook.renderers import Message, ToolCall, get_text_content
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

MODEL = "thinkingmachines/Inkling"
TASK = "zenith-zslr01"
MAX_GENERATED_TOKENS = 32000
EFFORT = 0.7
SEED = 992001
TASK_INSTRUCTION = (
    "From this Zenith UK search results page, extract a list of smart lighting products, "
    "including the product title, price, and the name of the seller."
)
BROWSER_HARNESS_SKILL = (
    Path(__file__).resolve().parents[1] / "skills/browser-harness/SKILL.md"
).read_text()
SYSTEM_PROMPT = (
    "You are a browser agent. "
    "Use the browser_harness tool to achieve your goal in the already-connected local Chromium browser.\n\n"
    + BROWSER_HARNESS_SKILL
)
BROWSER_TOOL = {
    "name": "browser_harness",
    "description": (
        "Execute synchronous Browser Harness Python against the already-open task browser and "
        "return exact stdout, errors, current page information, and screenshot evidence."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python using the pre-imported Browser Harness helpers.",
            }
        },
        "required": ["code"],
        "additionalProperties": False,
    },
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def jsonable_message(message: Message) -> dict[str, Any]:
    result = dict(message)
    for key in ("tool_calls", "unparsed_tool_calls"):
        if key in result:
            result[key] = [
                value.model_dump(mode="json") if hasattr(value, "model_dump") else value
                for value in result[key]
            ]
    return result


def clean_program(text: str) -> str:
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        lines = value.splitlines()
        value = "\n".join(lines[1:-1]).strip()
    if value.startswith("python\n"):
        value = value[7:]
    compile(value, "<inkling-browser-harness>", "exec")
    return value + "\n"


@dataclass
class SampledTurn:
    message: Message
    generated_tokens: int
    termination: str
    record: dict[str, Any]


SampleTurn = Callable[[list[Message], int, int], Awaitable[SampledTurn]]
ExecuteTool = Callable[[str, int, int], Awaitable[str]]


async def run_agent_loop(
    initial_messages: list[Message],
    sample_turn: SampleTurn,
    execute_tool: ExecuteTool,
) -> dict[str, Any]:
    messages = list(initial_messages)
    events: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []
    total_generated = 0
    model_turns = 0
    tool_calls = 0
    final_response = ""
    termination_reason = "irrecoverable_model_error"
    error = None

    while total_generated < MAX_GENERATED_TOKENS:
        model_turns += 1
        remaining = MAX_GENERATED_TOKENS - total_generated
        try:
            sampled = await sample_turn(messages, remaining, model_turns)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            termination_reason = "irrecoverable_model_error"
            break
        if sampled.generated_tokens <= 0 or sampled.generated_tokens > remaining:
            error = (
                f"invalid generated token count {sampled.generated_tokens} "
                f"with {remaining} remaining"
            )
            termination_reason = "irrecoverable_model_error"
            break
        total_generated += sampled.generated_tokens
        messages.append(sampled.message)
        generations.append(sampled.record)
        events.append({
            "type": "assistant",
            "turn": model_turns,
            "generated_tokens": sampled.generated_tokens,
            "total_generated_tokens": total_generated,
            "termination": sampled.termination,
            "message": jsonable_message(sampled.message),
        })
        final_response = get_text_content(sampled.message)

        if total_generated == MAX_GENERATED_TOKENS:
            termination_reason = "generation_budget_32000"
            break
        if sampled.termination == "malformed":
            error = "model response was malformed before the generation budget was exhausted"
            termination_reason = "irrecoverable_model_error"
            break
        unparsed = list(sampled.message.get("unparsed_tool_calls") or [])
        if unparsed:
            error = "; ".join(value.error for value in unparsed)
            termination_reason = "irrecoverable_model_error"
            break
        calls = list(sampled.message.get("tool_calls") or [])
        if not calls:
            termination_reason = "final_answer" if final_response.strip() else "explicit_stop"
            break

        call_failed = False
        for call_index, call in enumerate(calls, start=1):
            if call.function.name != "browser_harness":
                error = f"unknown tool {call.function.name!r}"
                termination_reason = "irrecoverable_model_error"
                call_failed = True
                break
            try:
                arguments = json.loads(call.function.arguments)
                code = clean_program(str(arguments["code"]))
            except Exception as exc:
                error = f"invalid browser_harness arguments: {type(exc).__name__}: {exc}"
                termination_reason = "irrecoverable_model_error"
                call_failed = True
                break
            try:
                result_text = await execute_tool(code, model_turns, call_index)
            except Exception as exc:
                error = f"browser_harness protocol failed: {type(exc).__name__}: {exc}"
                termination_reason = "irrecoverable_tool_error"
                call_failed = True
                break
            tool_calls += 1
            tool_message: Message = {
                "role": "tool",
                "content": result_text,
                "tool_call_id": call.id or "",
                "name": "browser_harness",
            }
            messages.append(tool_message)
            events.append({
                "type": "tool_result",
                "turn": model_turns,
                "call": call_index,
                "tool_call_id": call.id,
                "content": result_text,
            })
        if call_failed:
            break

    return {
        "termination_reason": termination_reason,
        "error": error,
        "final_response": final_response,
        "model_turns": model_turns,
        "tool_calls": tool_calls,
        "total_generated_tokens": total_generated,
        "messages": [jsonable_message(message) for message in messages],
        "events": events,
        "generations": generations,
    }


def initial_messages(renderer: TmlV0Renderer, observation: str) -> list[Message]:
    return renderer.create_conversation_prefix_with_tools(
        tools=[BROWSER_TOOL], system_prompt=SYSTEM_PROMPT
    ) + [
        Message(role="user", content=TASK_INSTRUCTION),
        Message(
            role="user",
            content=(
                "Initial grounded browser observation; the browser is already on the task page:\n"
                + observation
            ),
        ),
    ]


async def validate_loop(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    renderer = TmlV0Renderer(get_tokenizer(MODEL))
    exact_tool_result = (
        '{"exit_code":0,"stdout":"ok\\n","error":null,"evidence_exit_code":0,'
        '"evidence_stdout":"","page_info":{"url":"http://127.0.0.1/task",'
        '"title":"Zenith"},"screenshot":"/tmp/outputs/turn.png"}'
    )
    observed_remaining = []

    async def fake_sample(messages: list[Message], remaining: int, turn: int) -> SampledTurn:
        renderer.build_generation_prompt(messages, effort=EFFORT)
        observed_remaining.append(remaining)
        if turn == 1:
            message: Message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [ToolCall(
                    id="validation-call-1",
                    function=ToolCall.FunctionBody(
                        name="browser_harness",
                        arguments=json.dumps({"code": "print(page_info())"}),
                    ),
                )],
            }
            return SampledTurn(message, 8, "stop_sequence", {"turn": 1})
        return SampledTurn(
            Message(role="assistant", content='[{"title":"x","price":1,"seller":"y"}]'),
            6,
            "stop_sequence",
            {"turn": 2},
        )

    executed = []

    async def fake_execute(code: str, turn: int, call: int) -> str:
        executed.append({"code": code, "turn": turn, "call": call})
        return exact_tool_result

    result = await run_agent_loop(
        initial_messages(renderer, '{"page_info":{"url":"http://127.0.0.1/task","title":"Zenith"}}'),
        fake_sample,
        fake_execute,
    )
    assert result["termination_reason"] == "final_answer"
    assert result["model_turns"] == 2
    assert result["tool_calls"] == 1
    assert result["total_generated_tokens"] == 14
    assert observed_remaining == [32000, 31992]
    assert executed == [{"code": "print(page_info())\n", "turn": 1, "call": 1}]
    assert result["messages"][-2]["content"] == exact_tool_result
    assert result["messages"][2]["content"] == TASK_INSTRUCTION
    receipt = {
        "validated_at": utc_now(),
        "zero_provider_calls": True,
        "zero_sandboxes": True,
        "result": result,
    }
    atomic_json(output / "validation.json", receipt)
    print(json.dumps(receipt, indent=2))


async def read_protocol_line(proc: asyncio.subprocess.Process) -> dict[str, Any]:
    assert proc.stdout is not None
    line = await proc.stdout.readline()
    if not line:
        raise RuntimeError("interactive verifier closed its protocol stream")
    return json.loads(line)


async def run_live(args: argparse.Namespace) -> None:
    if not os.environ.get("TINKER_API_KEY", "").strip():
        raise RuntimeError("TINKER_API_KEY is required")
    if not os.environ.get("DAYTONA_KEY", "").strip():
        raise RuntimeError("DAYTONA_KEY is required")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    customer = args.customer_repo.resolve()
    instruction = (customer / "tasks/zenith-zslr01/instruction.md").read_text().strip()
    if instruction != TASK_INSTRUCTION:
        raise RuntimeError("Zenith task instruction changed")
    atomic_json(root / "contract.json", {
        "started_at": utc_now(),
        "model": MODEL,
        "task": TASK,
        "rollouts": 1,
        "max_total_generated_tokens": MAX_GENERATED_TOKENS,
        "system_prompt_sha256": digest(SYSTEM_PROMPT.encode()),
        "browser_harness_skill_sha256": digest(BROWSER_HARNESS_SKILL.encode()),
        "task_instruction": TASK_INSTRUCTION,
        "browser_use_cloud": False,
        "sampling_retries": 0,
        "rollout_retries": 0,
    })
    execution = root / "execution"
    command = [
        sys.executable,
        str(Path(__file__).with_name("verify_zenith_daytona_sandbox.py")),
        "--world-binary", str(args.world_binary.resolve()),
        "--customer-repo", str(customer),
        "--output", str(execution),
        "--rollout-id", args.rollout_id,
        "--seed", "411001",
        "--interactive",
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
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    ready = await read_protocol_line(proc)
    if ready.get("type") != "ready":
        raise RuntimeError(f"unexpected verifier handshake: {ready}")
    atomic_json(root / "initial-grounding.json", ready)

    renderer = TmlV0Renderer(get_tokenizer(MODEL))
    tokenizer = get_tokenizer(MODEL)
    service = tinker.ServiceClient(user_metadata={"recipe": "zenith_multi_turn_agent"})
    client = await service.create_sampling_client_async(base_model=MODEL)

    async def sample_turn(messages: list[Message], remaining: int, turn: int) -> SampledTurn:
        prompt = renderer.build_generation_prompt(messages, effort=EFFORT)
        response = await client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=remaining,
                temperature=1.0,
                seed=SEED + turn - 1,
                stop=renderer.get_stop_sequences(),
            ),
        )
        sequence = response.sequences[0]
        tokens = [int(value) for value in sequence.tokens]
        message, termination = renderer.parse_response(tokens)
        termination_text = str(getattr(termination, "value", termination))
        decoded = tokenizer.decode(tokens, skip_special_tokens=False)
        record = {
            "turn": turn,
            "sampled_at": utc_now(),
            "seed": SEED + turn - 1,
            "remaining_generation_budget_before": remaining,
            "prompt_tokens": prompt.length,
            "generated_tokens": len(tokens),
            "completion_tokens": tokens,
            "completion_logprobs": (
                [float(value) for value in sequence.logprobs]
                if sequence.logprobs is not None else None
            ),
            "decoded_response": decoded,
            "stop_reason": str(sequence.stop_reason),
            "termination": termination_text,
            "prompt_cache_hit_tokens": int(response.prompt_cache_hit_tokens),
            "message": jsonable_message(message),
            "usage_or_cost_exposed": False,
        }
        atomic_json(root / "generations" / f"turn-{turn:04d}.json", record)
        return SampledTurn(message, len(tokens), termination_text, record)

    async def execute_tool(code: str, turn: int, call: int) -> str:
        if proc.stdin is None:
            raise RuntimeError("interactive verifier stdin is unavailable")
        proc.stdin.write((json.dumps({
            "type": "tool",
            "turn": turn,
            "call": call,
            "code": code,
        }, separators=(",", ":")) + "\n").encode())
        await proc.stdin.drain()
        response = await read_protocol_line(proc)
        if response.get("type") != "tool_result":
            raise RuntimeError(f"unexpected verifier tool response: {response}")
        return str(response["result"])

    loop_result = await run_agent_loop(
        initial_messages(renderer, str(ready["observation"])),
        sample_turn,
        execute_tool,
    )
    atomic_json(root / "trace.json", loop_result)
    usage = {
        "model_turns": loop_result["model_turns"],
        "tool_calls": loop_result["tool_calls"],
        "generated_tokens": loop_result["total_generated_tokens"],
        "prompt_tokens_sum": sum(
            int(record["prompt_tokens"]) for record in loop_result["generations"]
        ),
        "prompt_cache_hit_tokens_sum": sum(
            int(record["prompt_cache_hit_tokens"]) for record in loop_result["generations"]
        ),
        "generation_cap": MAX_GENERATED_TOKENS,
        "usage_or_cost_exposed": False,
    }
    assert proc.stdin is not None
    proc.stdin.write((json.dumps({
        "type": "finish",
        "final_response": loop_result["final_response"],
        "termination_reason": loop_result["termination_reason"],
        "usage": usage,
    }, separators=(",", ":")) + "\n").encode())
    await proc.stdin.drain()
    proc.stdin.close()
    stderr = await proc.stderr.read() if proc.stderr is not None else b""
    return_code = await proc.wait()
    (root / "verifier.stderr.txt").write_bytes(stderr)
    if return_code:
        raise RuntimeError(
            f"interactive verifier failed ({return_code}): {stderr.decode(errors='replace')[-4000:]}"
        )
    result = json.loads((execution / "result.json").read_text())
    cleanup = json.loads((execution / "cleanup.json").read_text())
    summary = {
        "artifact_root": str(root),
        "completed_at": utc_now(),
        "model": MODEL,
        "task": TASK,
        "rollouts": 1,
        "model_turns": loop_result["model_turns"],
        "tool_calls": loop_result["tool_calls"],
        "termination_reason": loop_result["termination_reason"],
        "error": loop_result["error"],
        "usage": usage,
        "deterministic_reward": result["deterministic_reward"],
        "deterministic_passed": result["deterministic_passed"],
        "agent_final_present": result["agent_final_present"],
        "outputs": result["outputs"],
        "cleanup": cleanup,
        "sandbox_id": result["sandbox_id"],
        "entry_url": result["world"]["entry_url"],
        "browser_use_cloud": False,
    }
    atomic_json(root / "summary.json", summary)
    hashes = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "artifact-hashes.json":
            continue
        payload = path.read_bytes()
        hashes.append({
            "path": str(path.relative_to(root)),
            "size": len(payload),
            "sha256": digest(payload),
        })
    atomic_json(root / "artifact-hashes.json", hashes)
    print(json.dumps(summary, indent=2))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-binary", type=Path)
    parser.add_argument("--customer-repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-id", default="fresh")
    parser.add_argument("--validate-loop", action="store_true")
    args = parser.parse_args()
    if args.validate_loop:
        await validate_loop(args.output.resolve())
        return
    if args.world_binary is None or args.customer_repo is None:
        parser.error("--world-binary and --customer-repo are required for a live rollout")
    await run_live(args)


if __name__ == "__main__":
    asyncio.run(main())
