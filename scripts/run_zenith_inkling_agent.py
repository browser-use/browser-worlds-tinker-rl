"""Run one grounded multi-turn Inkling agent against one local Zenith sandbox."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Awaitable, Callable

import tinker
from tinker_cookbook.renderers import Message, ToolCall, get_renderer, get_text_content
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

DEFAULT_MODEL = "thinkingmachines/Inkling"
SUPPORTED_MODEL_RENDERERS = {
    "thinkingmachines/Inkling": "TmlV0Renderer",
    "Qwen/Qwen3.6-27B": "qwen3_5_disable_thinking",
    "Qwen/Qwen3.6-35B-A3B": "qwen3_5",
}
QWEN27_MODEL = "Qwen/Qwen3.6-27B"
QWEN35_MODEL = "Qwen/Qwen3.6-35B-A3B"
THINKING_CHOICES = ("model-default", "enabled", "disabled")
TASK = "zenith-zslr01"
TASK_INSTRUCTION_PATH = "harbor/tasks/zenith-zslr01/instruction.md"
DEFAULT_RUNS = 1
DEFAULT_CONCURRENCY = 1
DEFAULT_MAX_GENERATED_TOKENS = 32000
DEFAULT_TIMEOUT_SECONDS = 1200
EFFORT = 0.7
DEFAULT_SEEDS = [991001]
WORLD_SEED = 411001
SHARED_SNAPSHOT = "browser-rl-local-harness-f5eaf904-c2m4d4-v1"
PROTOCOL_READER_LIMIT = 16 * 1024 * 1024
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


def browser_harness_result_is_error(result_text: str) -> bool:
    try:
        result = json.loads(result_text)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(result, dict):
        return False
    return bool(
        result.get("exit_code")
        or result.get("evidence_exit_code")
        or result.get("error")
    )


@dataclass
class SampledTurn:
    message: Message
    generated_tokens: int
    termination: str
    record: dict[str, Any]


SampleTurn = Callable[[list[Message], int, int], Awaitable[SampledTurn]]
ExecuteTool = Callable[[str, int, int], Awaitable[str]]


def renderer_for_model(model: str, thinking: str = "model-default") -> tuple[str, Any]:
    renderer_name = SUPPORTED_MODEL_RENDERERS[model]
    if model == QWEN27_MODEL and thinking != "model-default":
        renderer_name = "qwen3_5" if thinking == "enabled" else "qwen3_5_disable_thinking"
    elif model == QWEN35_MODEL and thinking == "enabled":
        renderer_name = "qwen3_5"
    elif thinking != "model-default":
        raise ValueError(f"--thinking {thinking} is not supported for {model}")
    tokenizer = get_tokenizer(model)
    if renderer_name == "TmlV0Renderer":
        return renderer_name, TmlV0Renderer(tokenizer)
    return renderer_name, get_renderer(renderer_name, tokenizer)


def build_generation_prompt(renderer: Any, messages: list[Message], model: str):
    if model == "thinkingmachines/Inkling":
        return renderer.build_generation_prompt(messages, effort=EFFORT)
    return renderer.build_generation_prompt(messages)


def is_model_context_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        ("context window" in message or "context limit" in message)
        and "prompt" in message
        and ("max_tokens" in message or "max tokens" in message)
    )


async def run_agent_loop(
    initial_messages: list[Message],
    sample_turn: SampleTurn,
    execute_tool: ExecuteTool,
    max_generated_tokens: int,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    messages = list(initial_messages)
    events: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []
    total_generated = 0
    model_turns = 0
    attempted_tool_calls = 0
    tool_calls = 0
    error_feedback_turns = 0
    final_response = ""
    termination_reason = "irrecoverable_model_error"
    error = None
    deadline = time.monotonic() + timeout_seconds if timeout_seconds else None

    while total_generated < max_generated_tokens:
        model_turns += 1
        remaining = max_generated_tokens - total_generated
        try:
            if deadline is None:
                sampled = await sample_turn(messages, remaining, model_turns)
            else:
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    termination_reason = f"rollout_timeout_{timeout_seconds}"
                    break
                sampled = await asyncio.wait_for(
                    sample_turn(messages, remaining, model_turns),
                    timeout=remaining_seconds,
                )
        except TimeoutError:
            termination_reason = f"rollout_timeout_{timeout_seconds}"
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            termination_reason = (
                "model_context_limit"
                if is_model_context_limit_error(exc)
                else "irrecoverable_infrastructure_error"
            )
            break
        if sampled.generated_tokens <= 0 or sampled.generated_tokens > remaining:
            error = (
                f"invalid generated token count {sampled.generated_tokens} "
                f"with {remaining} remaining"
            )
            termination_reason = "irrecoverable_infrastructure_error"
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

        if total_generated == max_generated_tokens:
            termination_reason = f"generation_budget_{max_generated_tokens}"
            break
        unparsed = list(sampled.message.get("unparsed_tool_calls") or [])
        if unparsed:
            attempted_tool_calls += len(unparsed)
            feedback = "; ".join(value.error for value in unparsed)
            messages.append(Message(role="user", content=feedback))
            error_feedback_turns += 1
            events.append({
                "type": "error_feedback",
                "turn": model_turns,
                "call": None,
                "stage": "unparsed_tool_call",
                "content": feedback,
            })
            continue
        if sampled.termination == "malformed":
            feedback = "model response was malformed before the generation budget was exhausted"
            messages.append(Message(role="user", content=feedback))
            error_feedback_turns += 1
            events.append({
                "type": "error_feedback",
                "turn": model_turns,
                "call": None,
                "stage": "malformed_response",
                "content": feedback,
            })
            continue
        calls = list(sampled.message.get("tool_calls") or [])
        if not calls:
            termination_reason = "final_answer" if final_response.strip() else "explicit_stop"
            break

        recoverable_error = False
        for call_index, call in enumerate(calls, start=1):
            attempted_tool_calls += 1
            if call.function.name != "browser_harness":
                feedback = f"unknown tool {call.function.name!r}"
                messages.append(Message(
                    role="tool",
                    content=feedback,
                    tool_call_id=call.id or "",
                    name=call.function.name,
                ))
                error_feedback_turns += 1
                events.append({
                    "type": "error_feedback",
                    "turn": model_turns,
                    "call": call_index,
                    "stage": "unknown_tool",
                    "tool_call_id": call.id,
                    "content": feedback,
                })
                recoverable_error = True
                break
            try:
                arguments = json.loads(call.function.arguments)
                if (
                    not isinstance(arguments, dict)
                    or set(arguments) != {"code"}
                    or not isinstance(arguments["code"], str)
                ):
                    raise ValueError(
                        "arguments must be an object containing only a string 'code' field"
                    )
                code = clean_program(arguments["code"])
            except Exception as exc:
                feedback = (
                    f"invalid browser_harness arguments: {type(exc).__name__}: {exc}"
                )
                messages.append(Message(
                    role="tool",
                    content=feedback,
                    tool_call_id=call.id or "",
                    name="browser_harness",
                ))
                error_feedback_turns += 1
                events.append({
                    "type": "error_feedback",
                    "turn": model_turns,
                    "call": call_index,
                    "stage": "invalid_arguments",
                    "tool_call_id": call.id,
                    "content": feedback,
                })
                recoverable_error = True
                break
            try:
                result_text = await execute_tool(code, model_turns, call_index)
            except Exception as exc:
                error = f"browser_harness protocol failed: {type(exc).__name__}: {exc}"
                termination_reason = "irrecoverable_infrastructure_error"
                break
            tool_calls += 1
            result_is_error = browser_harness_result_is_error(result_text)
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
                "is_error": result_is_error,
            })
            if result_is_error:
                error_feedback_turns += 1
                recoverable_error = True
                break
        if error is not None:
            break
        if recoverable_error:
            continue

    return {
        "termination_reason": termination_reason,
        "error": error,
        "final_response": final_response,
        "model_turns": model_turns,
        "attempted_tool_calls": attempted_tool_calls,
        "tool_calls": tool_calls,
        "error_feedback_turns": error_feedback_turns,
        "total_generated_tokens": total_generated,
        "messages": [jsonable_message(message) for message in messages],
        "events": events,
        "generations": generations,
    }


def initial_messages(renderer: Any, observation: str) -> list[Message]:
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
    exact_tool_result = (
        '{"exit_code":0,"stdout":"ok\\n","error":null,"evidence_exit_code":0,'
        '"evidence_stdout":"","page_info":{"url":"http://127.0.0.1/task",'
        '"title":"Zenith"},"screenshot":"/tmp/outputs/turn.png"}'
    )
    model_results = []
    renderer_selections = [
        ("thinkingmachines/Inkling", "model-default", "TmlV0Renderer"),
        (QWEN27_MODEL, "disabled", "qwen3_5_disable_thinking"),
        (QWEN27_MODEL, "enabled", "qwen3_5"),
        (QWEN35_MODEL, "enabled", "qwen3_5"),
    ]
    for model, thinking, expected_renderer in renderer_selections:
        renderer_name, renderer = renderer_for_model(model, thinking)
        assert renderer_name == expected_renderer
        observed_remaining = []
        executed = []

        async def fake_sample(
            messages: list[Message], remaining: int, turn: int
        ) -> SampledTurn:
            build_generation_prompt(renderer, messages, model)
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

        async def fake_execute(code: str, turn: int, call: int) -> str:
            executed.append({"code": code, "turn": turn, "call": call})
            return exact_tool_result

        result = await run_agent_loop(
            initial_messages(
                renderer,
                '{"page_info":{"url":"http://127.0.0.1/task","title":"Zenith"}}',
            ),
            fake_sample,
            fake_execute,
            DEFAULT_MAX_GENERATED_TOKENS,
        )
        assert result["termination_reason"] == "final_answer"
        assert result["model_turns"] == 2
        assert result["attempted_tool_calls"] == 1
        assert result["tool_calls"] == 1
        assert result["error_feedback_turns"] == 0
        assert result["total_generated_tokens"] == 14
        assert observed_remaining == [32000, 31992]
        assert executed == [{"code": "print(page_info())\n", "turn": 1, "call": 1}]
        assert result["messages"][-2]["content"] == exact_tool_result
        model_results.append({
            "model": model,
            "thinking": thinking,
            "renderer": renderer_name,
            "result": result,
        })

    semaphore = asyncio.Semaphore(2)
    active = 0
    peak_active = 0
    accounted = []
    lock = asyncio.Lock()

    async def fake_rollout(ordinal: int) -> None:
        nonlocal active, peak_active
        async with semaphore:
            async with lock:
                active += 1
                peak_active = max(peak_active, active)
            await asyncio.sleep(0)
            accounted.append(ordinal)
            async with lock:
                active -= 1

    await asyncio.gather(*(fake_rollout(ordinal) for ordinal in range(1, 5)))
    assert sorted(accounted) == [1, 2, 3, 4]
    assert peak_active == 2

    oversized_result = exact_tool_result + ("\nexact oversized evidence ☃" * 8192)
    framed = (json.dumps({
        "type": "tool_result",
        "turn": 4,
        "call": 1,
        "result": oversized_result,
    }, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=PROTOCOL_READER_LIMIT,
    )
    assert proc.stdin is not None
    proc.stdin.write(framed)
    await proc.stdin.drain()
    proc.stdin.close()
    round_tripped = await read_protocol_line(proc)
    stderr = await proc.stderr.read() if proc.stderr is not None else b""
    assert await proc.wait() == 0, stderr.decode(errors="replace")
    assert round_tripped["result"] == oversized_result
    oversized_validation = {
        "framed_bytes": len(framed),
        "framed_sha256": digest(framed),
        "result_bytes": len(oversized_result.encode()),
        "result_sha256": digest(oversized_result.encode()),
        "round_trip_exact": True,
        "reader_limit": PROTOCOL_READER_LIMIT,
    }
    receipt = {
        "validated_at": utc_now(),
        "zero_provider_calls": True,
        "zero_sandboxes": True,
        "model_renderer_results": model_results,
        "concurrency_accounting": {
            "requested": 4,
            "accounted": len(accounted),
            "concurrency": 2,
            "peak_active": peak_active,
        },
        "oversized_protocol_round_trip": oversized_validation,
    }
    atomic_json(output / "validation.json", receipt)
    print(json.dumps(receipt, indent=2))


async def read_protocol_line(proc: asyncio.subprocess.Process) -> dict[str, Any]:
    assert proc.stdout is not None
    line = await proc.stdout.readline()
    if not line:
        raise RuntimeError("interactive verifier closed its protocol stream")
    return json.loads(line)


async def run_rollout(
    args: argparse.Namespace,
    root: Path,
    ordinal: int,
    sampling_seed: int,
    client: Any,
    renderer: Any,
    renderer_name: str,
    attempt: int = 1,
) -> dict[str, Any]:
    rollout_root = root / f"rollout-{ordinal:02d}"
    rollout_root.mkdir()
    execution = rollout_root / "execution"
    task_id = TASK.removeprefix("zenith-")
    rollout_id = f"{task_id}-r{ordinal:02d}-a{attempt:02d}"
    command = [
        sys.executable,
        str(Path(__file__).with_name("verify_zenith_daytona_sandbox.py")),
        "--world-binary", str(args.world_binary.resolve()),
        "--world-sha256", digest(args.world_binary.read_bytes()),
        "--customer-repo", str(args.customer_repo.resolve()),
        "--output", str(execution),
        "--rollout-id", rollout_id,
        "--task-id", task_id,
        "--seed", str(WORLD_SEED),
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
        limit=PROTOCOL_READER_LIMIT,
    )
    ready = await read_protocol_line(proc)
    if ready.get("type") != "ready":
        raise RuntimeError(f"unexpected verifier handshake: {ready}")
    atomic_json(rollout_root / "initial-grounding.json", ready)

    tokenizer = get_tokenizer(args.model)

    async def sample_turn(messages: list[Message], remaining: int, turn: int) -> SampledTurn:
        prompt = build_generation_prompt(renderer, messages, args.model)
        response = await client.sample_async(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=remaining,
                temperature=1.0,
                seed=sampling_seed + turn - 1,
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
            "seed": sampling_seed + turn - 1,
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
        atomic_json(rollout_root / "generations" / f"turn-{turn:04d}.json", record)
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
        args.max_tokens,
        args.timeout,
    )
    atomic_json(rollout_root / "trace.json", loop_result)
    usage = {
        "model_turns": loop_result["model_turns"],
        "attempted_tool_calls": loop_result["attempted_tool_calls"],
        "tool_calls": loop_result["tool_calls"],
        "error_feedback_turns": loop_result["error_feedback_turns"],
        "generated_tokens": loop_result["total_generated_tokens"],
        "prompt_tokens_sum": sum(
            int(record["prompt_tokens"]) for record in loop_result["generations"]
        ),
        "prompt_cache_hit_tokens_sum": sum(
            int(record["prompt_cache_hit_tokens"]) for record in loop_result["generations"]
        ),
        "generation_cap": args.max_tokens,
        "usage_or_cost_exposed": False,
    }
    assert proc.stdin is not None
    proc.stdin.write((json.dumps({
        "type": "finish",
        "final_response": loop_result["final_response"],
        "termination_reason": loop_result["termination_reason"],
        "usage": usage,
        "rollout_timeout_seconds": args.timeout,
    }, separators=(",", ":")) + "\n").encode())
    await proc.stdin.drain()
    proc.stdin.close()
    stderr = await proc.stderr.read() if proc.stderr is not None else b""
    return_code = await proc.wait()
    (rollout_root / "verifier.stderr.txt").write_bytes(stderr)
    if return_code:
        raise RuntimeError(
            f"interactive verifier failed ({return_code}): {stderr.decode(errors='replace')[-4000:]}"
        )
    result = json.loads((execution / "result.json").read_text())
    cleanup = json.loads((execution / "cleanup.json").read_text())
    summary = {
        "ordinal": ordinal,
        "attempt": attempt,
        "rollout_id": rollout_id,
        "artifact_root": str(rollout_root),
        "completed_at": utc_now(),
        "model": args.model,
        "thinking": args.thinking,
        "renderer": renderer_name,
        "sampling_seed": sampling_seed,
        "world_seed": WORLD_SEED,
        "task": TASK,
        "model_turns": loop_result["model_turns"],
        "attempted_tool_calls": loop_result["attempted_tool_calls"],
        "tool_calls": loop_result["tool_calls"],
        "error_feedback_turns": loop_result["error_feedback_turns"],
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
        "cost": None,
        "cost_returned": False,
        "browser_use_cloud": False,
    }
    atomic_json(rollout_root / "summary.json", summary)
    hashes = []
    for path in sorted(item for item in rollout_root.rglob("*") if item.is_file()):
        if path.name == "artifact-hashes.json":
            continue
        payload = path.read_bytes()
        hashes.append({
            "path": str(path.relative_to(rollout_root)),
            "size": len(payload),
            "sha256": digest(payload),
        })
    atomic_json(rollout_root / "artifact-hashes.json", hashes)
    return summary


async def run_live(args: argparse.Namespace) -> None:
    if not os.environ.get("TINKER_API_KEY", "").strip():
        raise RuntimeError("TINKER_API_KEY is required")
    if not os.environ.get("DAYTONA_KEY", "").strip():
        raise RuntimeError("DAYTONA_KEY is required")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    customer = args.customer_repo.resolve()
    instruction = (customer / TASK_INSTRUCTION_PATH).read_text().strip()
    if instruction != TASK_INSTRUCTION:
        raise RuntimeError("Zenith task instruction changed")
    renderer_name, renderer = renderer_for_model(args.model, args.thinking)
    service = tinker.ServiceClient(user_metadata={"recipe": "zenith_multi_turn_agent"})
    client = await service.create_sampling_client_async(base_model=args.model)
    atomic_json(root / "contract.json", {
        "started_at": utc_now(),
        "model": args.model,
        "thinking": args.thinking,
        "renderer": renderer_name,
        "task": TASK,
        "rollouts": args.runs,
        "concurrency": args.concurrency,
        "max_total_generated_tokens_per_rollout": args.max_tokens,
        "rollout_timeout_seconds": args.timeout,
        "sampling_seeds": args.seeds,
        "world_seed": WORLD_SEED,
        "shared_snapshot": SHARED_SNAPSHOT,
        "system_prompt_sha256": digest(SYSTEM_PROMPT.encode()),
        "browser_harness_skill_sha256": digest(BROWSER_HARNESS_SKILL.encode()),
        "task_instruction": TASK_INSTRUCTION,
        "browser_use_cloud": False,
        "sampling_retries": 0,
        "rollout_retries": 0,
    })

    semaphore = asyncio.Semaphore(args.concurrency)
    barrier = asyncio.Event()
    state_lock = asyncio.Lock()
    active = 0
    peak_active = 0

    async def accounted_rollout(ordinal: int, seed: int) -> dict[str, Any]:
        nonlocal active, peak_active
        await barrier.wait()
        async with semaphore:
            async with state_lock:
                active += 1
                peak_active = max(peak_active, active)
            try:
                return await run_rollout(
                    args, root, ordinal, seed, client, renderer, renderer_name
                )
            except Exception as exc:
                rollout_root = root / f"rollout-{ordinal:02d}"
                cleanup_path = rollout_root / "execution" / "cleanup.json"
                record = {
                    "ordinal": ordinal,
                    "artifact_root": str(rollout_root),
                    "model": args.model,
                    "thinking": args.thinking,
                    "renderer": renderer_name,
                    "sampling_seed": seed,
                    "world_seed": WORLD_SEED,
                    "error": f"{type(exc).__name__}: {exc}",
                    "cleanup": (
                        json.loads(cleanup_path.read_text()) if cleanup_path.exists() else None
                    ),
                }
                atomic_json(rollout_root / "summary.json", record)
                return record
            finally:
                async with state_lock:
                    active -= 1

    tasks = [
        asyncio.create_task(accounted_rollout(ordinal, seed))
        for ordinal, seed in enumerate(args.seeds, start=1)
    ]
    released_at = utc_now()
    atomic_json(root / "barrier.json", {
        "released_at": released_at,
        "runs": args.runs,
        "concurrency": args.concurrency,
    })
    barrier.set()
    rows = await asyncio.gather(*tasks)
    rewards = [
        float(row["deterministic_reward"])
        for row in rows
        if "deterministic_reward" in row
    ]
    passed = sum(int(bool(row.get("deterministic_passed"))) for row in rows)
    cleaned = sum(
        int(bool((row.get("cleanup") or {}).get("sandbox_deleted"))) for row in rows
    )
    summary = {
        "artifact_root": str(root),
        "completed_at": utc_now(),
        "model": args.model,
        "thinking": args.thinking,
        "renderer": renderer_name,
        "task": TASK,
        "requested": args.runs,
        "accounted": len(rows),
        "completed": len(rewards),
        "concurrency": args.concurrency,
        "peak_active": peak_active,
        "max_tokens_per_rollout": args.max_tokens,
        "rollout_timeout_seconds": args.timeout,
        "strict_passes": passed,
        "strict_pass_rate": passed / args.runs,
        "mean_deterministic_reward": sum(rewards) / args.runs,
        "sandboxes_deleted": cleaned,
        "snapshot_retained": True,
        "shared_snapshot": SHARED_SNAPSHOT,
        "browser_use_cloud": False,
        "cost": None,
        "cost_returned": False,
        "rows": rows,
    }
    atomic_json(root / "summary.json", summary)
    all_hashes = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "all-artifact-hashes.json":
            continue
        payload = path.read_bytes()
        all_hashes.append({
            "path": str(path.relative_to(root)),
            "size": len(payload),
            "sha256": digest(payload),
        })
    atomic_json(root / "all-artifact-hashes.json", all_hashes)
    print(json.dumps(summary, indent=2))
    if len(rewards) != args.runs or cleaned != args.runs:
        raise RuntimeError("one or more rollouts failed or did not clean up")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=tuple(SUPPORTED_MODEL_RENDERERS), default=DEFAULT_MODEL
    )
    parser.add_argument("--thinking", choices=THINKING_CHOICES, default="model-default")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_GENERATED_TOKENS)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--world-binary", type=Path)
    parser.add_argument("--customer-repo", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-loop", action="store_true")
    args = parser.parse_args()
    try:
        args.seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    except ValueError:
        parser.error("--seeds must be a comma-separated list of integers")
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.concurrency <= 0 or args.concurrency > args.runs:
        parser.error("--concurrency must be between 1 and --runs")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if len(args.seeds) != args.runs or len(set(args.seeds)) != args.runs:
        parser.error("--seeds must contain exactly --runs unique integers")
    if args.validate_loop:
        await validate_loop(args.output.resolve())
        return
    if args.world_binary is None or args.customer_repo is None:
        parser.error("--world-binary and --customer-repo are required for a live rollout")
    await run_live(args)


if __name__ == "__main__":
    asyncio.run(main())
