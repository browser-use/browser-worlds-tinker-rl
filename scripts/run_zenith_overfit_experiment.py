"""Run the approved single-task Zenith Inkling overfitting experiment."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tinker
import torch
from tinker import TensorData, types
from tinker_cookbook.renderers import Message, get_text_content
from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from run_zenith_inkling_rollouts import BROWSER_TOOL, PROMPT, SYSTEM_PROMPT, clean_program

MODEL = "thinkingmachines/Inkling"
TASK = "zenith-zslr01"
TRAIN_ROLLOUTS = 8
EVAL_ROLLOUTS = 4
MAX_ITERATIONS = 10
EVAL_ITERATIONS = (2, 4, 6, 8, 10)
MAX_TRAINING_ROLLOUTS = 80
MAX_EVALUATION_ROLLOUTS = 20
MAX_TOKENS = 2750
ROLLOUT_TIMEOUT_SECONDS = 35 * 60
LEARNING_RATE = 1e-5
LORA_RANK = 32
MAX_INFRA_RERUNS = 1
WORLD_SEED = "411001"
BASELINE = {"mean": 0.30, "strict_passes": 0, "finals": 2, "rollouts": 4}


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


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def assert_clean(repo: Path, label: str) -> str:
    status = git(repo, "status", "--porcelain")
    if status:
        raise RuntimeError(f"{label} checkout is not clean")
    return git(repo, "rev-parse", "HEAD")


def validate(args: argparse.Namespace) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    if TRAIN_ROLLOUTS * MAX_ITERATIONS != MAX_TRAINING_ROLLOUTS:
        raise RuntimeError("training rollout cap mismatch")
    if EVAL_ROLLOUTS * len(EVAL_ITERATIONS) != MAX_EVALUATION_ROLLOUTS:
        raise RuntimeError("evaluation rollout cap mismatch")
    if EVAL_ITERATIONS != tuple(range(2, MAX_ITERATIONS + 1, 2)):
        raise RuntimeError("evaluation schedule mismatch")
    if MODEL != "thinkingmachines/Inkling":
        raise RuntimeError("model mismatch")
    world_binary = args.world_binary.resolve()
    customer_repo = args.customer_repo.resolve()
    if not world_binary.is_file() or not os.access(world_binary, os.X_OK):
        raise RuntimeError(f"world binary is missing or not executable: {world_binary}")
    task_dir = customer_repo / "tasks/zenith-zslr01"
    required_task_files = ("instruction.md", "task.toml", "tests/test.sh")
    if not all((task_dir / name).is_file() for name in required_task_files):
        raise RuntimeError("customer evaluator Zenith task package is incomplete")
    verifier = repo / "scripts/verify_zenith_daytona_sandbox.py"
    source = verifier.read_text()
    required = (
        'SNAPSHOT = "browser-rl-local-harness-f5eaf904-c2m4d4-v1"',
        '"BROWSER_USE_WORLDS_TASK_ID": "zslr01"',
        '"BROWSER_USE_WORLDS_SITE": "zenith"',
        '"browser_use_cloud": False',
        '"deterministic_reward"',
        '"outputs": outputs',
        'daytona.delete(sandbox, 120)',
    )
    missing = [fragment for fragment in required if fragment not in source]
    if missing:
        raise RuntimeError(f"evaluator contract fragments missing: {missing}")
    origin_main = git(repo, "rev-parse", "origin/main")
    head = git(repo, "rev-parse", "HEAD")
    if not args.allow_local_commit and head != origin_main:
        raise RuntimeError(f"target repo must start at pushed main: {head} != {origin_main}")
    customer_sha = assert_clean(customer_repo, "customer evaluator")
    return {
        "validated_at": utc_now(),
        "zero_provider_calls": True,
        "repo_head": head,
        "repo_origin_main": origin_main,
        "customer_repo_head": customer_sha,
        "world_binary": {
            "path": str(world_binary),
            "size": world_binary.stat().st_size,
            "sha256": digest(world_binary.read_bytes()),
        },
        "contract": experiment_contract(),
    }


def experiment_contract() -> dict[str, Any]:
    return {
        "model": MODEL,
        "task": TASK,
        "training_iterations_max": MAX_ITERATIONS,
        "training_rollouts_per_iteration": TRAIN_ROLLOUTS,
        "training_concurrency": TRAIN_ROLLOUTS,
        "evaluation_iterations": list(EVAL_ITERATIONS),
        "evaluation_rollouts_per_scheduled_evaluation": EVAL_ROLLOUTS,
        "evaluation_concurrency": EVAL_ROLLOUTS,
        "max_training_rollouts": MAX_TRAINING_ROLLOUTS,
        "max_evaluation_rollouts": MAX_EVALUATION_ROLLOUTS,
        "max_tokens_per_rollout": MAX_TOKENS,
        "generation_cap_evidence": {
            "valid_counts": [79, 265, 556, 1307],
            "valid_min": 79,
            "valid_median": 410.5,
            "valid_max": 1307,
            "policy": "2750 is just over 2x the valid maximum, rounded upward",
        },
        "length_truncation_policy": "scored_model_outcome_no_sampling_retry",
        "rollout_timeout_seconds": ROLLOUT_TIMEOUT_SECONDS,
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "optimizer_steps_per_iteration": 1,
        "deterministic_reward_only": True,
        "llm_judge": False,
        "browser_use_cloud": False,
        "baseline": BASELINE,
        "early_success": {
            "consecutive_scheduled_evaluations": 2,
            "mean_at_least": 0.80,
            "strict_passes_at_least": 3,
            "finals": 4,
        },
        "iteration_five_stop": {"training_mean_below": 0.50},
        "infrastructure_reruns_per_cell_max": MAX_INFRA_RERUNS,
    }


@dataclass
class Candidate:
    ordinal: int
    split: str
    iteration: int
    prompt: tinker.ModelInput
    tokens: list[int]
    logprobs: list[float]
    program: Path | None
    model_valid: bool
    model_error: str | None


async def sample_candidate(
    client: Any,
    renderer: TmlV0Renderer,
    tokenizer: Any,
    *,
    ordinal: int,
    split: str,
    iteration: int,
    root: Path,
) -> Candidate:
    rollout = root / f"rollout-{ordinal:02d}"
    rollout.mkdir(parents=True, exist_ok=False)
    messages = renderer.create_conversation_prefix_with_tools(
        tools=[BROWSER_TOOL], system_prompt=SYSTEM_PROMPT
    ) + [Message(role="user", content=PROMPT)]
    prompt = renderer.build_generation_prompt(messages, effort=0.7)
    seed = (881000 if split == "train" else 991000) + iteration * 100 + ordinal
    response = await client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=MAX_TOKENS,
            temperature=1.0,
            seed=seed,
            stop=renderer.get_stop_sequences(),
        ),
    )
    sequence = response.sequences[0]
    tokens = [int(value) for value in sequence.tokens]
    if sequence.logprobs is None:
        raise RuntimeError(f"{split} {iteration}/{ordinal} omitted sampling logprobs")
    logprobs = [float(value) for value in sequence.logprobs]
    if len(tokens) != len(logprobs):
        raise RuntimeError("completion token/logprob length mismatch")
    decoded = tokenizer.decode(tokens, skip_special_tokens=False)
    program_path: Path | None = None
    model_error: str | None = None
    tool_name: str | None = None
    tool_arguments_sha256: str | None = None
    termination: str | None = None
    try:
        message, termination_value = renderer.parse_response(tokens)
        termination = str(getattr(termination_value, "value", termination_value))
        tool_calls = list(message.get("tool_calls") or [])
        if len(tool_calls) != 1 or tool_calls[0].function.name != "browser_harness":
            names = [call.function.name for call in tool_calls]
            raise ValueError(f"expected one browser_harness tool call, observed {names}")
        tool_name = tool_calls[0].function.name
        arguments_text = tool_calls[0].function.arguments
        tool_arguments_sha256 = digest(arguments_text.encode())
        arguments = json.loads(arguments_text)
        program = clean_program(str(arguments["code"]))
        program_path = rollout / "inkling-program.py"
        program_path.write_text(program)
        program_path.chmod(0o600)
    except Exception as exc:
        model_error = f"{type(exc).__name__}: {exc}"
    atomic_json(
        rollout / "generation.json",
        {
            "split": split,
            "iteration": iteration,
            "ordinal": ordinal,
            "model": MODEL,
            "seed": seed,
            "effort": 0.7,
            "max_tokens": MAX_TOKENS,
            "prompt_tokens": prompt.length,
            "prompt_sha256": digest(PROMPT.encode()),
            "decoded_response": decoded,
            "decoded_response_sha256": digest(decoded.encode()),
            "completion_tokens": tokens,
            "completion_logprobs": logprobs,
            "generated_tokens": len(tokens),
            "stop_reason": str(sequence.stop_reason),
            "termination": termination,
            "prompt_cache_hit_tokens": int(response.prompt_cache_hit_tokens),
            "tool_name": tool_name,
            "tool_arguments_sha256": tool_arguments_sha256,
            "model_valid": program_path is not None,
            "model_error": model_error,
            "usage_or_cost_exposed": False,
        },
    )
    return Candidate(
        ordinal=ordinal,
        split=split,
        iteration=iteration,
        prompt=prompt,
        tokens=tokens,
        logprobs=logprobs,
        program=program_path,
        model_valid=program_path is not None,
        model_error=model_error,
    )


async def evaluator_attempt(
    candidate: Candidate,
    args: argparse.Namespace,
    rollout: Path,
    attempt: int,
) -> dict[str, Any]:
    attempt_dir = rollout / f"execution-attempt-{attempt:02d}"
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("verify_zenith_daytona_sandbox.py")),
        "--world-binary",
        str(args.world_binary),
        "--customer-repo",
        str(args.customer_repo),
        "--output",
        str(attempt_dir),
        "--program",
        str(candidate.program),
        "--rollout-id",
        f"{candidate.split}-i{candidate.iteration:02d}-r{candidate.ordinal:02d}-a{attempt:02d}",
        "--seed",
        WORLD_SEED,
    ]
    env = os.environ.copy()
    for key in (
        "BROWSER_USE_API_KEY",
        "BROWSER_USE_CLOUD_API_KEY",
        "BROWSER_USE_CLOUD_URL",
        "BROWSER_USE_API_URL",
    ):
        env.pop(key, None)
    started = asyncio.get_running_loop().time()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    timed_out = False
    try:
        stdout, _ = await asyncio.wait_for(
            proc.communicate(), timeout=ROLLOUT_TIMEOUT_SECONDS
        )
    except TimeoutError:
        timed_out = True
        proc.kill()
        stdout, _ = await proc.communicate()
    (rollout / f"evaluator-attempt-{attempt:02d}.log").write_bytes(stdout)
    result_path = attempt_dir / "result.json"
    cleanup_path = attempt_dir / "cleanup.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    cleanup = json.loads(cleanup_path.read_text()) if cleanup_path.exists() else None
    infrastructure_valid = bool(
        not timed_out
        and proc.returncode == 0
        and result
        and result.get("complete") is True
        and cleanup
        and cleanup.get("sandbox_deleted") is True
        and cleanup.get("snapshot_retained") is True
    )
    return {
        "attempt": attempt,
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "elapsed_seconds": round(asyncio.get_running_loop().time() - started, 3),
        "infrastructure_valid": infrastructure_valid,
        "result": result,
        "cleanup": cleanup,
        "error_tail": None
        if infrastructure_valid
        else stdout.decode(errors="replace")[-4000:],
    }


async def execute_candidate(
    candidate: Candidate,
    args: argparse.Namespace,
    root: Path,
    barrier: asyncio.Event,
) -> dict[str, Any]:
    await barrier.wait()
    rollout = root / f"rollout-{candidate.ordinal:02d}"
    if candidate.program is None:
        candidate.program = rollout / "model-invalid-program.py"
        candidate.program.write_text("# Preserve invalid or truncated model output as a scored outcome.\npass\n")
        candidate.program.chmod(0o600)
    attempts = []
    for attempt in range(1, MAX_INFRA_RERUNS + 2):
        receipt = await evaluator_attempt(candidate, args, rollout, attempt)
        attempts.append(receipt)
        if receipt["infrastructure_valid"]:
            break
    valid = bool(attempts[-1]["infrastructure_valid"])
    result = attempts[-1].get("result") or {}
    record = {
        "ordinal": candidate.ordinal,
        "model_valid": candidate.model_valid,
        "model_error": candidate.model_error,
        "infrastructure_valid": valid,
        "attempts": attempts,
        "reward": float(result.get("deterministic_reward", 0.0)) if valid else 0.0,
        "strict_pass": bool(result.get("deterministic_passed")) if valid else False,
        "final_present": bool(result.get("agent_final_present")) if valid else False,
    }
    atomic_json(rollout / "cell.json", record)
    return record


def batch_summary(split: str, iteration: int, records: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(record["reward"]) for record in records]
    return {
        "split": split,
        "iteration": iteration,
        "requested": len(records),
        "accounted": len(records),
        "infrastructure_valid": sum(int(record["infrastructure_valid"]) for record in records),
        "model_valid": sum(int(record["model_valid"]) for record in records),
        "mean_deterministic_reward": sum(rewards) / len(rewards),
        "strict_passes": sum(int(record["strict_pass"]) for record in records),
        "finals": sum(int(record["final_present"]) for record in records),
        "rewards": rewards,
        "records": records,
    }


async def run_batch(
    client: Any,
    renderer: TmlV0Renderer,
    tokenizer: Any,
    *,
    split: str,
    iteration: int,
    count: int,
    root: Path,
    args: argparse.Namespace,
) -> tuple[list[Candidate], dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=False)
    candidates = await asyncio.gather(
        *(
            sample_candidate(
                client,
                renderer,
                tokenizer,
                ordinal=ordinal,
                split=split,
                iteration=iteration,
                root=root,
            )
            for ordinal in range(1, count + 1)
        )
    )
    barrier = asyncio.Event()
    tasks = [
        asyncio.create_task(execute_candidate(candidate, args, root, barrier))
        for candidate in candidates
    ]
    barrier_receipt = {
        "released_at": utc_now(),
        "rollouts": count,
        "concurrency": count,
        "split": split,
        "iteration": iteration,
    }
    atomic_json(root / "barrier.json", barrier_receipt)
    barrier.set()
    records = await asyncio.gather(*tasks)
    summary = batch_summary(split, iteration, records)
    atomic_json(root / "summary.json", summary)
    return candidates, summary


def training_datums(candidates: list[Candidate], rewards: list[float]) -> list[tinker.Datum]:
    mean = sum(rewards) / len(rewards)
    datums = []
    for candidate, reward in zip(candidates, rewards, strict=True):
        advantage = reward - mean
        ob_len = candidate.prompt.length - 1
        model_input = candidate.prompt.append(
            types.EncodedTextChunk(tokens=candidate.tokens[:-1])
        )
        target_tokens = [0] * ob_len + candidate.tokens
        padded_logprobs = [0.0] * ob_len + candidate.logprobs
        padded_advantages = [0.0] * ob_len + [advantage] * (model_input.length - ob_len)
        if not (
            model_input.length
            == len(target_tokens)
            == len(padded_logprobs)
            == len(padded_advantages)
        ):
            raise RuntimeError("training datum length mismatch")
        datums.append(
            types.Datum(
                model_input=model_input,
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                    "logprobs": TensorData.from_torch(torch.tensor(padded_logprobs)),
                    "advantages": TensorData.from_torch(torch.tensor(padded_advantages)),
                },
            )
        )
    return datums


async def checkpoint(
    training_client: Any, root: Path, run_id: str, iteration: int
) -> tuple[Any, dict[str, Any]]:
    state_future = await training_client.save_state_async(
        f"{run_id}-iteration-{iteration:02d}-state"
    )
    sampler_future = await training_client.save_weights_for_sampler_async(
        f"{run_id}-iteration-{iteration:02d}-sampler"
    )
    state = await state_future.result_async()
    sampler = await sampler_future.result_async()
    sampling_client = await training_client.create_sampling_client_async(sampler.path)
    receipt = {
        "iteration": iteration,
        "saved_at": utc_now(),
        "state_path": state.path,
        "sampler_path": sampler.path,
    }
    atomic_json(root / "checkpoint.json", receipt)
    return sampling_client, receipt


def scheduled_success(summary: dict[str, Any]) -> bool:
    return bool(
        summary["mean_deterministic_reward"] >= 0.80
        and summary["strict_passes"] >= 3
        and summary["finals"] == 4
    )


def artifact_hashes(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "artifact-hashes.json":
            continue
        data = path.read_bytes()
        records.append(
            {"path": str(path.relative_to(root)), "size": len(data), "sha256": digest(data)}
        )
    return records


async def run(args: argparse.Namespace, validation: dict[str, Any]) -> Path:
    for key in ("TINKER_API_KEY", "DAYTONA_KEY"):
        if not os.environ.get(key, "").strip():
            raise RuntimeError(f"{key} is required")
    run_id = f"zenith-overfit-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    root = (args.output_parent.resolve() / run_id)
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "validation.json", validation)
    atomic_json(root / "contract.json", experiment_contract())
    renderer = TmlV0Renderer(get_tokenizer(MODEL))
    tokenizer = get_tokenizer(MODEL)
    service = tinker.ServiceClient(user_metadata={"recipe": "zenith_single_task_overfit"})
    training_client = await service.create_lora_training_client_async(
        base_model=MODEL, rank=LORA_RANK, seed=880808
    )
    sampling_client, initial_checkpoint = await checkpoint(training_client, root, run_id, 0)
    training_summaries = []
    evaluation_summaries = []
    checkpoints = [initial_checkpoint]
    training_rollouts = 0
    evaluation_rollouts = 0
    consecutive_successes = 0
    stop_reason = "maximum_iterations"
    optim_steps = 0
    for iteration in range(1, MAX_ITERATIONS + 1):
        candidates, train_summary = await run_batch(
            sampling_client,
            renderer,
            tokenizer,
            split="train",
            iteration=iteration,
            count=TRAIN_ROLLOUTS,
            root=root / f"iteration-{iteration:02d}" / "training",
            args=args,
        )
        training_rollouts += TRAIN_ROLLOUTS
        training_summaries.append(train_summary)
        if train_summary["infrastructure_valid"] != TRAIN_ROLLOUTS:
            stop_reason = f"unresolved_training_infrastructure_iteration_{iteration}"
            break
        datums = training_datums(candidates, train_summary["rewards"])
        fwd_future = await training_client.forward_backward_async(
            datums, loss_fn="importance_sampling"
        )
        optim_future = await training_client.optim_step_async(
            tinker.AdamParams(
                learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.95, eps=1e-8
            )
        )
        fwd = await fwd_future.result_async()
        optim = await optim_future.result_async()
        optim_steps += 1
        train_receipt = {
            "iteration": iteration,
            "datums": len(datums),
            "optimizer_steps": 1,
            "loss_fn": "importance_sampling",
            "learning_rate": LEARNING_RATE,
            "forward_backward_outputs": len(fwd.loss_fn_outputs),
            "optimizer_metrics": optim.metrics,
        }
        atomic_json(root / f"iteration-{iteration:02d}" / "optimizer.json", train_receipt)
        sampling_client, checkpoint_receipt = await checkpoint(
            training_client, root / f"iteration-{iteration:02d}", run_id, iteration
        )
        checkpoints.append(checkpoint_receipt)
        if iteration in EVAL_ITERATIONS:
            _, eval_summary = await run_batch(
                sampling_client,
                renderer,
                tokenizer,
                split="evaluation",
                iteration=iteration,
                count=EVAL_ROLLOUTS,
                root=root / f"iteration-{iteration:02d}" / "evaluation",
                args=args,
            )
            evaluation_rollouts += EVAL_ROLLOUTS
            evaluation_summaries.append(eval_summary)
            if eval_summary["infrastructure_valid"] != EVAL_ROLLOUTS:
                stop_reason = f"unresolved_evaluation_infrastructure_iteration_{iteration}"
                break
            consecutive_successes = (
                consecutive_successes + 1 if scheduled_success(eval_summary) else 0
            )
            if consecutive_successes >= 2:
                stop_reason = f"early_success_after_iteration_{iteration}"
                break
        if iteration == 5 and train_summary["mean_deterministic_reward"] < 0.50:
            stop_reason = "iteration_5_training_mean_below_0.50"
            break
        atomic_json(
            root / "progress.json",
            {
                "updated_at": utc_now(),
                "completed_iterations": iteration,
                "training_rollouts": training_rollouts,
                "evaluation_rollouts": evaluation_rollouts,
                "optimizer_steps": optim_steps,
            },
        )
    failures = Counter()
    for summary in [*training_summaries, *evaluation_summaries]:
        for record in summary["records"]:
            if not record["model_valid"]:
                failures["model_invalid_tool_call"] += 1
            elif not record["infrastructure_valid"]:
                failures["infrastructure_invalid"] += 1
            elif not record["strict_pass"]:
                failures["deterministic_non_pass"] += 1
            if not record["final_present"]:
                failures["missing_final"] += 1
    report = {
        "run_id": run_id,
        "completed_at": utc_now(),
        "stop_reason": stop_reason,
        "model": MODEL,
        "baseline": BASELINE,
        "training_rollouts": training_rollouts,
        "evaluation_rollouts": evaluation_rollouts,
        "optimizer_steps": optim_steps,
        "training": training_summaries,
        "evaluations": evaluation_summaries,
        "checkpoints": checkpoints,
        "diagnostics": dict(failures),
        "llm_judge": False,
        "browser_use_cloud": False,
        "usage_or_cost_exposed_by_sampling_response": False,
    }
    atomic_json(root / "report.json", report)
    atomic_json(root / "artifact-hashes.json", artifact_hashes(root))
    print(json.dumps({"root": str(root), **report}, indent=2))
    return root


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-binary", type=Path, required=True)
    parser.add_argument("--customer-repo", type=Path, required=True)
    parser.add_argument("--output-parent", type=Path, default=Path("/tmp"))
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-local-commit", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    validation = validate(args)
    if args.validate_only:
        print(json.dumps(validation, indent=2))
        return
    await run(args, validation)


if __name__ == "__main__":
    asyncio.run(main())
