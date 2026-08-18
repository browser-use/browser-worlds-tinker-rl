"""Verify Zenith World, Chromium, and Browser Harness in one Daytona sandbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from daytona import (
    CreateSandboxFromSnapshotParams,
    Daytona,
    DaytonaConfig,
    DaytonaNotFoundError,
)

SNAPSHOT = "browser-rl-local-harness-f5eaf904-c2m4d4-v1"
LABEL = "browser-use.browser-rl-local-harness"
PINNED_WORLD_BINARY_SHA256 = (
    "41bae34a7b46e7415ef911a767bd637b0c73daed379a4ec5f57d20ffad9c786f"
)
SOURCE_SCOPE = "hosted_worlds/go_sites/zenith"
TASK_PACKAGE_VERSION = "1.0.0"
TASK_IDS = ("zdib01", "zeko01", "zflt01", "zpal01", "zslr01")
INTERACTION_SKILLS_SOURCE_COMMIT = "f5eaf904b221dde0118eba1496961c3dc20fda88"
INTERACTION_SKILLS_LOCAL_DIR = (
    Path(__file__).resolve().parents[1] / "skills/browser-harness/interaction-skills"
)
INTERACTION_SKILLS_REMOTE_DIR = "/tmp/browser-harness-interaction-skills"
INTERACTION_SKILL_SHA256 = {
    "cookies.md": "cce3f46cf48b4662a05eee22c79278112ed12737277de42a1f6252e8ad9e09a8",
    "cross-origin-iframes.md": "c8bb661316b137257a5f36c0febb7abbcfd84b8baa41bf6127578e8db73cfa06",
    "dialogs.md": "b485e6978024dfbd8c0effd8e0635a4212ccb5732874f601ac525f17e0504d67",
    "downloads.md": "cf69b75bfb9cf84a876504ccfef0e9a37b60b60efa5db0aa5acf81644999ee30",
    "drag-and-drop.md": "eb2c567770ea51cdfa51ac9d00c8f7f070f211a978de8740c35a4f95c288c847",
    "dropdowns.md": "9bf8cbab80425a91898a574eacebbabf83b9d28a7181d88eeecb6f98b430b5f5",
    "iframes.md": "05d430364fc39f5778c8fd5d2da8ac1dae4ab69d0245126bc206a5d53cce30d5",
    "network-requests.md": "a90e83dae83bcf3c9f74e483f1b5bb68e2e43eae5e9fb0bdae8b6ba05e282f4d",
    "print-as-pdf.md": "73381a5e0e93d159aeab5afca4326e44eff622d2dd2a0a9d7a9e5d38e22d8429",
    "screenshots.md": "d630a07c3a7338d0699f1a15216c9dff8a7c334432ac359a7316db5e347c6586",
    "scrolling.md": "517ea4301c15d28655bc9ec2fd04ca3016d5201e2e1a91f5ecc2a88366eb34ce",
    "shadow-dom.md": "6f9af297b220f27b82b91d75d7dd49a70c0f087de5da64a24acaadf574b60fe9",
    "tabs.md": "58eff5821efc579be63c1248064e6b15ca530bb3d3424d4418618ce984fbcb6f",
    "uploads.md": "c45219a4f08287ee76eb363574dee81bc3ca2209033184aa5e23e9fdd430f1f9",
    "viewport.md": "4a824256c56d7fe4695a9ab82d5ca6cf58f692ee8a694c48367eb3dc52267dae",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_provenance(repo: Path, scope: str = SOURCE_SCOPE) -> dict[str, object]:
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--binary", "--no-ext-diff", "HEAD", "--", scope],
        check=True,
        capture_output=True,
    ).stdout
    untracked_raw = subprocess.run(
        [
            "git", "-C", str(repo), "ls-files", "--others", "--exclude-standard",
            "-z", "--", scope,
        ],
        check=True,
        capture_output=True,
    ).stdout
    untracked_names = sorted(
        name.decode("utf-8", errors="surrogateescape")
        for name in untracked_raw.split(b"\0")
        if name
    )
    scoped_delta = bytearray(b"tracked-diff\0")
    scoped_delta.extend(tracked_diff)
    for name in untracked_names:
        payload = (repo / name).read_bytes()
        scoped_delta.extend(b"\0untracked\0")
        scoped_delta.extend(name.encode("utf-8", errors="surrogateescape"))
        scoped_delta.extend(b"\0")
        scoped_delta.extend(str(len(payload)).encode())
        scoped_delta.extend(b"\0")
        scoped_delta.extend(payload)
    dirty = bool(tracked_diff or untracked_names)
    diff_hash = digest(bytes(scoped_delta)) if dirty else None
    identity = head if not dirty else f"{head}+dirty.sha256.{diff_hash}"
    return {
        "head": head,
        "scope": scope,
        "dirty": dirty,
        "scoped_diff_sha256": diff_hash,
        "identity": identity,
    }


def canonical_harbor_package_digest(customer_repo: Path, task_package: str) -> str:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to invoke the repository Harbor Packager")
    task_dir = customer_repo / task_package
    code = (
        "from pathlib import Path\n"
        "from harbor.publisher.packager import Packager\n"
        "value,_=Packager.compute_content_hash(Path(__import__('sys').argv[1]))\n"
        "print(value)\n"
    )
    result = subprocess.run(
        [
            uv, "run", "--frozen", "--project", str(customer_repo), "python", "-c",
            code, str(task_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"Harbor Packager returned an invalid digest: {value!r}")
    return value


def runtime_provenance(
    binary: Path,
    customer_repo: Path,
    task_id: str,
    expected_binary_sha256: str,
) -> dict[str, object]:
    binary_bytes = binary.read_bytes()
    binary_hash = digest(binary_bytes)
    if binary_hash != expected_binary_sha256:
        raise RuntimeError(
            "binary SHA-256 mismatch: "
            f"expected {expected_binary_sha256}, got {binary_hash}"
        )
    task_package = f"harbor/tasks/zenith-{task_id}"
    manifest = tomllib.loads(
        (customer_repo / task_package / "task.toml").read_text(encoding="utf-8")
    )
    version = str(manifest.get("task", {}).get("version", ""))
    if version != TASK_PACKAGE_VERSION:
        raise RuntimeError(
            f"Zenith {task_id} package version mismatch: expected {TASK_PACKAGE_VERSION}, got {version}"
        )
    package_digest = canonical_harbor_package_digest(customer_repo, task_package)
    return {
        "source": source_provenance(customer_repo),
        "task_package": {
            "path": task_package,
            "version": version,
            "digest": f"sha256:{package_digest}",
            "digest_algorithm": "Harbor Packager.compute_content_hash",
        },
        "binary": {
            "path": str(binary),
            "size": len(binary_bytes),
            "sha256": binary_hash,
            "pinned": True,
        },
    }


def ok(result, stage: str) -> str:
    text = str(result.result or "")
    if result.exit_code:
        raise RuntimeError(f"{stage} failed ({result.exit_code}): {text[-4000:]}")
    return text


def emit_protocol(value: dict) -> None:
    sys.stdout.write(json.dumps(value, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def upload_interaction_skills(sandbox) -> dict[str, dict[str, int | str]]:
    local_names = {
        path.name for path in INTERACTION_SKILLS_LOCAL_DIR.iterdir() if path.is_file()
    }
    expected_names = set(INTERACTION_SKILL_SHA256)
    if local_names != expected_names:
        raise RuntimeError(
            "interaction skill files do not match the pinned manifest: "
            f"expected={sorted(expected_names)} actual={sorted(local_names)}"
        )
    payloads = {}
    for name, expected_hash in INTERACTION_SKILL_SHA256.items():
        payload = (INTERACTION_SKILLS_LOCAL_DIR / name).read_bytes()
        actual_hash = digest(payload)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"interaction skill source hash mismatch for {name}: {actual_hash}"
            )
        payloads[name] = payload

    ok(sandbox.process.exec(
        f"rm -rf {INTERACTION_SKILLS_REMOTE_DIR} && "
        f"mkdir -p {INTERACTION_SKILLS_REMOTE_DIR}",
        timeout=30,
    ), "interaction skill directory")
    for name, payload in payloads.items():
        sandbox.fs.upload_file(
            payload,
            f"{INTERACTION_SKILLS_REMOTE_DIR}/{name}",
            timeout=30,
        )

    readback = json.loads(ok(sandbox.process.exec(
        "python3 - <<'PY'\n"
        "import hashlib,json,pathlib\n"
        f"root=pathlib.Path({INTERACTION_SKILLS_REMOTE_DIR!r})\n"
        "files={}\n"
        "for path in sorted(root.iterdir()):\n"
        "    payload=path.read_bytes()\n"
        "    files[path.name]={'size':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}\n"
        "print(json.dumps(files,sort_keys=True,separators=(',',':')))\n"
        "PY",
        timeout=30,
    ), "interaction skill readback"))
    expected_readback = {
        name: {"size": len(payloads[name]), "sha256": source_hash}
        for name, source_hash in INTERACTION_SKILL_SHA256.items()
    }
    if readback != expected_readback:
        raise RuntimeError(
            f"interaction skill sandbox readback mismatch: {readback}"
        )
    return readback


def execute_harness_turn(sandbox, out: Path, label: str, code: str) -> str:
    turns = out / "turns"
    turns.mkdir(exist_ok=True)
    (turns / f"{label}.program.py").write_text(code)
    remote_program = f"/tmp/browser-harness-{label}.py"
    sandbox.fs.upload_file(code.encode(), remote_program, timeout=30)
    result = sandbox.process.exec(
        "bash -lc 'BU_CDP_URL=http://127.0.0.1:9222 BH_REQUIRE_REMOTE=1 BH_RECORD=1 "
        "BH_RUNTIME_DIR=/tmp/browser-harness-runtime "
        "XDG_CONFIG_HOME=/tmp/browser-harness-config "
        f"browser-harness <{remote_program}'",
        timeout=300,
    )
    stdout = str(result.result or "")
    exit_code = int(result.exit_code)
    collector = (
        "import json\n"
        "info=page_info()\n"
        f"capture_screenshot('/tmp/outputs/{label}.png')\n"
        "print('PAGE_INFO_JSON='+json.dumps(info,ensure_ascii=False))\n"
    )
    remote_collector = f"/tmp/browser-harness-{label}-collector.py"
    sandbox.fs.upload_file(collector.encode(), remote_collector, timeout=30)
    collector_result = sandbox.process.exec(
        "bash -lc 'BU_CDP_URL=http://127.0.0.1:9222 BH_REQUIRE_REMOTE=1 BH_RECORD=1 "
        "BH_RUNTIME_DIR=/tmp/browser-harness-runtime "
        "XDG_CONFIG_HOME=/tmp/browser-harness-config "
        f"browser-harness <{remote_collector}'",
        timeout=300,
    )
    collector_stdout = str(collector_result.result or "")
    page = None
    for line in collector_stdout.splitlines():
        if line.startswith("PAGE_INFO_JSON="):
            page = json.loads(line.removeprefix("PAGE_INFO_JSON="))
    payload = json.dumps(
        {
            "exit_code": exit_code,
            "stdout": stdout,
            "error": stdout[-4000:] if exit_code else None,
            "evidence_exit_code": int(collector_result.exit_code),
            "evidence_stdout": collector_stdout,
            "page_info": page,
            "screenshot": f"/tmp/outputs/{label}.png",
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    (turns / f"{label}.result.json").write_text(payload + "\n")
    return payload


def run_interactive_episode(
    sandbox,
    out: Path,
    entry: str,
    runtime: dict,
    runtime_identity: dict,
    materialized: str,
    started: float,
    task_id: str,
) -> None:
    upload_interaction_skills(sandbox)
    initial_observation = execute_harness_turn(
        sandbox,
        out,
        "initial",
        f"new_tab({entry!r})\nwait_for_load()\n",
    )
    initial_value = json.loads(initial_observation)
    if initial_value["exit_code"] or initial_value["evidence_exit_code"]:
        raise RuntimeError(f"initial browser grounding failed: {initial_observation}")
    emit_protocol({
        "type": "ready",
        "entry_url": entry,
        "observation": initial_observation,
        "sandbox_id": str(sandbox.id),
    })
    final_response = ""
    termination_reason = "irrecoverable_protocol_eof"
    rollout_usage = None
    model_tool_calls = 0
    while line := sys.stdin.readline():
        request = json.loads(line)
        request_type = request.get("type")
        if request_type == "tool":
            turn = int(request["turn"])
            call = int(request["call"])
            result_text = execute_harness_turn(
                sandbox,
                out,
                f"turn-{turn:04d}-call-{call:02d}",
                str(request["code"]),
            )
            model_tool_calls += 1
            emit_protocol({
                "type": "tool_result",
                "turn": turn,
                "call": call,
                "result": result_text,
            })
        elif request_type == "finish":
            final_response = str(request.get("final_response") or "")
            termination_reason = str(request.get("termination_reason") or "")
            rollout_usage = request.get("usage")
            break
        else:
            raise RuntimeError(f"unknown interactive request: {request_type!r}")
    (out / "agent-final.txt").write_text(final_response)
    ok(sandbox.process.exec(
        "bash -lc 'tar -C /tmp -czf /tmp/outputs/browser-harness-runtime.tgz "
        "browser-harness-runtime'",
        timeout=60,
    ), "trajectory archive")
    sandbox.fs.upload_file(
        (final_response.strip() + "\n").encode(),
        "/tmp/customer-world/run/final_output.txt",
        timeout=30,
    )
    ok(sandbox.process.exec(
        "/tmp/customer-world/runtime/world-server finalize-episode", timeout=180
    ), "finalize")
    report = json.loads(ok(sandbox.process.exec(
        "cat /tmp/customer-world/evidence/verification-report.json", timeout=30
    ), "verifier report"))
    output_names = ok(sandbox.process.exec(
        "find /tmp/outputs -type f -printf '%P\\n' | sort", timeout=30
    ), "output enumeration").splitlines()
    outputs = []
    for name in output_names:
        if not name or name.startswith("/") or ".." in Path(name).parts:
            raise RuntimeError(f"unsafe output path: {name!r}")
        local = out / name
        local.parent.mkdir(parents=True, exist_ok=True)
        sandbox.fs.download_file(f"/tmp/outputs/{name}", str(local))
        payload = local.read_bytes()
        outputs.append({"path": name, "size": len(payload), "sha256": digest(payload)})
    (out / "verification-report.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "result.json").write_text(json.dumps({
        "complete": True,
        "architecture": "one_daytona_sandbox_multi_turn",
        "browser_use_cloud": False,
        "sandbox_id": str(sandbox.id),
        "world": {
            "site": "zenith",
            "task_id": task_id,
            "episode_id": runtime["episode_id"],
            "entry_url": entry,
        },
        "runtime_artifact": runtime_identity,
        "materialization": materialized,
        "interactive": True,
        "final_response": final_response,
        "termination_reason": termination_reason,
        "rollout_usage": rollout_usage,
        "model_tool_calls": model_tool_calls,
        "outputs": outputs,
        "agent_final_present": bool(final_response.strip()),
        "deterministic_reward": float(report.get("score", int(bool(report.get("passed"))))),
        "deterministic_passed": bool(report.get("passed")),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-binary", type=Path)
    parser.add_argument("--world-sha256", default=PINNED_WORLD_BINARY_SHA256)
    parser.add_argument("--customer-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--program", type=Path)
    parser.add_argument("--rollout-id", default="smoke")
    parser.add_argument("--task-id", choices=TASK_IDS, required=True)
    parser.add_argument("--seed", default="411001")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    if os.environ.get("BROWSER_USE_API_KEY") or os.environ.get("BROWSER_USE_CLOUD_API_KEY"):
        raise RuntimeError("Browser Use Cloud credentials must be absent")
    key = os.environ.get("DAYTONA_KEY", "").strip()
    if not key:
        raise RuntimeError("DAYTONA_KEY is required")

    customer = args.customer_repo.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = (args.world_binary or customer / "snapshots/worlds/zenith/world-server").resolve()
    binary_bytes = binary.read_bytes()
    runtime_identity = runtime_provenance(
        binary, customer, args.task_id, args.world_sha256
    )
    (out / "runtime-provenance.json").write_text(
        json.dumps(runtime_identity, indent=2, sort_keys=True) + "\n"
    )
    source_identity = str(runtime_identity["source"]["identity"])
    task_package_digest = str(runtime_identity["task_package"]["digest"])
    daytona = Daytona(DaytonaConfig(api_key=key, otel_enabled=False))
    sandbox = None
    started = time.monotonic()
    cleanup = {
        "sandbox_deleted": False,
        "sandbox_absence_verified": False,
        "snapshot_retained": True,
    }

    try:
        try:
            snapshot = daytona.snapshot.get(SNAPSHOT)
        except DaytonaNotFoundError as exc:
            raise RuntimeError(f"retained shared snapshot is missing: {SNAPSHOT}") from exc
        created = False
        state = str(getattr(snapshot.state, "value", snapshot.state)).lower()
        if state != "active":
            raise RuntimeError(f"snapshot not active: {state}")
        (out / "snapshot.json").write_text(json.dumps({
            "id": str(snapshot.id), "name": SNAPSHOT, "created": created, "state": state
        }, indent=2) + "\n")

        sandbox = daytona.create(
            CreateSandboxFromSnapshotParams(
                snapshot=SNAPSHOT,
                language="python",
                name=f"zenith-inkling-{args.rollout_id}",
                labels={LABEL: f"zenith-{args.rollout_id}"},
                public=False,
                ephemeral=True,
                auto_stop_interval=30,
                network_block_all=False,
                env_vars={
                    "BROWSER_USE_WORLDS_SITE": "zenith",
                    "BROWSER_USE_WORLDS_TASK_ID": args.task_id,
                    "BROWSER_USE_WORLDS_SEED": args.seed,
                    "BROWSER_USE_WORLDS_DIFFICULTY_CONFIGURATION": "standard",
                    "BROWSER_USE_WORLDS_SOURCE_COMMIT": source_identity,
                    "BROWSER_USE_WORLDS_HARBOR_COMMIT": "17bc7141ccb681e354e700fdf1dd90ee7c9856e3",
                    "BROWSER_USE_WORLDS_HARBOR_VERSION": "0.20.0",
                    "BROWSER_USE_WORLDS_INTERNAL_ORIGIN": "http://127.0.0.1:3000",
                    "BROWSER_USE_WORLDS_CONTROL_DIR": "/tmp/customer-world/run",
                    "BROWSER_USE_WORLDS_EVIDENCE_DIR": "/tmp/customer-world/evidence",
                    "DATA_DIR": "/tmp/customer-world/data",
                    "PORT": "3000",
                },
            ),
            timeout=300,
        )
        ok(sandbox.process.exec(
            "mkdir -p /tmp/customer-world/runtime /tmp/customer-world/run "
            "/tmp/customer-world/data /tmp/customer-world/evidence /tmp/outputs",
            timeout=30,
        ), "prepare")
        sandbox.fs.upload_file(binary_bytes, "/tmp/customer-world/runtime/world-server", timeout=120)
        materialized = ok(sandbox.process.exec(
            "python3 - <<'PY'\n"
            "import hashlib,pathlib\n"
            "b=pathlib.Path('/tmp/customer-world/runtime/world-server').read_bytes()\n"
            f"assert len(b)=={len(binary_bytes)}\n"
            f"assert hashlib.sha256(b).hexdigest()=='{digest(binary_bytes)}'\n"
            "print(len(b),hashlib.sha256(b).hexdigest())\n"
            "PY",
            timeout=60,
        ), "materialization").strip()

        common_env = {
            "CONTROL_TOKEN": "browser-rl-local-harness-control",
            "BROWSER_USE_WORLDS_SITE": "zenith",
            "BROWSER_USE_WORLDS_TASK_ID": args.task_id,
            "BROWSER_USE_WORLDS_SEED": args.seed,
            "BROWSER_USE_WORLDS_DIFFICULTY_CONFIGURATION": "standard",
            "BROWSER_USE_WORLDS_SOURCE_COMMIT": source_identity,
            "BROWSER_USE_WORLDS_IMAGE_DIGEST": f"sha256:{digest(binary_bytes)}",
            "BROWSER_USE_WORLDS_TASK_PACKAGE_VERSION": TASK_PACKAGE_VERSION,
            "BROWSER_USE_WORLDS_TASK_PACKAGE_DIGEST": task_package_digest,
            "BROWSER_USE_WORLDS_HARBOR_COMMIT": "17bc7141ccb681e354e700fdf1dd90ee7c9856e3",
            "BROWSER_USE_WORLDS_HARBOR_VERSION": "0.20.0",
            "HARBOR_TRIAL_ID": f"zenith-inkling-{args.rollout_id}",
            "BROWSER_USE_WORLDS_CONTROL_DIR": "/tmp/customer-world/run",
            "BROWSER_USE_WORLDS_EVIDENCE_DIR": "/tmp/customer-world/evidence",
            "DATA_DIR": "/tmp/customer-world/data",
            "PORT": "3000",
        }
        ok(sandbox.process.exec(
            "bash -lc 'set -e; chmod 755 /tmp/customer-world/runtime/world-server; "
            "umask 077; printf %s \"$CONTROL_TOKEN\" >/tmp/customer-world/run/control-token; "
            ": >/tmp/customer-world/world.log; nohup /tmp/customer-world/runtime/world-server "
            ">>/tmp/customer-world/world.log 2>&1 </dev/null & echo $! >/tmp/customer-world/world.pid'",
            env=common_env, timeout=30,
        ), "world bootstrap launch")
        ok(sandbox.process.exec(
            "bash -lc 'for _ in $(seq 1 120); do curl -fsS "
            "http://127.0.0.1:3000/__browser_test_environment__/health >/dev/null 2>&1 && exit 0; "
            "sleep 1; done; cat /tmp/customer-world/world.log; exit 1'",
            timeout=150,
        ), "world bootstrap health")
        ok(sandbox.process.exec(
            "bash -lc 'set -e; /tmp/customer-world/runtime/world-server initialize-world; "
            "kill $(cat /tmp/customer-world/world.pid); sleep 1; "
            "nohup /tmp/customer-world/runtime/world-server >>/tmp/customer-world/world.log 2>&1 </dev/null & "
            "echo $! >/tmp/customer-world/world.pid'",
            env=common_env, timeout=60,
        ), "world initialize")
        ok(sandbox.process.exec(
            "bash -lc 'for _ in $(seq 1 120); do curl -fsS "
            "http://127.0.0.1:3000/__browser_test_environment__/health >/dev/null 2>&1 && exit 0; "
            "sleep 1; done; cat /tmp/customer-world/world.log; exit 1'",
            timeout=150,
        ), "world final health")
        runtime = json.loads(ok(
            sandbox.process.exec("cat /tmp/customer-world/run/runtime.json", timeout=30),
            "runtime receipt",
        ))
        if runtime.get("site") != "zenith" or runtime.get("task_id") != args.task_id:
            raise RuntimeError(f"runtime identity mismatch: {runtime}")
        expected_runtime_identity = {
            "source_commit": source_identity,
            "image_digest": f"sha256:{digest(binary_bytes)}",
            "task_package_version": TASK_PACKAGE_VERSION,
            "task_package_digest": task_package_digest,
        }
        mismatches = {
            key: {"expected": expected, "actual": runtime.get(key)}
            for key, expected in expected_runtime_identity.items()
            if runtime.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(f"runtime provenance mismatch: {mismatches}")
        (out / "runtime-receipt.json").write_text(
            json.dumps(runtime, indent=2, sort_keys=True) + "\n"
        )
        entry = f"http://127.0.0.1:3000/e/{runtime['episode_id']}/"

        ok(sandbox.process.exec(
            "bash -lc 'rm -rf /tmp/chrome-profile; mkdir -p /tmp/chrome-profile; "
            "nohup chromium --headless=new --no-sandbox --disable-dev-shm-usage --disable-gpu "
            "--remote-debugging-address=127.0.0.1 --remote-debugging-port=9222 "
            "--user-data-dir=/tmp/chrome-profile about:blank >/tmp/chromium.log 2>&1 </dev/null & "
            "echo $! >/tmp/chromium.pid; "
            "for _ in $(seq 1 60); do curl -fsS http://127.0.0.1:9222/json/version >/tmp/cdp.json && exit 0; sleep 1; done; exit 1'",
            timeout=90,
        ), "chromium launch")

        if args.interactive:
            run_interactive_episode(
                sandbox,
                out,
                entry,
                runtime,
                runtime_identity,
                materialized,
                started,
                args.task_id,
            )
            return

        reference_program = """import json
new_tab(ENTRY)
wait_for_load()
cards = js(\"[...document.querySelectorAll('.zm-results-grid .zm-tile-link')].slice(0,3).map(a=>({href:a.href}))\")
js("(()=>{const f=document.querySelector('.zm-delivery-form');const gb=f?.querySelector('input[name=market][value=GB]');if(!f||!gb)throw new Error('Zenith market controls missing');gb.checked=true;f.requestSubmit();return true})()")
wait_for_load()
rows=[]
for card in cards:
    goto_url(card['href'])
    wait_for_load()
    rows.append(js(\"(()=>{const p=document.querySelector('.zm-detail-price .zm-price-split .zm-offscreen')?.textContent||'';return {title:document.querySelector('h1')?.innerText?.trim()||'',price:Number(p.replace(/[^0-9.]/g,'')),seller:document.querySelector('.zm-merchant-link')?.innerText?.trim()||''}})()\"))
answer=json.dumps(rows,ensure_ascii=False)
open('/tmp/outputs/final-answer.json','w').write(answer+'\\n')
open('/tmp/outputs/page-info.txt','w').write(json.dumps(page_info(),ensure_ascii=False))
capture_screenshot('/tmp/outputs/final.png')
print(answer)
""".replace("ENTRY", repr(entry))
        if args.program:
            generated = args.program.read_text()
            program = (
                f"import json, os\nENTRY = {entry!r}\n" + generated + "\n"
                "import os\n"
                "if not os.path.exists('/tmp/outputs/final.png'):\n"
                "    capture_screenshot('/tmp/outputs/final.png')\n"
                "if not os.path.exists('/tmp/outputs/page-info.txt'):\n"
                "    open('/tmp/outputs/page-info.txt','w').write(json.dumps(page_info(),ensure_ascii=False))\n"
            )
        else:
            program = reference_program
        (out / "agent-program.py").write_text(program)
        sandbox.fs.upload_file(program.encode(), "/tmp/browser-harness-task.py", timeout=30)
        harness_result = sandbox.process.exec(
            "bash -lc 'BU_CDP_URL=http://127.0.0.1:9222 BH_REQUIRE_REMOTE=1 BH_RECORD=1 "
            "BH_RUNTIME_DIR=/tmp/browser-harness-runtime XDG_CONFIG_HOME=/tmp/browser-harness-config "
            "browser-harness </tmp/browser-harness-task.py'",
            timeout=300,
        )
        harness_stdout = str(harness_result.result or "")
        harness_exit_code = int(harness_result.exit_code)
        (out / "browser-harness.stdout.txt").write_text(harness_stdout)
        if harness_exit_code:
            collector = (
                "import json\n"
                "capture_screenshot('/tmp/outputs/final.png')\n"
                "open('/tmp/outputs/page-info.txt','w').write(json.dumps(page_info(),ensure_ascii=False))\n"
            )
            sandbox.fs.upload_file(collector.encode(), "/tmp/browser-harness-collector.py", timeout=30)
            collector_stdout = ok(sandbox.process.exec(
                "bash -lc 'BU_CDP_URL=http://127.0.0.1:9222 BH_REQUIRE_REMOTE=1 BH_RECORD=1 "
                "BH_RUNTIME_DIR=/tmp/browser-harness-collector-runtime "
                "XDG_CONFIG_HOME=/tmp/browser-harness-collector-config "
                "browser-harness </tmp/browser-harness-collector.py'",
                timeout=300,
            ), "browser harness evidence collector")
            (out / "browser-harness-collector.stdout.txt").write_text(collector_stdout)
        ok(sandbox.process.exec(
            "bash -lc 'tar -C /tmp -czf /tmp/outputs/browser-harness-runtime.tgz browser-harness-runtime'",
            timeout=60,
        ), "trajectory archive")
        answer_result = sandbox.process.exec(
            "test -f /tmp/outputs/final-answer.json && cat /tmp/outputs/final-answer.json",
            timeout=30,
        )
        agent_final_present = answer_result.exit_code == 0
        answer = str(answer_result.result or "").strip() if agent_final_present else ""
        sandbox.fs.upload_file(
            (answer + "\n").encode(), "/tmp/customer-world/run/final_output.txt", timeout=30
        )
        ok(sandbox.process.exec(
            "/tmp/customer-world/runtime/world-server finalize-episode", timeout=180
        ), "finalize")
        report = json.loads(ok(sandbox.process.exec(
            "cat /tmp/customer-world/evidence/verification-report.json", timeout=30
        ), "verifier report"))

        output_names = ok(sandbox.process.exec(
            "find /tmp/outputs -type f -printf '%P\\n' | sort", timeout=30
        ), "output enumeration").splitlines()
        outputs = []
        for name in output_names:
            if not name or name.startswith("/") or ".." in Path(name).parts:
                raise RuntimeError(f"unsafe output path: {name!r}")
            local = out / name
            local.parent.mkdir(parents=True, exist_ok=True)
            sandbox.fs.download_file(f"/tmp/outputs/{name}", str(local))
            payload = local.read_bytes()
            outputs.append({"path": name, "size": len(payload), "sha256": digest(payload)})
        (out / "verification-report.json").write_text(json.dumps(report, indent=2) + "\n")
        (out / "result.json").write_text(json.dumps({
            "complete": True,
            "architecture": "one_daytona_sandbox",
            "browser_use_cloud": False,
            "sandbox_id": str(sandbox.id),
            "world": {"site": "zenith", "task_id": args.task_id, "episode_id": runtime["episode_id"], "entry_url": entry},
            "runtime_artifact": runtime_identity,
            "materialization": materialized,
            "browser_harness_stdout": harness_stdout[-8000:],
            "agent_program_exit_code": harness_exit_code,
            "agent_program_error": harness_stdout[-4000:] if harness_exit_code else None,
            "outputs": outputs,
            "agent_final_present": agent_final_present,
            "deterministic_reward": float(report.get("score", int(bool(report.get("passed"))))),
            "deterministic_passed": bool(report.get("passed")),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }, indent=2) + "\n")
    finally:
        if sandbox is not None:
            for remote, name in (("/tmp/customer-world/world.log", "world.log"), ("/tmp/chromium.log", "chromium.log")):
                try:
                    sandbox.fs.download_file(remote, str(out / name))
                except Exception:
                    pass
            try:
                sandbox.process.exec(
                    "bash -lc 'test -f /tmp/chromium.pid && kill $(cat /tmp/chromium.pid) 2>/dev/null || true; "
                    "test -f /tmp/customer-world/world.pid && kill $(cat /tmp/customer-world/world.pid) 2>/dev/null || true'",
                    timeout=30,
                )
            except Exception:
                pass
            try:
                daytona.delete(sandbox, 120)
                cleanup["sandbox_deleted"] = True
            except DaytonaNotFoundError:
                cleanup["sandbox_deleted"] = True
            try:
                daytona.get(str(sandbox.id))
            except DaytonaNotFoundError:
                cleanup["sandbox_absence_verified"] = True
        (out / "cleanup.json").write_text(json.dumps(cleanup, indent=2) + "\n")


if __name__ == "__main__":
    main()
