"""Verify Zenith World, Chromium, and Browser Harness in one Daytona sandbox."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from daytona import (
    CreateSandboxFromSnapshotParams,
    Daytona,
    DaytonaConfig,
    DaytonaNotFoundError,
)

SNAPSHOT = "browser-rl-local-harness-f5eaf904-c2m4d4-v1"
LABEL = "browser-use.browser-rl-local-harness"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ok(result, stage: str) -> str:
    text = str(result.result or "")
    if result.exit_code:
        raise RuntimeError(f"{stage} failed ({result.exit_code}): {text[-4000:]}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-binary", type=Path)
    parser.add_argument("--customer-repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--program", type=Path)
    parser.add_argument("--rollout-id", default="smoke")
    parser.add_argument("--seed", default="411001")
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
    runtime_identity = {"size": len(binary_bytes), "sha256": digest(binary_bytes)}
    daytona = Daytona(DaytonaConfig(api_key=key, otel_enabled=False))
    sandbox = None
    started = time.monotonic()
    cleanup = {"sandbox_deleted": False, "snapshot_retained": True}

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
                labels={LABEL: f"zenith-zslr01-{args.rollout_id}"},
                public=False,
                ephemeral=True,
                auto_stop_interval=30,
                network_block_all=False,
                env_vars={
                    "BROWSER_USE_WORLDS_SITE": "zenith",
                    "BROWSER_USE_WORLDS_TASK_ID": "zslr01",
                    "BROWSER_USE_WORLDS_SEED": args.seed,
                    "BROWSER_USE_WORLDS_DIFFICULTY_CONFIGURATION": "standard",
                    "BROWSER_USE_WORLDS_SOURCE_COMMIT": "217f853c0081b50c93651670df21bf78ea975875",
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
            "BROWSER_USE_WORLDS_TASK_ID": "zslr01",
            "BROWSER_USE_WORLDS_SEED": args.seed,
            "BROWSER_USE_WORLDS_DIFFICULTY_CONFIGURATION": "standard",
            "BROWSER_USE_WORLDS_SOURCE_COMMIT": "8dc91725c0e5294a21384e9556f0b85ea25feb83",
            "BROWSER_USE_WORLDS_IMAGE_DIGEST": f"sha256:{digest(binary_bytes)}",
            "BROWSER_USE_WORLDS_TASK_PACKAGE_VERSION": "1.0.0",
            "BROWSER_USE_WORLDS_TASK_PACKAGE_DIGEST": "sha256:2a1576ddbb4f6e49d2536d58c6b60c0b02d5710e5f33f54655f96e7414fc90ea",
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
        if runtime.get("site") != "zenith" or runtime.get("task_id") != "zslr01":
            raise RuntimeError(f"runtime identity mismatch: {runtime}")
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
            "world": {"site": "zenith", "task_id": "zslr01", "episode_id": runtime["episode_id"], "entry_url": entry},
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
        (out / "cleanup.json").write_text(json.dumps(cleanup, indent=2) + "\n")


if __name__ == "__main__":
    main()
