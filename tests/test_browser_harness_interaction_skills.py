from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import verify_zenith_daytona_sandbox as verifier


class Result:
    def __init__(self, result: str = "", exit_code: int = 0):
        self.result = result
        self.exit_code = exit_code


class FakeFS:
    def __init__(self, events: list[tuple[str, str]], files: dict[str, bytes]):
        self.events = events
        self.files = files

    def upload_file(self, payload: bytes, remote: str, timeout: int) -> None:
        self.events.append(("upload", remote))
        self.files[remote] = payload

    def download_file(self, remote: str, local: str) -> None:
        Path(local).write_bytes(self.files[remote])


class FakeProcess:
    def __init__(self, events: list[tuple[str, str]], files: dict[str, bytes]):
        self.events = events
        self.files = files
        self.read_files: list[str] = []

    def exec(self, command: str, **kwargs) -> Result:
        self.events.append(("exec", command))
        if command.startswith("rm -rf "):
            prefix = verifier.INTERACTION_SKILLS_REMOTE_DIR + "/"
            for path in [path for path in self.files if path.startswith(prefix)]:
                del self.files[path]
            return Result()
        if command.startswith("python3 - <<'PY'"):
            prefix = verifier.INTERACTION_SKILLS_REMOTE_DIR + "/"
            remote_files = {
                path.removeprefix(prefix): payload
                for path, payload in self.files.items()
                if path.startswith(prefix)
            }
            self.read_files = sorted(remote_files)
            value = {
                name: {
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for name, payload in sorted(remote_files.items())
            }
            return Result(json.dumps(value, sort_keys=True, separators=(",", ":")))
        if "browser-harness <" in command and "collector" in command:
            return Result('PAGE_INFO_JSON={"url":"http://127.0.0.1/task","title":"Zenith"}\n')
        if command == "cat /tmp/customer-world/evidence/verification-report.json":
            return Result('{"passed":true,"score":1}')
        if command.startswith("find /tmp/outputs"):
            return Result("")
        return Result()


class FakeSandbox:
    id = "fake-sandbox"

    def __init__(self):
        self.events: list[tuple[str, str]] = []
        self.files: dict[str, bytes] = {}
        self.fs = FakeFS(self.events, self.files)
        self.process = FakeProcess(self.events, self.files)


class InteractionSkillUploadTest(unittest.TestCase):
    def test_manifest_hashes_upload_and_interactive_readability(self) -> None:
        self.assertEqual(
            verifier.INTERACTION_SKILLS_SOURCE_COMMIT,
            "f5eaf904b221dde0118eba1496961c3dc20fda88",
        )
        skill_text = (
            Path(__file__).resolve().parents[1] / "skills/browser-harness/SKILL.md"
        ).read_text()
        interaction_section = skill_text.split("## Interaction Skills\n", 1)[1].split(
            "\n## ", 1
        )[0]
        listed_names = {
            line.removeprefix("- ").strip()
            for line in interaction_section.splitlines()
            if line.startswith("- ")
        }
        self.assertEqual(listed_names, set(verifier.INTERACTION_SKILL_SHA256))
        self.assertTrue(
            {"connection.md", "profile-sync.md", "make-video.md"}.isdisjoint(listed_names)
        )

        sandbox = FakeSandbox()
        old_stdin = sys.stdin
        try:
            sys.stdin = io.StringIO(
                json.dumps({
                    "type": "finish",
                    "final_response": "done",
                    "termination_reason": "final_answer",
                    "usage": {},
                }) + "\n"
            )
            with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
                verifier.run_interactive_episode(
                    sandbox,
                    Path(temporary),
                    "http://127.0.0.1/task",
                    {"episode_id": "episode-1"},
                    {"size": 1, "sha256": "x"},
                    "materialized",
                    0.0,
                )
        finally:
            sys.stdin = old_stdin

        expected_remote = {
            f"{verifier.INTERACTION_SKILLS_REMOTE_DIR}/{name}"
            for name in verifier.INTERACTION_SKILL_SHA256
        }
        uploaded_remote = {
            value for event, value in sandbox.events
            if event == "upload" and value.startswith(
                verifier.INTERACTION_SKILLS_REMOTE_DIR + "/"
            )
        }
        self.assertEqual(uploaded_remote, expected_remote)
        self.assertEqual(
            set(sandbox.process.read_files), set(verifier.INTERACTION_SKILL_SHA256)
        )
        initial_harness_index = next(
            index for index, (event, value) in enumerate(sandbox.events)
            if event == "exec" and "browser-harness <" in value
        )
        for remote in expected_remote:
            self.assertLess(sandbox.events.index(("upload", remote)), initial_harness_index)


if __name__ == "__main__":
    unittest.main()
