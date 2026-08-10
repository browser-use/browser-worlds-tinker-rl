from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import verify_zenith_daytona_sandbox as verifier


class ZenithRuntimeProvenanceTest(unittest.TestCase):
    def test_source_identity_hashes_tracked_and_untracked_scoped_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(repo), "config", "user.name", "Test"], check=True
            )
            scope = repo / "world"
            scope.mkdir()
            (scope / "tracked.go").write_text("package world\n")
            subprocess.run(["git", "-C", str(repo), "add", "world/tracked.go"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "base"], check=True)
            head = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

            clean = verifier.source_provenance(repo, "world")
            self.assertEqual(clean["head"], head)
            self.assertFalse(clean["dirty"])
            self.assertEqual(clean["identity"], head)

            (scope / "tracked.go").write_text("package world\n// dirty\n")
            (scope / "untracked.go").write_text("package world\n// untracked\n")
            first = verifier.source_provenance(repo, "world")
            self.assertTrue(first["dirty"])
            self.assertEqual(first["scope"], "world")
            self.assertTrue(first["identity"].startswith(head + "+dirty.sha256."))
            self.assertEqual(len(first["scoped_diff_sha256"]), 64)
            self.assertEqual(first, verifier.source_provenance(repo, "world"))

            (repo / "outside.txt").write_text("ignored")
            self.assertEqual(first, verifier.source_provenance(repo, "world"))

    def test_runtime_provenance_pins_binary_and_uses_packager_digest(self) -> None:
        expected_binary = bytes.fromhex("00")
        expected_hash = hashlib.sha256(expected_binary).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "world-server"
            binary.write_bytes(expected_binary)
            task = root / "harbor/tasks/zenith-zslr01"
            task.mkdir(parents=True)
            (task / "task.toml").write_text(
                '[task]\nname="browser-use-worlds/zenith-zslr01"\nversion="1.0.0"\n'
            )
            source = {
                "head": "a" * 40,
                "scope": verifier.SOURCE_SCOPE,
                "dirty": True,
                "scoped_diff_sha256": "b" * 64,
                "identity": "a" * 40 + "+dirty.sha256." + "b" * 64,
            }
            with (
                patch.object(verifier, "PINNED_WORLD_BINARY_SHA256", expected_hash),
                patch.object(verifier, "source_provenance", return_value=source),
                patch.object(
                    verifier,
                    "canonical_harbor_package_digest",
                    return_value="c" * 64,
                ) as package_digest,
            ):
                observed = verifier.runtime_provenance(binary, root)

            package_digest.assert_called_once_with(root)
            self.assertEqual(observed["binary"]["sha256"], expected_hash)
            self.assertEqual(observed["source"], source)
            self.assertEqual(observed["task_package"]["version"], "1.0.0")
            self.assertEqual(observed["task_package"]["digest"], "sha256:" + "c" * 64)

    def test_supplied_binary_must_match_exact_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "world-server"
            binary.write_bytes(b"wrong")
            with self.assertRaisesRegex(RuntimeError, "binary SHA-256 mismatch"):
                verifier.runtime_provenance(binary, root)

    def test_stale_runtime_hashes_are_absent(self) -> None:
        source = Path(verifier.__file__).read_text()
        self.assertNotIn("a133fb8f1d79e03cc1330920dcfb1674fe9f45d2", source)
        self.assertNotIn(
            "2a1576ddbb4f6e49d2536d58c6b60c0b02d5710e5f33f54655f96e7414fc90ea",
            source,
        )
        self.assertEqual(
            verifier.PINNED_WORLD_BINARY_SHA256,
            "41bae34a7b46e7415ef911a767bd637b0c73daed379a4ec5f57d20ffad9c786f",
        )


if __name__ == "__main__":
    unittest.main()
