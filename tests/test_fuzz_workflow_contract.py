# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — fuzz workflow target list, crash collection and reproducer upload

"""The fuzz workflow runs every cargo-fuzz target and keeps the crash reproducers.

The fuzz step loops over the targets of ``engine/fuzz/Cargo.toml``. A crash in one target must
not stop the later targets: the step runs them all, names every crashing target and fails at the
end. On failure the reproducers under ``engine/fuzz/artifacts/`` are uploaded with the same pinned
``upload-artifact`` revision as the other workflows. The step script itself is run here, under the
``bash -e`` that GitHub Actions uses for a ``run`` step without a ``shell``, with a stand-in
``cargo`` on ``PATH``.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tomllib
from typing import Any, cast

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = _ROOT / ".github" / "workflows" / "fuzz.yml"
_FUZZ_MANIFEST = _ROOT / "engine" / "fuzz" / "Cargo.toml"
_UPLOAD_PIN = "actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9"
_FUZZ_STEP = "Fuzz the IR parser, printer round-trip, bitstream and sparse matrix"

_FAKE_CARGO = """#!/usr/bin/env bash
# Record the target of "cargo +toolchain fuzz run <target> ..." and fail for the chosen ones.
echo "$4" >> "$FAKE_CARGO_LOG"
for crashing in $FAKE_CARGO_CRASH; do
  if [[ "$4" == "$crashing" ]]; then
    exit 77
  fi
done
exit 0
"""


def _steps() -> list[dict[str, Any]]:
    """Return the steps of the workflow's single job."""
    workflow = cast(dict[str, Any], yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8")))
    jobs = cast(dict[str, dict[str, Any]], workflow["jobs"])
    assert len(jobs) == 1
    (job,) = jobs.values()
    return cast(list[dict[str, Any]], job["steps"])


def _step(name: str) -> dict[str, Any]:
    """Return the workflow step with the given name."""
    matches = [step for step in _steps() if step.get("name") == name]
    assert len(matches) == 1, name
    return matches[0]


def _manifest_targets() -> list[str]:
    """Return the cargo-fuzz binary names declared in the fuzz manifest, in order."""
    manifest = tomllib.loads(_FUZZ_MANIFEST.read_text(encoding="utf-8"))
    return [cast(str, entry["name"]) for entry in manifest["bin"]]


def _loop_targets(script: str) -> list[str]:
    """Return the targets of the fuzz step's ``for target in ...; do`` loop."""
    found = re.search(r"for target in ([^;]+); do", script)
    assert found is not None
    return found.group(1).split()


def _run_fuzz_step(tmp_path: Path, crashing: list[str]) -> tuple[int, list[str], str]:
    """Run the fuzz step script with a stand-in cargo; return its rc, the targets run, its output."""
    step = _step(_FUZZ_STEP)
    assert "shell" not in step
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cargo = bin_dir / "cargo"
    cargo.write_text(_FAKE_CARGO, encoding="utf-8")
    cargo.chmod(0o755)
    script = tmp_path / "step.sh"
    script.write_text(cast(str, step["run"]), encoding="utf-8")
    log = tmp_path / "cargo.log"
    bash = shutil.which("bash")
    assert bash is not None
    env = {
        **os.environ,
        "PATH": os.pathsep.join([str(bin_dir), os.environ.get("PATH", "")]),
        "MAX_TOTAL_TIME": "1",
        "FAKE_CARGO_LOG": str(log),
        "FAKE_CARGO_CRASH": " ".join(crashing),
    }
    done = subprocess.run(
        [bash, "-e", str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    ran = log.read_text(encoding="utf-8").split() if log.exists() else []
    return done.returncode, ran, done.stdout + done.stderr


def test_loop_runs_exactly_the_manifest_targets() -> None:
    """The workflow loop lists the fuzz manifest's binaries, no more and no fewer."""
    assert _loop_targets(cast(str, _step(_FUZZ_STEP)["run"])) == _manifest_targets()


def test_every_target_passing_makes_the_step_pass(tmp_path: Path) -> None:
    """With no crash the step runs every target once and exits 0."""
    rc, ran, _ = _run_fuzz_step(tmp_path, crashing=[])
    assert rc == 0
    assert ran == _manifest_targets()


@pytest.mark.parametrize("crashing", [["parse_ir"], ["bitstream"], ["roundtrip_ir", "csr_matrix"]])
def test_a_crash_does_not_stop_the_later_targets(tmp_path: Path, crashing: list[str]) -> None:
    """A crashing target leaves the later targets running; the step fails and names each crash."""
    rc, ran, output = _run_fuzz_step(tmp_path, crashing=crashing)
    assert rc != 0
    assert ran == _manifest_targets()
    for target in crashing:
        assert f"::error::fuzz target {target} found a crash" in output
    assert f"::error::crashing fuzz targets: {' '.join(crashing)}" in output


def test_a_non_numeric_time_budget_is_refused_before_fuzzing(tmp_path: Path) -> None:
    """The time-budget guard still stops the step before any target runs."""
    step = _step(_FUZZ_STEP)
    script = tmp_path / "step.sh"
    script.write_text(cast(str, step["run"]), encoding="utf-8")
    bash = shutil.which("bash")
    assert bash is not None
    done = subprocess.run(
        [bash, "-e", str(script)],
        cwd=tmp_path,
        env={**os.environ, "MAX_TOTAL_TIME": "300; rm -rf x"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode != 0
    assert "::group::fuzz" not in done.stdout


def test_reproducers_are_uploaded_on_failure_with_the_pinned_action() -> None:
    """On failure the crash artifacts are uploaded by the repository's pinned upload-artifact."""
    names = [step.get("name") for step in _steps()]
    upload = _step("Upload crash reproducers")
    assert names.index("Upload crash reproducers") > names.index(_FUZZ_STEP)
    assert upload["if"] == "failure()"
    assert upload["uses"] == _UPLOAD_PIN
    assert upload["with"]["path"] == "engine/fuzz/artifacts/"
    assert upload["with"]["if-no-files-found"] == "warn"
    assert upload["with"]["name"] == "fuzz-crash-reproducers"
