# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — engine crate public API compatibility gate tests

"""Hold the engine SemVer gate: the baseline rule, the verdicts, and the CI job that runs it."""

from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from tools.engine_semver_check import (
    ENGINE_MANIFEST,
    UNCHECKED_FEATURE_NOTE,
    check_command,
    crate_identity,
    main,
    published_versions,
    sparse_index_path,
)

ROOT = Path(__file__).resolve().parents[2]
NAME, VERSION = crate_identity(ROOT / ENGINE_MANIFEST)


def _index(*records: tuple[str, bool]) -> str:
    return "\n".join(
        json.dumps({"name": NAME, "vers": vers, "yanked": yanked}) for vers, yanked in records
    )


def _recording_run(code: int) -> tuple[list[list[str]], Any]:
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["cwd"] == ROOT and kwargs["check"] is False
        calls.append(command)
        return subprocess.CompletedProcess(command, code)

    return calls, run


def test_the_engine_manifest_names_the_published_crate() -> None:
    """The gate reads the engine's own package name and version, not a copy of them."""
    package = tomllib.loads((ROOT / "engine/Cargo.toml").read_text(encoding="utf-8"))["package"]
    assert (package["name"], package["version"]) == (NAME, VERSION)
    assert NAME == "sc_neurocore_engine"


@pytest.mark.parametrize(
    ("name", "path"),
    [
        ("a", "1/a"),
        ("ab", "2/ab"),
        ("abc", "3/a/abc"),
        ("Serde", "se/rd/serde"),
        ("sc_neurocore_engine", "sc/_n/sc_neurocore_engine"),
    ],
)
def test_the_sparse_index_path_follows_cargos_layout(name: str, path: str) -> None:
    """Names of one, two, three and more characters map to Cargo's index directories, lower case."""
    assert sparse_index_path(name) == path


def test_yanked_versions_are_not_baselines_and_a_misread_index_refuses() -> None:
    """Only published, unyanked versions can be a baseline; a line that is no record stops the gate."""
    assert published_versions(_index(("4.1.0", False), ("4.1.1", True), ("4.2.0", False))) == {
        "4.1.0",
        "4.2.0",
    }
    with pytest.raises(ValueError, match="line 2"):
        published_versions(_index(("4.2.0", False)) + "\n[1, 2]")


def test_between_releases_the_published_version_is_the_baseline_and_breaks_are_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With the manifest's version published, that release is compared and the change is minor."""
    calls, run = _recording_run(0)
    assert main(["--repo", str(ROOT)], run=run, fetch=lambda name: _index((VERSION, False))) == 0
    assert calls == [
        [
            "cargo",
            "semver-checks",
            "check-release",
            "--manifest-path",
            "engine/Cargo.toml",
            "--only-explicit-features",
            "--features",
            "gpu",
            "--baseline-version",
            VERSION,
            "--release-type",
            "minor",
        ]
    ]
    out = capsys.readouterr().out
    assert UNCHECKED_FEATURE_NOTE in out and "No incompatible public change" in out


def test_at_the_release_bump_the_tool_picks_the_baseline_and_the_release_type() -> None:
    """With the manifest's version unpublished, no baseline or release type is forced."""
    command = check_command(ENGINE_MANIFEST, "5.0.0", {"4.1.0", "4.2.0"})
    assert "--baseline-version" not in command and "--release-type" not in command
    assert command[:3] == ["cargo", "semver-checks", "check-release"]
    calls, run = _recording_run(0)
    assert main(["--repo", str(ROOT)], run=run, fetch=lambda name: _index(("0.0.1", False))) == 0
    assert "--baseline-version" not in calls[0]


@pytest.mark.parametrize(
    ("code", "message"),
    [(100, "incompatible public change"), (101, "did not complete (exit 101)"), (2, "exit 2")],
)
def test_a_violation_or_a_check_that_did_not_complete_fails(
    code: int, message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exit 100 is a refused change; any other non-zero exit is a check that did not run."""
    _, run = _recording_run(code)
    assert main(["--repo", str(ROOT)], run=run, fetch=lambda name: _index((VERSION, False))) == 1
    assert message in capsys.readouterr().err


def test_no_baseline_or_no_tool_fails_rather_than_passing_unchecked(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unreachable index or a missing cargo-semver-checks is a failure, never a pass."""

    def offline(name: str) -> str:
        raise OSError("index unreachable")

    calls, run = _recording_run(0)
    assert main(["--repo", str(ROOT)], run=run, fetch=offline) == 1 and calls == []
    assert "baseline could not be determined" in capsys.readouterr().err

    def missing(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("cargo")

    assert (
        main(["--repo", str(ROOT)], run=missing, fetch=lambda name: _index((VERSION, False))) == 1
    )
    assert "did not run" in capsys.readouterr().err


def test_the_ci_job_installs_the_pinned_tool_and_runs_the_gate() -> None:
    """CI installs cargo-semver-checks at a fixed version with its lock file, then runs this gate."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    runs = [step.get("run", "").strip() for step in workflow["jobs"]["engine-semver"]["steps"]]
    install = "cargo install cargo-semver-checks --version 0.51.0 --locked"
    assert install in runs and "python tools/engine_semver_check.py" in runs
    assert runs.index(install) < runs.index("python tools/engine_semver_check.py")
