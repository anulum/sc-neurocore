# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust crate formatting gate tests

"""Hold the crate formatting gate: every crate is checked, and one difference fails the gate."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from tools.crate_format_check import crate_manifests, main, unformatted_crates
from tools.preflight import GATE_GUARDS, GATES, gate_applies

ROOT = Path(__file__).resolve().parents[2]


def _recording_run(failing: set[str]) -> tuple[list[list[str]], Any]:
    calls: list[list[str]] = []

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert kwargs["cwd"] == ROOT and kwargs["check"] is False
        calls.append(command)
        code = 1 if command[-1] in failing else 0
        return subprocess.CompletedProcess(command, code, stdout="", stderr="")

    return calls, run


def test_every_crate_of_the_repository_is_found() -> None:
    """The gate finds each crate directory's manifest, so a new crate is checked without an edit."""
    expected = sorted(
        (path / "Cargo.toml").relative_to(ROOT)
        for path in (ROOT / "crates").iterdir()
        if (path / "Cargo.toml").is_file()
    )
    assert crate_manifests(ROOT) == expected
    assert len(expected) >= 7


def test_each_crate_is_checked_once_and_a_formatted_tree_passes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every crate gets its own cargo fmt --check, and all formatted gives exit 0."""
    calls, run = _recording_run(set())
    assert main(["--repo", str(ROOT)], run=run) == 0
    assert calls == [
        ["cargo", "fmt", "--check", "--manifest-path", manifest.as_posix()]
        for manifest in crate_manifests(ROOT)
    ]
    assert f"All {len(calls)} crates" in capsys.readouterr().out


def test_one_unformatted_crate_fails_the_gate_and_is_named(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A single crate that differs makes the gate fail and is named; the others are still checked."""
    target = "crates/neuro_symbolic/Cargo.toml"
    calls, run = _recording_run({target})
    checked, differing = unformatted_crates(ROOT, run)
    assert differing == [target] and checked == len(calls)
    assert main(["--repo", str(ROOT)], run=run) == 1
    assert target in capsys.readouterr().err


def test_a_tree_without_crates_fails_rather_than_passing_unchecked(tmp_path: Path) -> None:
    """With no crate to check the gate refuses instead of reporting success."""
    (tmp_path / "crates").mkdir()
    with pytest.raises(ValueError, match="nothing to check"):
        crate_manifests(tmp_path)
    calls, run = _recording_run(set())
    assert main(["--repo", str(tmp_path)], run=run) == 1 and calls == []


def test_the_preflight_and_the_ci_job_run_the_same_crate_gate() -> None:
    """The local preflight and the CI Rust job both run this gate, and the preflight runs it for crates."""
    gates = dict(GATES)
    assert gates["cargo-fmt-crates"] == ["python", "tools/crate_format_check.py"]
    assert "crates/*" in GATE_GUARDS["cargo-fmt-crates"]
    assert gate_applies("cargo-fmt-crates", ["crates/core_engine/src/bitstream.rs"])
    assert gate_applies("cargo-fmt-crates", ["tools/crate_format_check.py"])
    assert not gate_applies("cargo-fmt-crates", ["engine/src/lib.rs", "README.md"])
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["rust-documentation"]["steps"]
    runs = [step.get("run", "") for step in steps]
    assert "python tools/crate_format_check.py" in [run.strip() for run in runs]
