# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The HDL toolchain restores the one published SDK archive

"""Bind the shared HDL toolchain to the published SDK archive the receipts use.

The formal receipts record the SHA-256 of the solver binaries they ran. A
source build on a newer runner image produced a different Yices binary, so the
toolchain workflow restores one published archive by its hash and builds from
source only when a caller asks for it. These checks read the workflow and the
committed receipts; the hosted run checks the bytes themselves.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
TOOLCHAIN = "./.github/workflows/hdl-toolchain.yml"
SHA256 = re.compile(r"[0-9a-f]{64}")
RELEASE_ASSET = re.compile(
    r"https://github\.com/anulum/sc-neurocore/releases/download/"
    r"hdl-sdk-[a-z0-9-]+/scn-hdl\.tar\.gz"
)


def _load(path: Path) -> dict[str, Any]:
    """Return a workflow file parsed with the YAML loader the other contracts use."""
    return cast(dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8")))


def _toolchain() -> dict[str, Any]:
    """Return the shared HDL toolchain workflow."""
    return _load(WORKFLOWS / "hdl-toolchain.yml")


def _steps() -> list[dict[str, Any]]:
    """Return the toolchain job's steps in order."""
    return cast(list[dict[str, Any]], _toolchain()["jobs"]["build"]["steps"])


def _step(name: str) -> dict[str, Any]:
    """Return the one step with this name."""
    matches = [step for step in _steps() if step.get("name") == name]
    assert len(matches) == 1, f"exactly one step is named {name!r}"
    return matches[0]


def _condition(step: dict[str, Any]) -> str:
    """Return a step's `if` expression without its `${{ }}` wrapper, or an empty string."""
    raw = str(step.get("if", "")).strip()
    if raw.startswith("${{") and raw.endswith("}}"):
        raw = raw[3:-2].strip()
    return raw


def _runs_only_on_rebuild(step: dict[str, Any]) -> bool:
    """Return whether a step runs only when a caller asks for a source build."""
    condition = _condition(step)
    return condition.startswith("inputs.rebuild_sdk") and "!inputs.rebuild_sdk" not in condition


def _runs_only_on_published(step: dict[str, Any]) -> bool:
    """Return whether a step runs only on the default, published-archive path."""
    return _condition(step).startswith("!inputs.rebuild_sdk")


def test_rebuild_is_an_explicit_boolean_input_off_by_default() -> None:
    """A source build happens only when a caller passes `rebuild_sdk: true`."""
    workflow = _toolchain()
    events = cast(dict[str, Any], workflow.get("on", cast(dict[Any, Any], workflow).get(True)))
    assert list(events) == ["workflow_call"]
    rebuild = events["workflow_call"]["inputs"]["rebuild_sdk"]
    assert rebuild["type"] == "boolean"
    assert rebuild["default"] is False
    assert workflow["permissions"] == {"contents": "read"}


def test_every_caller_restores_the_published_archive() -> None:
    """No workflow that runs on its own asks the toolchain for a source build."""
    callers: list[str] = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for job in _load(path).get("jobs", {}).values():
            if job.get("uses") == TOOLCHAIN:
                callers.append(path.name)
                assert job.get("with", {}).get("rebuild_sdk", False) is False, path.name
    assert callers == ["audit-cadence.yml", "ci.yml", "yosys-synth.yml"]


def test_the_published_archive_and_its_solvers_are_pinned_by_hash() -> None:
    """The archive's URL is a release asset of this repository; every pin is a SHA-256."""
    env = _toolchain()["env"]
    assert RELEASE_ASSET.fullmatch(env["SCN_HDL_SDK_URL"])
    for name in ("SCN_HDL_SDK_SHA256", "SCN_HDL_SDK_YICES_SHA256", "SCN_HDL_SDK_Z3_SHA256"):
        assert SHA256.fullmatch(env[name]), name
    restore = _step("Restore the published SDK archive")
    assert _runs_only_on_published(restore)
    assert restore["with"]["key"] == "scn-hdl-sdk-${{ env.SCN_HDL_SDK_SHA256 }}"
    download = _step("Download the published SDK archive")
    assert _runs_only_on_published(download)
    assert "steps.published.outputs.cache-hit != 'true'" in _condition(download)
    assert '"$SCN_HDL_SDK_URL"' in download["run"]


def test_the_archive_is_checked_before_it_is_unpacked() -> None:
    """Hash, then member names and types, then extraction, then both solver binaries."""
    unpack = _step("Unpack the published SDK archive after checking its pinned hash")
    assert _runs_only_on_published(unpack)
    run = cast(str, unpack["run"])
    order = [
        run.index('echo "$SCN_HDL_SDK_SHA256  $archive" | sha256sum --check'),
        run.index("$0 !~ /^scn-hdl(\\/|$)/"),
        run.index("substr($1, 1, 1) !~ /^[-d]$/"),
        run.index('tar -xzf "$archive" -C "$RUNNER_TEMP"'),
        run.index('echo "$SCN_HDL_SDK_YICES_SHA256  $RUNNER_TEMP/scn-hdl/bin/yices-smt2"'),
        run.index('echo "$SCN_HDL_SDK_Z3_SHA256  $RUNNER_TEMP/scn-hdl/bin/z3"'),
    ]
    assert order == sorted(order)
    assert run.count("tar -xzf") == 1


def test_the_solver_pins_are_the_binaries_every_committed_receipt_ran() -> None:
    """A receipt recorded with another solver binary fails here, not on the hosted run."""
    env = _toolchain()["env"]
    pins = {
        "yices-smt2": env["SCN_HDL_SDK_YICES_SHA256"],
        "z3": env["SCN_HDL_SDK_Z3_SHA256"],
    }
    seen: set[str] = set()
    receipts = sorted((ROOT / "benchmarks" / "results" / "formal").glob("*/receipt.json"))
    assert receipts, "the repository holds formal receipts"
    for receipt in receipts:
        solver = json.loads(receipt.read_text(encoding="utf-8"))["toolchain"]["solver"]
        assert solver["sha256"] == pins[solver["name"]], receipt.parent.name
        seen.add(solver["name"])
    assert seen == set(pins)


def test_a_source_build_runs_only_on_request() -> None:
    """Building, its notices and its provenance never run on the default path."""
    for name in (
        "Identify the actual compiler and build dependencies",
        "Restore the source-pinned HDL tools",
        "Build the verified stable sources once",
        "Retain upstream and bundled dependency notices",
        "Record the provenance of a source build and package it",
    ):
        assert _runs_only_on_rebuild(_step(name)), name
    for name in (
        "Build the verified stable sources once",
        "Retain upstream and bundled dependency notices",
    ):
        assert "steps.tools.outputs.cache-hit != 'true'" in _condition(_step(name)), name
    provenance = _step("Record the provenance of a source build and package it")
    assert '} > "$tool_prefix/PROVENANCE.txt"' in provenance["run"]


def test_both_paths_run_the_same_verification_and_keep_the_published_provenance() -> None:
    """The checks run on every path; only a source build writes PROVENANCE.txt."""
    verify = _step("Verify the toolchain for this run")
    assert "if" not in verify
    run = cast(str, verify["run"])
    assert "sha256sum --check share/yices-build/SHA256SUMS" in run
    assert 'test "$("$tool_prefix/bin/yices-smt2" --version | head -1)" = "Yices 2.7.0"' in run
    assert '"$tool_prefix/PROVENANCE.txt"' not in run
    assert "tar " not in run


def test_the_published_path_hands_on_the_exact_bytes() -> None:
    """The uploaded archive is the published one, compared with its pin once more."""
    hand_on = _step("Hand on the published SDK archive unchanged")
    assert _runs_only_on_published(hand_on)
    run = cast(str, hand_on["run"])
    assert 'cp "$RUNNER_TEMP/scn-hdl-published/scn-hdl.tar.gz" "$RUNNER_TEMP/scn-hdl.tar.gz"' in run
    assert 'test "$(cut -d \' \' -f 1 scn-hdl.tar.gz.sha256)" = "$SCN_HDL_SDK_SHA256"' in run
    upload = _steps()[-1]
    assert str(upload["uses"]).startswith("actions/upload-artifact@")
    assert upload["with"]["name"] == "stable-hdl-sdk"
    assert upload["with"]["path"].splitlines() == [
        "${{ runner.temp }}/scn-hdl.tar.gz",
        "${{ runner.temp }}/scn-hdl.tar.gz.sha256",
    ]
