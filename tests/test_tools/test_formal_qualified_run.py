# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native proof qualification entry point

"""Exercise the command used by hosted formal jobs against actual catalogue RTL."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tools.formal_qualified_run import main

ROOT = Path(__file__).resolve().parents[2]
JOB = Path("hdl/formal/catalogue/benda_herz.sby")


def test_actual_cli_proof_and_audit_share_one_work_root(tmp_path: Path) -> None:
    """A real native proof qualifies and refuses to overwrite the retained invocation."""
    work, report = tmp_path / "work", tmp_path / "report.json"
    args = [str(JOB), "--repository", str(ROOT), "--work-root", str(work), "--report", str(report)]
    assert main(args) == 0
    result = json.loads(report.read_text())
    assert result["status"] == result["primary_status"] == "PASS"
    assert result["nonvacuity_status"] == "NONVACUOUS"
    assert result["tasks"][""]["first_reached_steps"] == {"scn_assertion_enable_000000": 1}
    original = (work / "primary.log").read_bytes()
    assert main(args) == 2
    assert (work / "primary.log").read_bytes() == original


def test_real_reset_counterexample_stays_a_failure(tmp_path: Path) -> None:
    """A one-bit reset defect in actual catalogue RTL remains a primary FAIL."""
    directory = tmp_path / JOB.parent
    directory.mkdir(parents=True)
    for suffix in (".sby", ".v", "_formal.v"):
        source = ROOT / JOB.parent / (JOB.stem + suffix)
        shutil.copyfile(source, directory / source.name)
    rtl = directory / "benda_herz.v"
    original = rtl.read_text()
    reset = "if (rst) begin adaptation_q <= 0; phase_q <= 0; spike <= 0; end"
    assert original.count(reset) == 1
    rtl.write_text(original.replace(reset, reset.replace("spike <= 0", "spike <= 1")))
    report = tmp_path / "report.json"
    assert (
        main(
            [
                str(JOB),
                "--repository",
                str(tmp_path),
                "--work-root",
                str(tmp_path / "work"),
                "--report",
                str(report),
            ]
        )
        == 1
    )
    result = json.loads(report.read_text())
    assert result["status"] == result["primary_status"] == "FAIL"
    assert result["nonvacuity_status"] == "NOT_RUN"


def test_missing_job_remains_unqualified(tmp_path: Path) -> None:
    """The real source resolver refuses a missing job before proof execution."""
    report = tmp_path / "report.json"
    assert (
        main(
            [
                "missing.sby",
                "--repository",
                str(tmp_path),
                "--work-root",
                str(tmp_path / "work"),
                "--report",
                str(report),
            ]
        )
        == 2
    )
    assert json.loads(report.read_text())["primary_status"] == "NOT_RUN"


def test_real_multiple_assertions_each_need_a_reached_enable(tmp_path: Path) -> None:
    """Qualify every actual Halton assertion through the command-line entry point."""
    report = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.formal_qualified_run",
            "hdl/formal/halton16.sby",
            "--repository",
            str(ROOT),
            "--work-root",
            str(tmp_path / "work"),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(report.read_text())
    task = data["tasks"]["prove"]
    assert len(task["assertions"]) == 3
    assert set(task["first_reached_steps"]) == set(task["assertions"])


@pytest.mark.parametrize("case", ["vacuous", "empty", "native-error", "cover"])
def test_real_native_qualification_boundaries(tmp_path: Path, case: str) -> None:
    """Keep primary success distinct from absent or vacuous proof qualification."""
    directory = tmp_path / JOB.parent
    directory.mkdir(parents=True)
    for suffix in (".sby", ".v", "_formal.v"):
        source = ROOT / JOB.parent / (JOB.stem + suffix)
        shutil.copyfile(source, directory / source.name)
    job = directory / JOB.name
    if case == "vacuous":
        harness = directory / "benda_herz_formal.v"
        boundary = (
            "end\n// Check the post-reset state directly within the existing depth-two bound.\n"
            "always @(*) begin\n"
        )
        original = harness.read_text()
        assert original.count(boundary) == 1
        harness.write_text(original.replace(boundary, ""))
    elif case == "empty":
        job.write_text(job.read_text().replace("[files]", "chformal -assert -remove\n[files]"))
    elif case == "native-error":
        job.write_text(job.read_text().replace("[script]", "[script]\nread -formal absent.v"))
    else:
        job.write_text(job.read_text().replace("mode bmc", "mode cover"))
        harness = directory / "benda_herz_formal.v"
        harness.write_text(harness.read_text().replace("assert(", "cover("))
    report = tmp_path / "report.json"
    result = main(
        [
            str(job),
            "--repository",
            str(tmp_path),
            "--work-root",
            str(tmp_path / "work"),
            "--report",
            str(report),
        ]
    )
    data = json.loads(report.read_text())
    assert result == (0 if case == "cover" else 2)
    assert data["primary_status"] == ("ERROR" if case == "native-error" else "PASS")
    assert data["status"] == ("PASS" if case == "cover" else "ERROR")
    if case in ("vacuous", "empty"):
        assert data["nonvacuity_status"] == "ERROR"
