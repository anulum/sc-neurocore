# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Recorder CLI input and evidence contracts

"""Exercise recorder arguments and real selected validator execution through its CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sc_neurocore.neurons.facet_receipts import credit_problems, load_receipt
from sc_neurocore.neurons.readiness import verify_receipt
from tests.facet_receipt_budget_support import command_budget
from tools.facet_receipt import REPO_ROOT, main, run_command

_VALIDATOR = (
    "tests/test_reference_mcculloch_pitts.py"
    "::test_truth_rows_match_an_independent_absolute_veto_rule"
)


@pytest.mark.parametrize(
    ("options", "reason"),
    [
        (["--model", "UnregisteredNeuron"], "not a registered class"),
        (["--facet", "unknown-facet"], "unknown facet"),
        (["--profile", "unknown-profile"], "unknown profile"),
        (["--model", "LapicqueNeuron"], "multiple model profiles"),
        (["--facet", "backend:python"], "declares no evidence reference"),
        (["--evidence", "tests/nonexistent_validator.py"], "does not resolve"),
        (["--evidence", "tests/test_receipt_execution.py"], "outside declared evidence"),
        (["--subject", "validator"], "expects KIND=PATH"),
        (["--subject", "unknown=README.md"], "is not one of"),
        (["--subject", "validator=tests/nonexistent_validator.py"], "does not exist"),
    ],
)
def test_cli_refuses_invalid_identity_evidence_and_subjects_before_execution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], options: list[str], reason: str
) -> None:
    """Reject unreviewed or unresolved recording inputs without starting a validator."""
    receipts = tmp_path / "receipts"
    private = tmp_path / "raw"
    status = main(
        [
            "record",
            "--model",
            "McCullochPittsNeuron",
            "--facet",
            "class_validated",
            "--receipt-dir",
            str(receipts),
            "--raw-evidence-dir",
            str(private),
            *options,
            "--",
            sys.executable,
            "-m",
            "pytest",
            _VALIDATOR,
            "-q",
        ]
    )
    assert status == 2
    assert reason in capsys.readouterr().err
    assert not receipts.exists()
    assert not private.exists()


@pytest.mark.parametrize("collect_only", [False, True])
def test_cli_explicit_validator_subject_preserves_execution_and_collection_credit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], collect_only: bool
) -> None:
    """An explicit real subject records actual checks; collection still earns no credit."""
    receipts = tmp_path / "receipts"
    status = main(
        [
            "record",
            "--model",
            "McCullochPittsNeuron",
            "--facet",
            "class_validated",
            "--subject",
            "validator=" + _VALIDATOR.split("::")[0],
            "--receipt-dir",
            str(receipts),
            "--raw-evidence-dir",
            str(tmp_path / "raw"),
            "--",
            sys.executable,
            "-m",
            "pytest",
            _VALIDATOR,
            "-q",
            *(["--collect-only"] if collect_only else []),
        ]
    )
    assert status == int(collect_only)
    receipt = load_receipt(next(receipts.glob("*.json")))
    assert receipt.counts["passed"] == int(not collect_only)
    assert bool(credit_problems(receipt)) == collect_only
    assert ("not creditable:" in capsys.readouterr().out) == collect_only


def test_cli_requires_a_validator_before_creating_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A model and facet alone cannot create a receipt or a raw execution archive."""
    receipts = tmp_path / "receipts"
    private = tmp_path / "raw"
    with pytest.raises(SystemExit) as error:
        main(
            [
                "record",
                "--model",
                "McCullochPittsNeuron",
                "--facet",
                "class_validated",
                "--receipt-dir",
                str(receipts),
                "--raw-evidence-dir",
                str(private),
            ]
        )
    assert error.value.code == 2
    assert "an evidence command is required" in capsys.readouterr().err
    assert not receipts.exists()
    assert not private.exists()


def test_cli_cosimulation_binds_registered_committed_rtl_without_an_override(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run the registered scientific validator and bind its automatically located RTL."""
    receipts = tmp_path / "receipts"
    status = main(
        [
            "record",
            "--model",
            "AdaptiveThresholdIFNeuron",
            "--facet",
            "cosim",
            "--profile",
            "adaptive_threshold_if",
            "--receipt-dir",
            str(receipts),
            "--raw-evidence-dir",
            str(tmp_path / "raw"),
            "--",
            sys.executable,
            "-m",
            "pytest",
            "tests/test_cosim_adaptive_threshold_if.py"
            "::test_q3232_rtl_tracks_exact_relaxation_and_complete_event_vector",
            "-q",
        ]
    )
    assert status == 0, capsys.readouterr().out
    receipt = load_receipt(next(receipts.glob("*.json")))
    assert receipt.counts["passed"] == 1
    assert credit_problems(receipt) == ()
    assert verify_receipt(receipt, class_name="AdaptiveThresholdIFNeuron")[0] == "bound"
    assert [
        (item.path, item.scope) for item in receipt.subjects if item.kind == "committed-rtl"
    ] == [("hdl/formal/catalogue/sc_adaptive_threshold_if.v", "file")]


def test_cli_cannot_credit_an_execution_with_junit_reporting_disabled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An actual pytest startup refusal retains raw custody and earns no validation credit."""
    receipts = tmp_path / "receipts"
    private = tmp_path / "raw"
    status = main(
        [
            "record",
            "--model",
            "McCullochPittsNeuron",
            "--facet",
            "class_validated",
            "--receipt-dir",
            str(receipts),
            "--raw-evidence-dir",
            str(private),
            "--",
            sys.executable,
            "-m",
            "pytest",
            _VALIDATOR,
            "-q",
            "-p",
            "no:junitxml",
        ]
    )
    assert status == 1, capsys.readouterr().err
    receipt = load_receipt(next(receipts.glob("*.json")))
    assert receipt.outcome == "error"
    assert receipt.exit_code != 0
    assert receipt.counts["passed"] == 0
    assert receipt.counts["errors"] == 1
    assert credit_problems(receipt)
    assert verify_receipt(receipt, class_name="McCullochPittsNeuron")[0] == "invalid"
    archive = next(private.glob("validator-*"))
    assert (archive / "junit.xml").read_bytes() == b""
    assert b"unrecognized arguments" in (archive / "stderr.bin").read_bytes()


def test_cli_refuses_an_unavailable_validator_executable_without_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refuse an actual startup failure without exposing interpreter diagnostics or paths."""
    receipts = tmp_path / "receipts"
    private = tmp_path / "raw"
    missing = tmp_path / "python-unavailable"
    status = main(
        [
            "record",
            "--model",
            "McCullochPittsNeuron",
            "--facet",
            "class_validated",
            "--receipt-dir",
            str(receipts),
            "--raw-evidence-dir",
            str(private),
            "--",
            str(missing),
            "-m",
            "pytest",
            _VALIDATOR,
            "-q",
        ]
    )
    assert status == 2
    stderr = capsys.readouterr().err
    assert "validator process could not be started" in stderr
    assert "Traceback" not in stderr
    assert str(missing) not in stderr
    assert not receipts.exists()
    assert not private.exists()


def test_cli_preserves_scientific_source_binding_without_git(tmp_path: Path) -> None:
    """An installation without Git still seals the actual validator and source digests."""
    started = time.monotonic()
    status, counts, outcome, _ = run_command(
        [sys.executable, str(REPO_ROOT / "tools/facet_receipt.py"), "--help"], timeout=None
    )
    baseline = time.monotonic() - started
    assert status == 0 and counts["passed"] == 0 and outcome == "skipped"
    budget = command_budget(baseline)
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    environment = dict(os.environ)
    environment["PATH"] = str(empty_path)
    assert shutil.which("git", path=environment["PATH"]) is None
    receipts = tmp_path / "receipts"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.facet_receipt",
            "record",
            "--model",
            "McCullochPittsNeuron",
            "--facet",
            "class_validated",
            "--receipt-dir",
            str(receipts),
            "--raw-evidence-dir",
            str(tmp_path / "raw"),
            "--timeout",
            str(budget),
            "--",
            sys.executable,
            "-m",
            "pytest",
            _VALIDATOR,
            "-q",
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=budget,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    paths = list(receipts.glob("*.json"))
    assert len(paths) == 1
    receipt = load_receipt(paths[0])
    assert receipt.outcome == "passed"
    assert receipt.counts == {"collected": 1, "passed": 1, "failed": 0, "errors": 0, "skipped": 0}
    assert receipt.runtime["git_head"] == receipt.runtime["dirty_subjects"] == ""
    assert credit_problems(receipt) == ()
    assert verify_receipt(receipt, class_name="McCullochPittsNeuron")[0] == "bound"
