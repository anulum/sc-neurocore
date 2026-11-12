# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Recorder refusals for incomplete source checkouts

"""Exercise checkout refusals and generated RTL binding through the real module CLI."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sc_neurocore.neurons.facet_receipts import credit_problems, load_receipt
from sc_neurocore.neurons.readiness import current_digest, derive_subjects, verify_receipt
from tools.facet_receipt import REPO_ROOT, run_command


def _copy_public_checkout_tree(
    directory: str,
    checkout: Path,
    *,
    budget: float,
    omitted: frozenset[str] = frozenset(),
) -> None:
    """Copy repository sources without ignored environments, caches or builds.

    Parameters
    ----------
    directory : str
        Source subtree relative to the actual working repository.
    checkout : pathlib.Path
        Isolated checkout receiving tracked and new nonignored source files.
    budget : float
        Measured finite allowance for inventorying the source subtree.
    omitted : frozenset[str], optional
        Exact basenames deliberately withdrawn by the refusal test.

    Raises
    ------
    subprocess.CalledProcessError
        If Git cannot inventory the real source checkout.
    subprocess.TimeoutExpired
        If source inventory exceeds the measured command allowance.
    OSError
        If a declared source cannot be copied.
    """
    inventory = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", directory],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
        timeout=budget,
    )
    for raw in inventory.stdout.split(b"\0"):
        if not raw:
            continue
        relative = Path(os.fsdecode(raw))
        assert relative.is_relative_to(directory)
        if relative.name in omitted:
            continue
        destination = checkout / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / relative, destination, follow_symlinks=False)


@pytest.mark.parametrize(
    ("omission", "model", "facet", "selector", "reason"),
    [
        (
            "mapping-tool",
            "AdaptiveThresholdIFNeuron",
            "cosim",
            "tests/test_cosim_adaptive_threshold_if.py"
            "::test_q3232_rtl_tracks_exact_relaxation_and_complete_event_vector",
            "formal RTL mapping tool is unavailable",
        ),
        (
            "model-subjects",
            "McCullochPittsNeuron",
            "class_validated",
            "tests/test_reference_mcculloch_pitts.py"
            "::test_truth_rows_match_an_independent_absolute_veto_rule",
            "cannot derive required subject kind(s)",
        ),
        (
            "model-descriptor",
            "McCullochPittsNeuron",
            "class_validated",
            "tests/test_reference_mcculloch_pitts.py"
            "::test_truth_rows_match_an_independent_absolute_veto_rule",
            "declares no evidence reference",
        ),
    ],
)
def test_cli_refuses_incomplete_checkout_inputs_without_running_a_validator(
    tmp_path: Path, omission: str, model: str, facet: str, selector: str, reason: str
) -> None:
    """Missing local inputs produce an authored refusal even with the library available."""
    started = time.monotonic()
    status, _, outcome, _ = run_command(
        [sys.executable, str(REPO_ROOT / "tools/facet_receipt.py"), "--help"], timeout=None
    )
    baseline = time.monotonic() - started
    assert status == 0 and outcome == "skipped"
    cpus = len(os.sched_getaffinity(0))
    budget = min(4 * 60 * 60, baseline * (1 + cpus) * max(1.0, os.getloadavg()[0] / cpus))
    checkout = tmp_path / "checkout"
    if omission != "model-subjects":
        omitted = (
            frozenset({"McCullochPittsNeuron.toml"})
            if omission == "model-descriptor"
            else frozenset()
        )
        _copy_public_checkout_tree("src", checkout, budget=budget, omitted=omitted)
    omitted = frozenset({"emit_catalogue_formal.py"}) if omission == "mapping-tool" else frozenset()
    _copy_public_checkout_tree("tools", checkout, budget=budget, omitted=omitted)
    copied_paths = [
        "tools/facet_receipt.py",
        "tools/facet_receipt_evidence.py",
        "tools/facet_receipt_runtime.py",
    ]
    if omission != "model-subjects":
        copied_paths.append("src/sc_neurocore/neurons/readiness.py")
    source_pins = {
        name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest() for name in copied_paths
    }
    assert all(
        hashlib.sha256((checkout / name).read_bytes()).hexdigest() == digest
        for name, digest in source_pins.items()
    )
    validator = Path(selector.split("::", 1)[0])
    (checkout / validator).parent.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / validator, checkout / validator)
    receipts = checkout / "receipts"
    raw = checkout / "docs/internal/raw"
    command = [
        sys.executable,
        "-m",
        "tools.facet_receipt",
        "record",
        "--model",
        model,
        "--facet",
        facet,
        "--receipt-dir",
        str(receipts),
        "--raw-evidence-dir",
        str(raw),
        "--",
        sys.executable,
        "-m",
        "pytest",
        selector,
        "-q",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT)])
    completed = subprocess.run(
        command,
        cwd=checkout,
        env=environment,
        capture_output=True,
        text=True,
        timeout=budget,
        check=False,
    )
    (tmp_path / "checkout-refusal-observation.json").write_text(
        json.dumps(
            {
                "omission": omission,
                "argv": command,
                "returncode": completed.returncode,
                "stderr": completed.stderr,
                "baseline_seconds": baseline,
                "budget_seconds": budget,
                "source_pins": source_pins,
            },
            indent=2,
        )
        + "\n"
    )
    assert all(
        hashlib.sha256((checkout / name).read_bytes()).hexdigest() == digest
        for name, digest in source_pins.items()
    )
    assert completed.returncode == 2
    assert reason in completed.stderr
    assert "Traceback" not in completed.stderr
    assert str(checkout) not in completed.stderr
    assert not receipts.exists()
    assert not raw.exists()


def test_cli_binds_generated_cosimulation_without_optional_committed_rtl(tmp_path: Path) -> None:
    """A real generated-RTL validator binds its schema and compiler without a committed RTL file."""
    started = time.monotonic()
    status, _, outcome, _ = run_command(
        [sys.executable, str(REPO_ROOT / "tools/facet_receipt.py"), "--help"], timeout=None
    )
    baseline = time.monotonic() - started
    assert status == 0 and outcome == "skipped"
    cpus = len(os.sched_getaffinity(0))
    budget = min(4 * 60 * 60, baseline * (1 + cpus) * max(1.0, os.getloadavg()[0] / cpus))
    checkout = tmp_path / "checkout"
    for directory in ("src", "tools"):
        _copy_public_checkout_tree(directory, checkout, budget=budget)
    validator = Path("tests/test_cosim_adaptive_threshold_if.py")
    (checkout / validator).parent.mkdir()
    shutil.copy2(REPO_ROOT / validator, checkout / validator)
    for support in [
        REPO_ROOT / "tests/__init__.py",
        *sorted((REPO_ROOT / "tests").glob("cosim_*.py")),
    ]:
        shutil.copy2(support, checkout / "tests" / support.name)
    assert not (checkout / "hdl").exists()
    receipts = checkout / "receipts"
    raw = checkout / "docs/internal/raw"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(checkout / "src"), str(checkout)])
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.facet_receipt",
            "record",
            "--model",
            "AdaptiveThresholdIFNeuron",
            "--facet",
            "cosim",
            "--receipt-dir",
            str(receipts),
            "--raw-evidence-dir",
            str(raw),
            "--timeout",
            str(budget),
            "--",
            sys.executable,
            "-m",
            "pytest",
            str(validator) + "::test_q3232_rtl_tracks_exact_relaxation_and_complete_event_vector",
            "-q",
        ],
        cwd=checkout,
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
    assert receipt.outcome == "passed" and receipt.exit_code == 0
    assert receipt.counts == {"collected": 1, "passed": 1, "failed": 0, "errors": 0, "skipped": 0}
    assert credit_problems(receipt) == ()
    assert verify_receipt(receipt, class_name="AdaptiveThresholdIFNeuron", repo_root=checkout) == (
        "bound",
        (),
        (),
    )
    assert "committed-rtl" not in {subject.kind for subject in receipt.subjects}
    assert {"compiler", "schema-profile", "model-module", "descriptor-contract", "validator"} <= {
        subject.kind for subject in receipt.subjects
    }
    archive = next(raw.glob("validator-*"))
    manifest = json.loads((archive / "manifest.json").read_bytes())
    assert manifest["cwd"] == str(checkout) and manifest["exit_code"] == 0
    for filename, digest in manifest["artifacts"].items():
        assert hashlib.sha256((archive / filename).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("addition", [False, True])
def test_cli_refuses_subject_membership_changes_during_a_real_validator(
    tmp_path: Path, addition: bool
) -> None:
    """A schema added or withdrawn during execution cannot receive validation credit."""
    started = time.monotonic()
    status, _, outcome, _ = run_command(
        [sys.executable, str(REPO_ROOT / "tools/facet_receipt.py"), "--help"], timeout=None
    )
    baseline = time.monotonic() - started
    assert status == 0 and outcome == "skipped"
    cpus = len(os.sched_getaffinity(0))
    budget = min(4 * 60 * 60, baseline * (1 + cpus) * max(1.0, os.getloadavg()[0] / cpus))
    interval = baseline / (1 + cpus)
    checkout = tmp_path / "checkout"
    for directory in ("src", "tools"):
        _copy_public_checkout_tree(directory, checkout, budget=budget)
    validator = "tests/test_reference_mcculloch_pitts.py"
    selector = validator + "::test_truth_rows_match_an_independent_absolute_veto_rule"
    (checkout / validator).parent.mkdir()
    shutil.copy2(REPO_ROOT / validator, checkout / validator)
    schema = checkout / "src/sc_neurocore/neurons/model_schemas/mcculloch_pitts.json"
    retained = schema.with_suffix(".json.retained")
    schema_digest = hashlib.sha256(schema.read_bytes()).hexdigest()
    if addition:
        schema.rename(retained)
    subjects = derive_subjects(
        "McCullochPittsNeuron", "class_validated", repo_root=checkout, evidence_refs=[selector]
    )
    pins = {subject.path: current_digest(subject, checkout) for subject in subjects}
    receipts = checkout / "receipts"
    raw = checkout / "docs/internal/raw"
    command = [
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
        str(raw),
        "--timeout",
        str(budget),
        "--",
        sys.executable,
        "-m",
        "pytest",
        selector,
        "-q",
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(checkout / "src"), str(checkout)])
    validator_pid: int | None = None
    with (
        (tmp_path / "stdout.bin").open("wb") as stdout,
        (tmp_path / "stderr.bin").open("wb") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=checkout,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + budget
            children = Path(f"/proc/{process.pid}/task/{process.pid}/children")
            while process.poll() is None and time.monotonic() < deadline:
                for child in children.read_text().split():
                    arguments = Path(f"/proc/{child}/cmdline").read_bytes().split(b"\0")
                    if selector.encode() in arguments and b"pytest" in arguments:
                        validator_pid = int(child)
                        os.kill(validator_pid, signal.SIGSTOP)
                        break
                if validator_pid is not None:
                    break
                time.sleep(interval)
            assert validator_pid is not None
            while time.monotonic() < deadline:
                state = Path(f"/proc/{validator_pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
                if state == "T":
                    break
                time.sleep(interval)
            assert state == "T"
            if addition:
                retained.rename(schema)
            else:
                schema.rename(retained)
            os.kill(validator_pid, signal.SIGCONT)
            assert process.wait(timeout=budget) == 1
        finally:
            if validator_pid is not None:
                try:
                    os.killpg(validator_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=budget)
    relative = schema.relative_to(checkout).as_posix()
    assert (
        hashlib.sha256((schema if addition else retained).read_bytes()).hexdigest() == schema_digest
    )
    receipt_paths = list(receipts.glob("*.json"))
    assert len(receipt_paths) == 1
    receipt = load_receipt(receipt_paths[0])
    assert receipt.exit_code == 0
    assert receipt.counts == {"collected": 1, "passed": 1, "failed": 0, "errors": 0, "skipped": 0}
    assert receipt.outcome == "error"
    assert "subject changed during run: subject manifest membership" in receipt.notes
    assert credit_problems(receipt)
    assert (
        verify_receipt(receipt, class_name="McCullochPittsNeuron", repo_root=checkout)[0]
        == "invalid"
    )
    assert (relative in {subject.path for subject in receipt.subjects}) != addition
    after = derive_subjects(
        "McCullochPittsNeuron", "class_validated", repo_root=checkout, evidence_refs=[selector]
    )
    assert (relative in {subject.path for subject in after}) == addition
    assert all(
        current_digest(subject, checkout) == pins[subject.path]
        for subject in subjects
        if subject.path != relative
    )
    archive = next(raw.glob("validator-*"))
    manifest = json.loads((archive / "manifest.json").read_bytes())
    assert manifest["exit_code"] == 0
    assert selector in manifest["executed_command"]
    for filename, digest in manifest["artifacts"].items():
        assert hashlib.sha256((archive / filename).read_bytes()).hexdigest() == digest
    (tmp_path / "membership-change-observation.json").write_text(
        json.dumps(
            {
                "addition": addition,
                "schema": relative,
                "schema_sha256": schema_digest,
                "validator_pid": validator_pid,
                "baseline_seconds": baseline,
                "budget_seconds": budget,
                "before": pins,
                "recorded_outcome": receipt.outcome,
                "scientific_receipt_credit": False,
            },
            indent=2,
        )
        + "\n"
    )
