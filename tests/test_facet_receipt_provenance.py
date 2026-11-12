# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real recorder provenance and public execution projection

"""Exercise actual scientific validators through both supported pytest commands."""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from xml.etree import ElementTree

import pytest

from sc_neurocore.neurons.facet_receipts import credit_problems, load_receipt
from sc_neurocore.neurons.readiness import verify_receipt
from sc_neurocore.neurons.receipt_execution import junit_checks
from tools.facet_receipt import REPO_ROOT, record_receipt

_VALIDATOR = (
    "tests/test_reference_mcculloch_pitts.py"
    "::test_truth_rows_match_an_independent_absolute_veto_rule"
)


@pytest.mark.parametrize("entrypoint", ["module", "console"])
def test_scientific_receipt_preserves_actual_execution_without_host_identity(
    tmp_path: Path, entrypoint: str
) -> None:
    """Publish equivalent scientific evidence while retaining exact raw custody."""
    command = (
        [sys.executable, "-m", "pytest"]
        if entrypoint == "module"
        else [str(Path(sys.executable).with_name("pytest"))]
    )
    selector = str(REPO_ROOT / _VALIDATOR) if entrypoint == "module" else _VALIDATOR
    command += [selector, "-q", "-p", "no:cacheprovider", "--rootdir=" + str(REPO_ROOT)]
    private = tmp_path / "private"
    path, receipt = record_receipt(
        model="McCullochPittsNeuron",
        facet="class_validated",
        command=command,
        receipt_dir=tmp_path / "receipts",
        raw_evidence_dir=private,
    )
    assert load_receipt(path) == receipt
    assert credit_problems(receipt) == ()
    assert verify_receipt(receipt, class_name="McCullochPittsNeuron")[0] == "bound"
    assert receipt.command == tuple(
        (["python", "-m", "pytest"] if entrypoint == "module" else ["pytest"])
        + [_VALIDATOR, "-q", "-p", "no:cacheprovider", "--rootdir=."]
    )
    assert receipt.runtime["implementation"] == platform.python_implementation()
    assert receipt.runtime["python"] == platform.python_version()
    assert (
        receipt.runtime["executable_sha256"]
        == hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
    )
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *(s.path for s in receipt.subjects)],
        cwd=REPO_ROOT,
        text=True,
    )
    assert receipt.runtime["dirty_subjects"] == ", ".join(
        sorted(line[3:] for line in dirty.splitlines())
    )
    public = path.read_text()
    for private_value in (str(REPO_ROOT), platform.node(), str(Path.home()), getpass.getuser()):
        assert private_value not in public
    assert 'hostname="' not in receipt.validator["junit_xml"]
    archives = list(private.glob("validator-*"))
    assert len(archives) == 1
    archive = archives[0]
    manifest_bytes = (archive / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert receipt.tool["raw_manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert manifest["command"] == command
    assert manifest["cwd"] == str(REPO_ROOT)
    assert manifest["exit_code"] == receipt.exit_code == 0
    for filename, digest in manifest["artifacts"].items():
        assert hashlib.sha256((archive / filename).read_bytes()).hexdigest() == digest
    raw_xml = (archive / "junit.xml").read_text()
    assert junit_checks(raw_xml) == junit_checks(receipt.validator["junit_xml"])
    raw_cases = ElementTree.fromstring(raw_xml).findall(".//testcase")
    public_cases = ElementTree.fromstring(receipt.validator["junit_xml"]).findall(".//testcase")
    assert len(raw_cases) == len(public_cases) > 0
    for raw, projected in zip(raw_cases, public_cases, strict=True):
        assert (raw.get("classname"), raw.get("name"), raw.get("time")) == (
            projected.get("classname"),
            projected.get("name"),
            projected.get("time"),
        )
    assert receipt.validator["public_projection"] == "repository-relative-execution.v1"
    assert "tools/facet_receipt_runtime.py" in {s.path for s in receipt.subjects}
    assert "tools/facet_receipt_evidence.py" in {s.path for s in receipt.subjects}


@pytest.mark.parametrize("public_override", [False, True])
def test_recorder_refuses_external_selectors_and_public_raw_archives(
    tmp_path: Path, public_override: bool
) -> None:
    """Refuse escaping commands and raw identity publication before execution."""
    from tools.facet_receipt import RecordError

    command = [sys.executable, "-m", "pytest", _VALIDATOR]
    private = REPO_ROOT / "src/rejected-validator-raw" if public_override else tmp_path / "raw"
    if not public_override:
        command.append("--rootdir=" + str(tmp_path))
    with pytest.raises(RecordError, match="private archive|outside the repository"):
        record_receipt(
            model="McCullochPittsNeuron",
            facet="class_validated",
            command=command,
            receipt_dir=tmp_path / "receipts",
            raw_evidence_dir=private,
        )
    assert not private.exists()
    assert not (tmp_path / "receipts").exists()


@pytest.mark.parametrize(
    "arguments", [["--collect-only"], ["-p", "no:tools.facet_receipt_runtime"]]
)
def test_collection_cannot_credit_and_plugin_opt_out_cannot_remove_provenance(
    tmp_path: Path, arguments: list[str]
) -> None:
    """Collection earns no credit; recorder-added provenance cannot be disabled."""
    path, receipt = record_receipt(
        model="McCullochPittsNeuron",
        facet="class_validated",
        command=[sys.executable, "-m", "pytest", _VALIDATOR, "-q", *arguments],
        receipt_dir=tmp_path / "receipts",
        raw_evidence_dir=tmp_path / "private",
    )
    assert load_receipt(path) == receipt
    if arguments == ["--collect-only"]:
        assert receipt.outcome == "error"
        assert "complete runtime provenance" in receipt.tool["provenance_error"]
        assert credit_problems(receipt)
        assert verify_receipt(receipt, class_name="McCullochPittsNeuron")[0] == "invalid"
    else:
        assert receipt.outcome == "passed"
        assert "provenance_error" not in receipt.tool
        assert receipt.runtime["python"] == platform.python_version()
        assert credit_problems(receipt) == ()
    assert len(list((tmp_path / "private").glob("validator-*"))) == 1


def test_recorder_refuses_absolute_subject_metadata_without_publishing(
    tmp_path: Path,
) -> None:
    """A genuine passing validator cannot publish a host-specific subject path."""
    from sc_neurocore.neurons.facet_receipts import FacetReceiptError, Subject

    validator_file = REPO_ROOT / _VALIDATOR.split("::")[0]
    subject = Subject(
        "validator", str(validator_file), hashlib.sha256(validator_file.read_bytes()).hexdigest()
    )
    with pytest.raises(FacetReceiptError, match="repository-relative"):
        record_receipt(
            model="McCullochPittsNeuron",
            facet="class_validated",
            command=[sys.executable, "-m", "pytest", _VALIDATOR, "-q"],
            extra_subjects=[subject],
            receipt_dir=tmp_path / "receipts",
            raw_evidence_dir=tmp_path / "private",
        )
    assert not (tmp_path / "receipts").exists()
    archives = list((tmp_path / "private").glob("validator-*"))
    assert len(archives) == 1
    assert (archives[0] / "junit.xml").is_file()


def test_posix_cli_interrupt_reaps_its_validator_and_retains_original_bytes(
    tmp_path: Path,
) -> None:
    """An interrupted actual validator leaves private artifacts and no public receipt."""
    started = time.monotonic()
    record_receipt(
        model="McCullochPittsNeuron",
        facet="class_validated",
        command=[sys.executable, "-m", "pytest", _VALIDATOR, "-q"],
        receipt_dir=tmp_path / "baseline-receipts",
        raw_evidence_dir=tmp_path / "baseline-raw",
    )
    measured_baseline = time.monotonic() - started
    cpus = len(os.sched_getaffinity(0))
    pressure = max(1.0, os.getloadavg()[0] / cpus)
    budget = min(4 * 60 * 60, measured_baseline * (1 + cpus) * pressure)
    interval = measured_baseline / (1 + cpus)
    archive = tmp_path / "interrupted-raw"
    receipts = tmp_path / "interrupted-receipts"
    command = [
        sys.executable,
        str(REPO_ROOT / "tools/facet_receipt.py"),
        "record",
        "--model",
        "McCullochPittsNeuron",
        "--facet",
        "class_validated",
        "--receipt-dir",
        str(receipts),
        "--raw-evidence-dir",
        str(archive),
        "--timeout",
        str(budget),
        "--",
        sys.executable,
        "-m",
        "pytest",
        _VALIDATOR,
        "-q",
    ]
    # Julia owns the caller's signal mask; only the fresh CLI must accept SIGINT.
    env_program = shutil.which("env")
    assert env_program is not None, "the interruption test requires GNU env"
    command = [env_program, "--default-signal=INT", *command]
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    deadline = time.monotonic() + budget
    validator_pid = 0
    try:
        while time.monotonic() < deadline and process.poll() is None:
            children = Path(f"/proc/{process.pid}/task/{process.pid}/children")
            for candidate in children.read_text().split():
                try:
                    argv = Path(f"/proc/{candidate}/cmdline").read_bytes().split(b"\0")
                except FileNotFoundError:
                    continue
                if _VALIDATOR.encode() in argv and b"pytest" in argv:
                    validator_pid = int(candidate)
                    break
            if validator_pid:
                break
            time.sleep(interval)
        assert validator_pid, "the real recorder never started its scientific validator"
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=max(interval, deadline - time.monotonic()))
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            process.communicate(timeout=budget)
    (tmp_path / "interruption-control.json").write_text(
        json.dumps(
            {
                "measured_baseline_seconds": measured_baseline,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "budget_seconds": budget,
                "poll_interval_seconds": interval,
                "recorder_argv": command,
                "recorder_pid": process.pid,
                "validator_pid": validator_pid,
                "returncode": process.returncode,
            },
            indent=2,
        )
        + "\n"
    )
    assert process.returncode != 0
    assert b"KeyboardInterrupt" in stderr
    assert b"wrote " not in stdout
    assert not receipts.exists()
    assert not Path(f"/proc/{validator_pid}").exists()
    archives = list(archive.glob("validator-*"))
    assert len(archives) == 1
    retained = archives[0]
    manifest = json.loads((retained / "manifest.json").read_bytes())
    assert manifest["exit_code"] == -signal.SIGKILL
    assert _VALIDATOR in manifest["executed_command"]
    for name, digest in manifest["artifacts"].items():
        assert hashlib.sha256((retained / name).read_bytes()).hexdigest() == digest
