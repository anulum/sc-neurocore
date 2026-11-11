# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real filesystem, interpreter and loaded-source admission contracts

"""Require refusal before measurement when genuine admission conditions do not hold."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

# A parent measured with ``patch = subprocess`` exports these, and coverage's own
# startup hook then starts a second tracer in every child interpreter that has
# coverage installed. The unsupported interpreter is traced explicitly instead.
_COVERAGE_STARTUP_VARIABLES = ("COVERAGE_PROCESS_START", "COVERAGE_PROCESS_CONFIG")


def _explicitly_traced_environment() -> dict[str, str]:
    """Return the environment without coverage's automatic child-process startup."""
    return {k: v for k, v in os.environ.items() if k not in _COVERAGE_STARTUP_VARIABLES}


def _command(source: Path, checkout: Path, output: Path, python: Path | None = None) -> list[str]:
    """Build the actual source command, tracing a separately installed negative runtime when requested."""
    args = [str(python) if python is not None else sys.executable, "-B"]
    rcfile = os.environ.get("COVERAGE_RCFILE")
    if python is not None and rcfile:
        args += ["-m", "coverage", "run", "--timid", "--rcfile=" + rcfile]
    return [
        *args,
        str(source),
        "--checkout",
        str(checkout),
        "--output-dir",
        str(output),
        "--diagnostic",
    ]


@pytest.mark.parametrize("condition", ["missing-checkout", "changed-source", "existing-output"])
def test_real_filesystem_admission_refuses_before_measurement(
    tmp_path: Path, condition: str
) -> None:
    """Missing roots, altered original bytes and an occupied destination never reach measurement."""
    source = Path(__file__).resolve().parents[1] / "capture_lapicque_samples.py"
    checkout = Path(os.environ["LAPICQUE_ORIGINAL_CHECKOUT"])
    output = tmp_path / "output"
    if condition == "missing-checkout":
        checkout = tmp_path / "absent-checkout"
        expected = "FileNotFoundError"
    elif condition == "changed-source":
        original = checkout / "benchmarks/bench_model_lapicque.py"
        checkout = tmp_path / "changed-checkout"
        changed = checkout / "benchmarks/bench_model_lapicque.py"
        changed.parent.mkdir(parents=True)
        changed.write_bytes(original.read_bytes() + b"\\n")
        expected = "differs from the reviewed frozen source"
    else:
        output.mkdir()
        expected = "FileExistsError"
    result = subprocess.run(
        _command(source, checkout, output),
        env=os.environ,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0 and expected in result.stderr
    assert not (output / "original-harness-report.json").exists()
    assert not (output / "original-seven-duration-samples.json").exists()


@pytest.mark.parametrize(
    "fault,expected",
    [
        ("loader", "original benchmark loader unavailable"),
        ("steps", "original workload changed: N_STEPS"),
        ("repeats", "original workload changed: N_REPEATS"),
        ("backends", "original workload changed: BACKENDS"),
        ("measurement-type", "original function objects unavailable"),
        ("entrypoint-type", "original function objects unavailable"),
        ("measurement-origin", "measurement code does not belong to the original source"),
    ],
)
def test_real_loaded_admission_faults_refuse_before_measurement(
    tmp_path: Path, fault: str, expected: str
) -> None:
    """Real loader unavailability and damaged process-local metadata fail before original main."""
    source = Path(__file__).resolve().parents[1] / "capture_lapicque_samples.py"
    checkout = Path(os.environ["LAPICQUE_ORIGINAL_CHECKOUT"])
    output = tmp_path / "refused"
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(Path(__file__).with_name("lapicque_admission_driver.py")),
            str(source),
            str(checkout),
            str(output),
            fault,
        ],
        env=os.environ,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0 and expected in result.stderr
    assert not (output / "original-harness-report.json").exists()
    assert not (output / "original-seven-duration-samples.json").exists()


def test_actual_unsupported_interpreter_is_refused_before_measurement(tmp_path: Path) -> None:
    """A separately installed real unsupported minor fails before any backend is imported."""
    source = Path(__file__).resolve().parents[1] / "capture_lapicque_samples.py"
    checkout = Path(os.environ["LAPICQUE_ORIGINAL_CHECKOUT"])
    python = Path(os.environ["LAPICQUE_UNSUPPORTED_PYTHON"]).resolve(strict=True)
    version = subprocess.run(
        [
            str(python),
            "-B",
            "-c",
            "import sys;print(sys.implementation.name,sys.version_info.major,sys.version_info.minor)",
        ],
        env=_explicitly_traced_environment(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert version != "cpython 3 12"
    output = tmp_path / "unsupported"
    result = subprocess.run(
        _command(source, checkout, output, python),
        env=_explicitly_traced_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0 and "requires CPython 3.12" in result.stderr
    assert not output.exists()
