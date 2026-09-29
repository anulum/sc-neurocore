# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Five-runtime QCFS comparison command acceptance

"""Run the real QCFS comparison and require refusal of every incomplete or drifted capture."""

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

from tests.test_conversion_qcfs_native import ROOT, qcfs_environment

__all__ = ["qcfs_environment"]

SCRIPT = ROOT / "benchmarks/bench_qcfs_runtimes.py"


def capture(
    settings: dict[str, str], output: Path, *arguments: str
) -> subprocess.CompletedProcess[str]:
    """Run the comparison command on the lowest allowed CPU.

    Parameters
    ----------
    settings:
        Process environment.
    output:
        Report path.
    arguments:
        Further command arguments.

    Returns
    -------
    subprocess.CompletedProcess
        The finished command.
    """
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output),
            "--cpu",
            str(min(os.sched_getaffinity(0))),
            *arguments,
        ],
        cwd=ROOT,
        env=settings,
        capture_output=True,
        text=True,
        timeout=900,
    )


def test_complete_capture_binds_five_runtimes(
    qcfs_environment: dict[str, str], tmp_path: Path
) -> None:
    """The programmatic entry publishes identical digests, raw samples and bound bytes for all five."""
    output = tmp_path / "qcfs.json"
    program = (
        "import sys; from bench_qcfs_runtimes import main; raise SystemExit(main(sys.argv[1:]))"
    )
    settings = dict(
        qcfs_environment,
        PYTHONPATH=os.pathsep.join((str(ROOT / "benchmarks"), qcfs_environment["PYTHONPATH"])),
    )
    cpu = str(min(os.sched_getaffinity(0)))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            "--output",
            str(output),
            "--cpu",
            cpu,
            "--samples",
            "3",
            "--warmup",
            "1",
        ],
        cwd=ROOT,
        env=settings,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(output.read_text())
    assert record["schema_version"] == "sc-neurocore.qcfs-comparison.v1"
    assert set(record["backends"]) == {"numpy", "rust", "go", "mojo", "julia"}
    reference = record["backends"]["numpy"]["cases"]
    assert [case["workload"] for case in reference] == [
        f"{operation}-n{size}"
        for size in (1, 64, 4096, 262144)
        for operation in ("forward", "backward")
    ]
    for backend, entry in record["backends"].items():
        assert entry["bit_parity"] is True
        for case, expected in zip(entry["cases"], reference, strict=True):
            assert case["response_sha256"] == expected["response_sha256"], backend
            assert len(case["samples_ns"]) == 3 and min(case["samples_ns"]) > 0
            assert case["median_ms"] > 0
        assert output.with_suffix(f".{backend}.stdout").is_file()
    for name, digest in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / "src/sc_neurocore" / name).read_bytes()).hexdigest() == digest
    for name, digest in record["artifact_sha256"].items():
        assert hashlib.sha256(Path(qcfs_environment[name]).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("argument,value", [("--samples", "2"), ("--warmup", "0"), ("--cpu", "-1")])
def test_invalid_arguments_preserve_existing_report(
    tmp_path: Path, argument: str, value: str
) -> None:
    """Incomplete budgets or a disallowed CPU are refused before anything is measured."""
    output = tmp_path / "qcfs.json"
    output.write_text("prior owner report")
    settings = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    command = [sys.executable, str(SCRIPT), "--output", str(output), "--cpu", "0", argument, value]
    result = subprocess.run(
        command, cwd=ROOT, env=settings, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 2
    assert output.read_text() == "prior owner report"


def test_missing_library_prevents_publication(
    qcfs_environment: dict[str, str], tmp_path: Path
) -> None:
    """An absent configured library stops the capture before any runtime is measured."""
    settings = dict(qcfs_environment, SC_NEUROCORE_QCFS_GO_LIB=str(tmp_path / "absent.so"))
    output = tmp_path / "qcfs.json"
    output.write_text("prior owner report")
    result = capture(settings, output)
    assert result.returncode != 0 and "absent.so" in result.stderr
    assert output.read_text() == "prior owner report"


_WRONG = r"""
#include <stddef.h>
#include <stdint.h>
uint32_t sc_qcfs_abi_version(void) { return 1; }
int32_t sc_qcfs_forward(uint32_t s, double t, const double *x, size_t n, double *o) {
    (void)s; (void)t; (void)x;
    for (size_t i = 0; i < n; ++i) o[i] = 0.0;
    return 0;
}
int32_t sc_qcfs_backward(uint32_t s, double t, const double *x, const double *g, size_t n,
                         double *i, double *h) {
    (void)s; (void)t; (void)x; (void)g;
    for (size_t k = 0; k < n; ++k) { i[k] = 0.0; h[k] = 0.0; }
    return 0;
}
"""


def test_wrong_bits_refuse_publication(qcfs_environment: dict[str, str], tmp_path: Path) -> None:
    """A real library answering with the wrong values fails its worker; nothing is published."""
    source = tmp_path / "wrong.c"
    source.write_text(_WRONG)
    library = tmp_path / "wrong.so"
    subprocess.run(
        ["cc", "-shared", "-fPIC", "-Wall", "-Werror", "-o", str(library), str(source)],
        capture_output=True,
        check=True,
        timeout=120,
    )
    output = tmp_path / "qcfs.json"
    output.write_text("prior owner report")
    result = capture(dict(qcfs_environment, SC_NEUROCORE_QCFS_RUST_LIB=str(library)), output)
    assert result.returncode != 0
    assert "differs from NumPy" in output.with_suffix(".rust.stderr").read_text()
    assert output.read_text() == "prior owner report"


def test_library_drift_during_capture_refuses_publication(
    qcfs_environment: dict[str, str], tmp_path: Path
) -> None:
    """A library changed while the runtimes are measured invalidates the whole capture."""
    library = tmp_path / "go.so"
    shutil.copyfile(qcfs_environment["SC_NEUROCORE_QCFS_GO_LIB"], library)
    output = tmp_path / "qcfs.json"
    output.write_text("prior owner report")
    process = subprocess.Popen(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output),
            "--cpu",
            str(min(os.sched_getaffinity(0))),
        ],
        cwd=ROOT,
        env=dict(qcfs_environment, SC_NEUROCORE_QCFS_GO_LIB=str(library)),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 300
        while not output.with_suffix(".numpy.stdout").exists():
            assert process.poll() is None, "comparison ended before the first runtime finished"
            assert time.monotonic() < deadline, "first runtime did not finish"
            time.sleep(0.05)
        with library.open("ab") as stream:
            stream.write(b"owned drift trailer")
        stdout, stderr = process.communicate(timeout=900)
        assert process.returncode != 0, stdout + stderr
        assert "changed during capture" in stderr
        assert output.read_text() == "prior owner report"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.communicate(timeout=30)
