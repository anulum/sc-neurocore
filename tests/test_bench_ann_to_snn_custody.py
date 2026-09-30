# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dense IF comparison result custody

"""Refuse real numerical profile errors and concurrent source drift through the actual command."""

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Literal

import pytest

from tests.test_bench_ann_to_snn_replay import (
    ROOT,
    SCRIPT,
    environment as environment,
    rust_library as rust_library,
    go_library as go_library,
    mojo_library as mojo_library,
)
from sc_neurocore.accel.mojo.isa_baseline import pin_isa


def test_fused_actual_mojo_library_refuses_full_comparison_and_preserves_report(
    environment: dict[str, str],
    tmp_path: Path,
) -> None:
    """Compile the real kernel with default FMA and require a full-response refusal."""
    kernels = ROOT / "src/sc_neurocore/accel/mojo/kernels"
    library = tmp_path / "fused.so"
    subprocess.run(
        pin_isa(
            [
                "mojo",
                "build",
                "--Werror",
                "--diagnose-missing-doc-strings",
                "-I",
                str(kernels),
                "--emit",
                "shared-lib",
                "-o",
                str(library),
                str(kernels / "ann_to_snn_native.mojo"),
            ]
        ),
        capture_output=True,
        check=True,
        timeout=120,
    )
    output = tmp_path / "report.json"
    output.write_text("prior owner report")
    settings = dict(environment, SC_NEUROCORE_IF_MOJO_LIB=str(library))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output),
            "--cpu",
            str(min(os.sched_getaffinity(0))),
            "--samples",
            "3",
            "--warmup",
            "1",
        ],
        cwd=ROOT,
        env=settings,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert output.read_text() == "prior owner report"
    assert "complete response mismatch" in output.with_suffix(".mojo.stderr").read_text()


@pytest.mark.parametrize("drift", ["source", "artifact"])
def test_actual_source_or_artifact_drift_refuses_atomic_publication(
    environment: dict[str, str], tmp_path: Path, drift: Literal["source", "artifact"]
) -> None:
    """Change owned fixture metadata or an actual provider during a live comparison."""
    fixture = tmp_path / "source"
    benchmark = fixture / "benchmarks"
    benchmark.mkdir(parents=True)
    (fixture / "src").symlink_to(ROOT / "src", target_is_directory=True)
    for source in (ROOT / "benchmarks").glob("*ann_to_snn_replay*.py"):
        shutil.copyfile(source, benchmark / source.name)
    metadata = fixture / "pyproject.toml"
    shutil.copyfile(ROOT / "pyproject.toml", metadata)
    output = tmp_path / "report.json"
    output.write_text("prior owner report")
    settings = dict(environment)
    provider = tmp_path / "actual-go.so"
    shutil.copyfile(environment["SC_NEUROCORE_IF_GO_LIB"], provider)
    settings["SC_NEUROCORE_IF_GO_LIB"] = str(provider)
    process = subprocess.Popen(
        [
            sys.executable,
            str(benchmark / SCRIPT.name),
            "--output",
            str(output),
            "--cpu",
            str(min(os.sched_getaffinity(0))),
            "--samples",
            "3",
            "--warmup",
            "1",
        ],
        cwd=fixture,
        env=settings,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 120
        while not output.with_suffix(".numpy.stdout").exists():
            assert process.poll() is None, "comparison ended before actual first-worker receipt"
            assert time.monotonic() < deadline, "actual first-worker receipt did not arrive"
            time.sleep(0.05)
        assert process.poll() is None
        if drift == "source":
            metadata.write_text(metadata.read_text() + "\n# owned fixture source drift\n")
        else:
            with provider.open("ab") as stream:
                stream.write(b"owned fixture artifact trailer")
        stdout, stderr = process.communicate(timeout=300)
        assert process.returncode != 0, stdout + stderr
        assert "runtime sources or configured artifacts changed during capture" in stderr
        assert output.read_text() == "prior owner report"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.communicate(timeout=30)
