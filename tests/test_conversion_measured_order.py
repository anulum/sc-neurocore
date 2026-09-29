# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual measured dense IF selection acceptance

"""Exercise source-bound measured selection through public replay with genuine providers."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_bench_ann_to_snn_replay import (
    ROOT,
    SCRIPT,
    environment,
    go_library,
    mojo_library,
    rust_library,
)

__all__ = ["comparison", "environment", "go_library", "mojo_library", "rust_library"]


@pytest.fixture(scope="module")
def comparison(environment: dict[str, str], tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Generate an actual uninstrumented five-runtime report for the current live sources."""
    path = tmp_path_factory.mktemp("measured-order") / "comparison.json"
    settings = dict(environment)
    for key in ("COVERAGE_PROCESS_START", "COVERAGE_FILE", "SC_NEUROCORE_IF_BENCHMARK"):
        settings.pop(key, None)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(path),
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
    assert result.returncode == 0, result.stdout + result.stderr
    return path


_PROBE = r"""
import json
import os
import sys
measurement = None
if sys.argv[1]:
    import coverage
    measurement = coverage.Coverage(
        data_file=sys.argv[1], branch=True, source=[sys.argv[2]], config_file=False
    )
    measurement.start()
from sc_neurocore.conversion import ConvertedSNN
from sc_neurocore.conversion.if_dispatch import select_native
from sc_neurocore.conversion.if_native import load_native
model = ConvertedSNN([[[1.0]]], [None], [1.0], T=7)
mode = sys.argv[3]
if mode == "refuse":
    try:
        model.run([1.0])
    except RuntimeError as error:
        assert "dense IF comparison cannot order this runtime" in str(error), str(error)
    else:
        raise AssertionError("invalid explicit comparison accepted")
    assert model.run([1.0], backend="rust").tolist() == [7.0]
    assert model.run([1.0], backend="numpy").tolist() == [7.0]
else:
    assert model.run([1.0]).tolist() == [7.0]
    native = select_native("auto")
    expected = sys.argv[4]
    if expected == "julia":
        from sc_neurocore.conversion.if_julia import load_julia
        wanted = load_julia()
    else:
        wanted = load_native(os.environ["SC_NEUROCORE_IF_" + expected.upper() + "_LIB"])
    assert native.library is wanted.library
assert "torch" not in sys.modules
if measurement is not None:
    measurement.stop()
    measurement.save()
print("actual measured admission and public replay passed", flush=True)
"""


@pytest.mark.parametrize(
    "variation",
    [
        "valid",
        "other-cpu",
        "go-only",
        "bad-json",
        "stale-source",
        "stale-library",
        "instrumented",
        "bad-samples",
        "bad-median",
        "bad-aggregate",
        "bad-response",
        "missing-provider",
    ],
)
def test_measured_public_runtime_selection_and_report_refusal(
    comparison: Path, environment: dict[str, str], tmp_path: Path, variation: str
) -> None:
    """Use real timed providers; reject edited report inputs without changing explicit selection."""
    record = json.loads(comparison.read_text())
    expected = min(
        ("rust", "go", "mojo", "julia"), key=lambda n: record["backends"][n]["median_call_ms"]
    )
    settings = dict(environment)
    mode = "refuse"
    if variation == "valid":
        mode = "accept"
    elif variation == "other-cpu":
        record["meta"]["cpu"] += " other CPU"
        expected, mode = "rust", "accept"
    elif variation == "go-only":
        for key in (
            "SC_NEUROCORE_IF_RUST_LIB",
            "SC_NEUROCORE_IF_MOJO_LIB",
            "SC_NEUROCORE_IF_JULIA_ENABLED",
        ):
            settings.pop(key, None)
        expected, mode = "go", "accept"
    elif variation == "stale-source":
        record["source_sha256"]["pyproject.toml"] = "0" * 64
    elif variation == "stale-library":
        record["artifact_sha256"]["SC_NEUROCORE_IF_RUST_LIB"] = "0" * 64
    elif variation == "instrumented":
        record["meta"]["coverage_process_start_requested"] = True
    elif variation == "bad-samples":
        record["backends"]["rust"]["cases"][0]["samples_ns"] = []
    elif variation == "bad-median":
        record["backends"]["rust"]["cases"][0]["median_call_ms"] = -1
    elif variation == "bad-aggregate":
        record["backends"]["rust"]["median_call_ms"] = -1
    elif variation == "bad-response":
        record["backends"]["rust"]["cases"][0]["response_sha256"] = "0" * 64
    elif variation == "missing-provider":
        del record["backends"]["mojo"]
    path = tmp_path / "comparison.json"
    path.write_text("{broken" if variation == "bad-json" else json.dumps(record))
    settings["SC_NEUROCORE_IF_BENCHMARK"] = str(path)
    parent_data = os.environ.get("COVERAGE_FILE", "")
    child_data = f"{parent_data}-{variation}" if parent_data else ""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
            child_data,
            str(ROOT / "src/sc_neurocore/conversion"),
            mode,
            expected,
        ],
        cwd=ROOT,
        env=settings,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (tmp_path / "stdout.txt").write_text(result.stdout)
    (tmp_path / "stderr.txt").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "actual measured admission and public replay passed" in result.stdout
