# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual framework interchange benchmark evidence

"""Run the real benchmark command and verify measurements and producer binding."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import sc_neurocore

pytest.importorskip("snntorch")
pytest.importorskip("nirtorch")

_ROOT = Path(__file__).resolve().parents[1]


def test_real_framework_benchmark_checks_each_path_before_reporting_times(tmp_path: Path) -> None:
    """Measure both reset conventions and bind actual traces to their imported producers."""
    script = _ROOT / "benchmarks/bench_nir_interop.py"
    output = tmp_path / "framework-benchmark.json"
    env = dict(os.environ)
    if env.get("SC_NEUROCORE_INSTALLED_INTEROP") == "1":
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = str(_ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--steps",
            "64",
            "--warmups",
            "0",
            "--iterations",
            "1",
            "--output",
            str(output),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(output.read_text())
    assert report["benchmark_sha256"] == hashlib.sha256(script.read_bytes()).hexdigest()
    assert report["source_hashes"]
    package_parent = Path(sc_neurocore.__file__).resolve().parent.parent
    for relative, expected in report["source_hashes"].items():
        assert hashlib.sha256((package_parent / relative).read_bytes()).hexdigest() == expected
    assert report["configuration"]["dtype"] == "float64"
    assert report["configuration"]["device"] == "cpu"
    assert report["measurement_context"]["production_rerun_requirement"]
    assert {case["sc_reset"] for case in report["cases"]} == {"reset", "snntorch-subtract"}
    for case in report["cases"]:
        assert case["spikes_identical"] and case["spikes_per_trace"] > 0
        assert case["final_membrane_max_abs_error"] <= 1e-12
        assert len(case["nir_file_sha256"]) == len(case["trace_sha256"]) == 64
        assert set(case["paths"]) == {
            "source_snntorch",
            "sc_reference",
            "sc_nir_roundtrip",
            "exported_snntorch",
        }
        for measured in case["paths"].values():
            assert len(measured["elapsed_ns"]) == 1 and measured["elapsed_ns"][0] > 0
            assert measured["median_ns_per_step"] == measured["elapsed_ns"][0] / 64


def test_committed_framework_measurements_remain_bound_to_actual_producers() -> None:
    """Require complete measured samples and the current installed producer bytes."""
    report = json.loads((_ROOT / "benchmarks/results/bench_nir_interop.json").read_text())
    assert report["configuration"]["iterations"] == 30
    assert report["configuration"]["warmups"] == 2
    assert (
        report["benchmark_sha256"]
        == hashlib.sha256((_ROOT / "benchmarks/bench_nir_interop.py").read_bytes()).hexdigest()
    )
    package_parent = Path(sc_neurocore.__file__).resolve().parent.parent
    for relative, expected in report["source_hashes"].items():
        assert hashlib.sha256((package_parent / relative).read_bytes()).hexdigest() == expected
    for case in report["cases"]:
        assert case["spikes_identical"] and case["spikes_per_trace"] > 0
        for path in case["paths"].values():
            assert len(path["elapsed_ns"]) == 30
            assert path["median_ns_per_step"] > 0
