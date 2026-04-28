# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Smoke tests for benchmark harness scripts (import +

"""Smoke tests for benchmark harness scripts (import + basic execution)."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path


def test_neurobench_harness_importable() -> None:
    """neurobench_harness module imports without error."""
    sys.path.insert(0, "benchmarks")
    try:
        neurobench_harness = importlib.import_module("neurobench_harness")

        assert hasattr(neurobench_harness, "NeuroBenchMetrics")
        assert hasattr(neurobench_harness, "BENCH_SUITE")
    finally:
        sys.path.pop(0)


def test_snn_comparison_importable() -> None:
    """snn_comparison module imports without error."""
    sys.path.insert(0, "benchmarks")
    try:
        snn_comparison = importlib.import_module("snn_comparison")

        assert hasattr(snn_comparison, "VARIANTS")
        assert hasattr(snn_comparison, "VariantResult")
    finally:
        sys.path.pop(0)


def test_cross_framework_harness_exposes_gap_runners_and_versions() -> None:
    """cross_framework_benchmark tracks opt-in gap runners and dependency versions."""
    sys.path.insert(0, "benchmarks")
    try:
        cross_framework_benchmark = importlib.import_module("cross_framework_benchmark")

        registry = cross_framework_benchmark._benchmark_registry()
        assert "nest" in registry
        assert "spikingjelly" in registry
        versions = cross_framework_benchmark.dependency_versions(
            ("definitely-missing-sc-neurocore-dependency",)
        )
        assert versions["definitely-missing-sc-neurocore-dependency"] is None
    finally:
        sys.path.pop(0)


def test_cross_framework_harness_writes_dependency_versions(tmp_path: Path) -> None:
    """Opt-in gap runner rows are represented in JSON even when deps are absent."""
    out = tmp_path / "cross_framework.json"
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/cross_framework_benchmark.py",
            "--scales",
            "5",
            "--skip-standalone",
            "--frameworks",
            "nest",
            "spikingjelly",
            "--json",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "dependency_versions" in payload
    assert {"NEST", "SpikingJelly"} <= {row["framework"] for row in payload["results"]}
    for row in payload["results"]:
        assert isinstance(row["mode"], str)
        assert isinstance(row["n_neurons"], int)


def test_fpga_deploy_list_parts() -> None:
    """fpga_deploy.py --list-parts runs without error."""
    result = subprocess.run(
        [sys.executable, "tools/fpga_deploy.py", "--list-parts"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    assert "xc7a35t" in result.stdout
    assert "Cyclone" in result.stdout


def test_fpga_deploy_emit_verilog(tmp_path: Path) -> None:
    """fpga_deploy.py --emit-verilog copies HDL files."""
    out_dir = tmp_path / "rtl_test"
    result = subprocess.run(
        [sys.executable, "tools/fpga_deploy.py", "--emit-verilog", "--out", str(out_dir)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    rtl_dir = out_dir / "rtl"
    assert rtl_dir.exists()
    assert any(rtl_dir.glob("*.v"))
