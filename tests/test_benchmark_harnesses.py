# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Smoke tests for benchmark harness scripts (import + basic...

"""Smoke tests for benchmark harness scripts (import + basic execution)."""

from __future__ import annotations

import subprocess
import sys


def test_neurobench_harness_importable():
    """neurobench_harness module imports without error."""
    sys.path.insert(0, "benchmarks")
    try:
        import neurobench_harness

        assert hasattr(neurobench_harness, "NeuroBenchMetrics")
        assert hasattr(neurobench_harness, "BENCH_SUITE")
    finally:
        sys.path.pop(0)


def test_snn_comparison_importable():
    """snn_comparison module imports without error."""
    sys.path.insert(0, "benchmarks")
    try:
        import snn_comparison

        assert hasattr(snn_comparison, "VARIANTS")
        assert hasattr(snn_comparison, "VariantResult")
    finally:
        sys.path.pop(0)


def test_fpga_deploy_list_parts():
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


def test_fpga_deploy_emit_verilog(tmp_path):
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
