# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for benchmark harness contracts

"""Contracts for benchmark harness metadata and bounded CLI outputs."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module_name", "benchmark_name"),
    [
        ("bench_model_de_schutter_purkinje", "BenchmarkDeSchutterPurkinjeRK4"),
        ("bench_model_dendritic_nmda", "BenchmarkDendriticNMDARK4"),
        ("bench_model_durstewitz_dopamine", "BenchmarkDurstewitzRK4"),
        ("bench_model_golomb_fs", "BenchmarkGolombFSRK4"),
        ("bench_model_hay_l5", "BenchmarkHayL5RK4"),
        ("bench_model_hill_tononi", "BenchmarkHillTononiRK4"),
        ("bench_model_martinotti_neuron", "BenchmarkMartinottiRK4"),
        ("bench_model_multicompartment_mcn", "BenchmarkMulticompartmentMCNRK4"),
        ("bench_model_nlif", "BenchmarkNonlinearLIFRK4"),
        ("bench_model_pospischil", "BenchmarkPospischilRK4"),
        ("bench_model_pv_fast_spiking_neuron", "BenchmarkPVFastSpikingRK4"),
        ("bench_model_sfa", "BenchmarkSFARK4"),
        ("bench_model_mat", "BenchmarkMATSource"),
        ("bench_model_energy_lif", "BenchmarkEnergyLIFExactFlow"),
        ("bench_model_spike_response", "BenchmarkSpikeResponseKernel"),
        ("bench_model_spinnaker_lif", "BenchmarkSpiNNakerLIFExactFlow"),
        ("bench_model_srm0", "BenchmarkSRM0ExactFlow"),
        ("bench_model_sst_neuron", "BenchmarkSSTRK4"),
        ("bench_model_upper_motor_neuron", "BenchmarkUpperMotorExpEuler"),
        ("bench_model_vip_neuron", "BenchmarkVIPRK4"),
    ],
)
def test_go_benchmark_parser_accepts_optional_cpu_suffix(
    module_name: str, benchmark_name: str
) -> None:
    """Accept Go benchmark names with or without a GOMAXPROCS suffix."""

    sys.path.insert(0, "benchmarks")
    try:
        benchmark_module = importlib.import_module(module_name)
    finally:
        sys.path.pop(0)

    for emitted_name in (benchmark_name, f"{benchmark_name}-12"):
        match = benchmark_module.GO_BENCH_RE.match(  # type: ignore[attr-defined]
            f"{emitted_name}\t  200000\t  73.90 ns/op\t  4563 spikes"
        )
        assert match is not None
        assert match.group(1) == "73.90"


def test_cross_framework_harness_reports_missing_dependency_versions() -> None:
    sys.path.insert(0, "benchmarks")
    try:
        cross_framework_benchmark = importlib.import_module("cross_framework_benchmark")

        versions = cross_framework_benchmark.dependency_versions(
            ("definitely-missing-sc-neurocore-dependency",)
        )

        assert versions["definitely-missing-sc-neurocore-dependency"] is None
    finally:
        sys.path.pop(0)


def test_cross_framework_harness_records_opt_in_framework_rows(tmp_path: Path, monkeypatch) -> None:
    out = tmp_path / "cross_framework.json"

    sys.path.insert(0, "benchmarks")
    try:
        cross_framework_benchmark = importlib.import_module("cross_framework_benchmark")

        def fake_nest(n_neurons: int):
            return cross_framework_benchmark.BenchResult(
                framework="NEST",
                mode="iaf_psc_delta",
                n_neurons=n_neurons,
                wall_time_s=0.01,
                peak_memory_mb=1.0,
                n_spikes=3,
                rate_hz=2.0,
            )

        def fake_spikingjelly(n_neurons: int):
            return cross_framework_benchmark.BenchResult(
                framework="SpikingJelly",
                mode="PyTorch CPU",
                n_neurons=n_neurons,
                wall_time_s=0.02,
                peak_memory_mb=2.0,
                n_spikes=4,
                rate_hz=3.0,
            )

        monkeypatch.setattr(
            cross_framework_benchmark,
            "_benchmark_registry",
            lambda: {
                "nest": ("NEST", fake_nest),
                "spikingjelly": ("SpikingJelly", fake_spikingjelly),
            },
        )
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "cross_framework_benchmark.py",
                "--scales",
                "5",
                "--skip-standalone",
                "--frameworks",
                "nest",
                "spikingjelly",
                "--json",
                str(out),
            ],
        )
        cross_framework_benchmark.main()
    finally:
        sys.path.pop(0)

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "dependency_versions" in payload
    assert {"NEST", "SpikingJelly"} <= {row["framework"] for row in payload["results"]}
    for row in payload["results"]:
        assert isinstance(row["mode"], str)
        assert isinstance(row["n_neurons"], int)


def test_fpga_deploy_lists_supported_parts() -> None:
    result = subprocess.run(
        [sys.executable, "tools/fpga_deploy.py", "--list-parts"],
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "xc7a35t" in result.stdout
    assert "Cyclone" in result.stdout


def test_fpga_deploy_emit_verilog_creates_rtl_directory(tmp_path: Path) -> None:
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
