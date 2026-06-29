# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for heterogeneous per-neuron parameters in NIR -> FPGA

"""Heterogeneous per-neuron parameters lower to per-instance Verilog overrides."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("nir")

import nir

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.nir_bridge import compile_network_to_fpga, from_nir, from_scnetwork


def _compile_lif(taus, *, r=None, v_leak=None, module_name="m"):
    n = len(taus)
    r = np.ones(n) if r is None else np.asarray(r, dtype=float)
    v_leak = np.zeros(n) if v_leak is None else np.asarray(v_leak, dtype=float)
    graph = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(weight=np.full((n, 2), 0.5), bias=np.zeros(n)),
            "lif": nir.LIF(
                tau=np.asarray(taus, dtype=float),
                r=r,
                v_leak=v_leak,
                v_threshold=np.ones(n),
            ),
            "output": nir.Output(output_type={"output": np.array([n])}),
        },
        edges=[("input", "aff"), ("aff", "lif"), ("lif", "output")],
    )
    network = from_nir(graph, dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)
    return compile_network_to_fpga(neuron_graph, module_name=module_name)


def _override_instances(top: str) -> list[str]:
    return [line.strip() for line in top.splitlines() if "#(" in line and "_inst" in line]


def test_homogeneous_population_emits_no_parameter_overrides() -> None:
    result = _compile_lif([10.0, 10.0, 10.0])
    assert "#(" not in result.top_module  # identical RTL to the pre-feature output


def test_heterogeneous_tau_emits_correct_per_neuron_overrides() -> None:
    result = _compile_lif([10.0, 20.0, 30.0])
    q = Q88(data_width=16, fraction=8)

    overrides = _override_instances(result.top_module)
    # Neuron 0 matches the module default (tau=10) -> no override; 1 and 2 differ.
    assert len(overrides) == 2
    assert f".P_TAU(16'sd{q.encode(20.0)})" in result.top_module
    assert f".P_TAU(16'sd{q.encode(30.0)})" in result.top_module
    # The shared parameterised module is generated exactly once.
    assert list(result.neuron_modules) == ["lif"]
    assert result.total_neurons == 3


def test_heterogeneous_multiple_parameters_override_together() -> None:
    result = _compile_lif([10.0, 12.0], v_leak=[0.0, -1.0])
    q = Q88(data_width=16, fraction=8)

    # Neuron 1 differs in both tau and v_leak; both appear in its single override.
    assert f".P_TAU(16'sd{q.encode(12.0)})" in result.top_module
    assert f".P_V_LEAK(16'sd{q.encode(-1.0)})" in result.top_module
    inst1 = next(line for line in _override_instances(result.top_module) if "p0_n1_inst" in line)
    assert "P_TAU" in inst1 and "P_V_LEAK" in inst1


def test_overrides_reference_only_declared_module_parameters() -> None:
    result = _compile_lif([10.0, 20.0, 30.0])
    declared = set(
        re.findall(r"parameter signed \[\d+:0\] (P_[A-Z_]+)", result.neuron_modules["lif"])
    )
    used = set(re.findall(r"\.(P_[A-Z_]+)\(", result.top_module))
    assert used and used <= declared


def test_heterogeneous_population_compiles_on_aer_path() -> None:
    # >64 neurons routes through the AER interconnect; overrides must appear there too.
    taus = [10.0] * 70
    taus[5] = 25.0
    taus[40] = 33.0
    result = _compile_lif(taus)
    q = Q88(data_width=16, fraction=8)

    assert "Weighted address-event source vector" in result.top_module  # AER path taken
    assert f".P_TAU(16'sd{q.encode(25.0)})" in result.top_module
    assert f".P_TAU(16'sd{q.encode(33.0)})" in result.top_module
    assert len(_override_instances(result.top_module)) == 2


@pytest.mark.skipif(not shutil.which("iverilog"), reason="iverilog not installed")
def test_heterogeneous_rtl_synthesises_with_iverilog(tmp_path: Path) -> None:
    # Gold standard: the generated heterogeneous design is valid synthesisable
    # Verilog, parameter overrides and all.
    result = _compile_lif([10.0, 20.0, 30.0], v_leak=[0.0, -1.0, 0.5], module_name="het")

    files: list[str] = []

    def _write(name: str, source: str) -> None:
        path = tmp_path / name
        path.write_text(source, encoding="utf-8")
        files.append(str(path))

    _write("het.v", result.top_module)
    _write("weight_rom.v", result.weight_rom)
    for neuron_type, source in result.neuron_modules.items():
        _write(f"neuron_{neuron_type}.v", source)
    for module_name, source in result.scnir_source_modules.items():
        _write(f"source_{module_name}.v", source)

    proc = subprocess.run(
        ["iverilog", "-g2012", "-o", str(tmp_path / "het.out"), *files],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, f"iverilog failed:\n{proc.stderr}"
