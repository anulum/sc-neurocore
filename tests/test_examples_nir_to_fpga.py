# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Smoke test for the NIR -> FPGA Verilog example

"""Keep examples/nir_to_fpga_rtl.py runnable: build a NIR graph and lower it to RTL."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("nir")

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "examples" / "nir_to_fpga_rtl.py"


def _load_example() -> ModuleType:
    spec = importlib.util.spec_from_file_location("nir_to_fpga_rtl_example", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_builds_a_nir_graph() -> None:
    import nir

    graph = _load_example().build_two_layer_snn()
    assert isinstance(graph, nir.NIRGraph)
    assert {"input", "fc1", "lif1", "fc2", "lif2", "output"} <= set(graph.nodes)


def test_example_lowers_nir_to_synthesisable_verilog() -> None:
    example = _load_example()
    result = example.compile_nir_to_rtl(example.build_two_layer_snn(), module_name="snn_demo")

    assert result.module_name == "snn_demo"
    assert result.total_neurons == 5
    assert result.total_synapses == 12
    assert "lif" in result.neuron_modules
    assert "module snn_demo" in result.top_module
    assert "module" in result.weight_rom
    # Each spiking population gets a stochastic source module.
    assert result.scnir_source_modules


def test_example_writes_verilog_artefacts(tmp_path: Path) -> None:
    example = _load_example()
    result = example.compile_nir_to_rtl(example.build_two_layer_snn())

    written = example.write_artefacts(result, tmp_path)

    assert (tmp_path / f"{result.module_name}_top.v") in written
    assert (tmp_path / "weight_rom.v").exists()
    assert all(path.suffix == ".v" and path.read_text() for path in written)
