#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# SC-NeuroCore -- NIR -> synthesisable FPGA Verilog
#
# Usage:
#   pip install sc-neurocore nir
#   python examples/nir_to_fpga_rtl.py [output_dir]

"""Compile a NIR graph all the way to synthesisable FPGA Verilog.

Every NIR backend in the wider ecosystem targets a simulator (snnTorch, Norse,
Sinabs, Rockpool, Nengo, Spyx, ...) or a neuromorphic chip (Loihi 2, SpiNNaker 2,
Speck, Xylo). SC-NeuroCore is the FPGA / stochastic-computing target: it lowers
the same vendor-neutral NIR graph into synthesisable RTL.

Because the input is plain NIR, the network can come from *any* NIR-emitting
framework. Train in snnTorch or Norse, export to NIR, and this example turns it
into Verilog you can hand to Yosys / Vivado.

Pipeline:

    NIR graph
      -> from_nir()             (NIR -> SCNetwork)
      -> from_scnetwork()       (SCNetwork -> hardware NeuronGraph)
      -> compile_network_to_fpga()
           - quantise to a fixed-point Q-format
           - one Verilog module per unique neuron type (canonical ODEs)
           - a weight ROM
           - a top-level interconnect (direct wiring, or AER routing for
             larger networks)
           - LFSR / Sobol stochastic-source modules

The result is real Verilog text, written to disk here so it can be inspected or
fed straight to an open-source synthesis flow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import nir
except ImportError as exc:  # pragma: no cover - example guard, not a unit under test
    raise ImportError("pip install nir") from exc

from sc_neurocore.nir_bridge import compile_network_to_fpga, from_nir, from_scnetwork


def build_two_layer_snn() -> nir.NIRGraph:
    """Build a small feed-forward spiking network as a NIR graph.

    Input(2) -> Affine(2->3) -> LIF(3) -> Affine(3->2) -> LIF(2) -> Output(2).

    This is the kind of graph any NIR-emitting framework produces; the weights
    here are fixed only so the example is deterministic.
    """
    rng = np.random.default_rng(0)
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "fc1": nir.Affine(weight=rng.normal(size=(3, 2)), bias=np.zeros(3)),
            "lif1": nir.LIF(
                tau=np.full(3, 10.0),
                r=np.ones(3),
                v_leak=np.zeros(3),
                v_threshold=np.ones(3),
            ),
            "fc2": nir.Affine(weight=rng.normal(size=(2, 3)), bias=np.zeros(2)),
            "lif2": nir.LIF(
                tau=np.full(2, 10.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[
            ("input", "fc1"),
            ("fc1", "lif1"),
            ("lif1", "fc2"),
            ("fc2", "lif2"),
            ("lif2", "output"),
        ],
    )


def compile_nir_to_rtl(graph: nir.NIRGraph, *, module_name: str = "snn_demo"):
    """Run the full NIR -> FPGA Verilog pipeline and return the artefacts."""
    network = from_nir(graph, dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)
    return compile_network_to_fpga(
        neuron_graph,
        module_name=module_name,
        data_width=16,
        fraction=8,
        source_kind="lfsr",
        target="artix7",
    )


def write_artefacts(result, out_dir: Path) -> list[Path]:
    """Write every Verilog artefact in the result to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _write(name: str, source: str) -> None:
        path = out_dir / name
        path.write_text(source, encoding="utf-8")
        written.append(path)

    _write(f"{result.module_name}_top.v", result.top_module)
    _write("weight_rom.v", result.weight_rom)
    for neuron_type, source in result.neuron_modules.items():
        _write(f"neuron_{neuron_type}.v", source)
    for module_name, source in result.scnir_source_modules.items():
        _write(f"source_{module_name}.v", source)
    return written


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/nir_fpga_demo")

    graph = build_two_layer_snn()
    result = compile_nir_to_rtl(graph)
    written = write_artefacts(result, out_dir)

    print("NIR -> FPGA Verilog")
    print(f"  module          : {result.module_name}")
    print(f"  neurons         : {result.total_neurons}")
    print(f"  synapses        : {result.total_synapses}")
    print(f"  fixed-point     : {result.q_format}")
    print(f"  neuron modules  : {sorted(result.neuron_modules)}")
    print(f"  source modules  : {sorted(result.scnir_source_modules)}")
    if result.warnings:
        print(f"  warnings        : {list(result.warnings)}")
    print(f"  wrote {len(written)} Verilog files to {out_dir}/")

    top_lines = result.top_module.splitlines()
    print("\n  top-module preview:")
    for line in top_lines[:12]:
        print(f"    {line}")


if __name__ == "__main__":
    main()
