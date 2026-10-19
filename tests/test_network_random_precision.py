# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — the network compiler lowers randn() in the requested precision on both interconnects

"""``compile_network_to_fpga(random_precision=...)`` reaches every neuron module and PE.

The exact ``randn()`` mode (:mod:`sc_neurocore.compiler.exact_normal`) was available
for a single neuron but not for a network. The option now reaches the per-type
modules of the direct and address-event interconnects and the folded PEs. A
two-neuron population ``v = randn()`` (map, seed 2026, streams 0 and 1) is
compiled at Q3.29, where the standard unit is many LSB off; each neuron's state in
the direct and in the folded co-simulation must equal the exact reference
``normal_code_exact`` of its own Philox draw at every step, and differ from the
standard reference somewhere. The standard mode is unchanged and an unknown mode
is refused. Without Icarus the co-simulation cases skip.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.compiler import exact_normal as xn
from sc_neurocore.compiler import random_units as ru
from sc_neurocore.compiler.verilog_compiler import compile_to_datapath, compile_to_verilog
from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga
from sc_neurocore.nir_bridge.fpga_neuron_rtl import _population_neuron
from sc_neurocore.nir_bridge.neuron_graph_contracts import ConnectionSpec, NeuronGraph, NeuronSpec
from tests.cosim_support import HAS_IVERILOG

_DW, _FR, _STEPS, _SEED = 32, 29, 12, 2026
_TEMPLATE = {
    "equations": ["dv/dt = randn()"],
    "threshold": "v > 1.0",
    "reset": "v = v",
    "method": "map",
    "default_params": {},
    "rng_seed": _SEED,
}


def _graph() -> NeuronGraph:
    """Return a stimulus feeding (with zero weight) two v = randn() neurons."""
    noise = NeuronSpec(name="src", neuron_type="noise", n_neurons=2, dt=1.0, template=_TEMPLATE)
    return NeuronGraph(
        populations=[noise],
        connections=[ConnectionSpec(src="stim", dst="src", weights=np.zeros((2, 1), np.float32))],
        input_pop="stim",
        output_pop="src",
        dt=1.0,
    )


def _reference(precision: str) -> list[list[int]]:
    """Return each step's codes of neurons 0 and 1 (streams 0 and 1) in ``precision``."""
    code = (
        xn.normal_code_exact
        if precision == "exact"
        else lambda w0, w1, dw, fr: ru.normal_code(w0, w1, dw, fr)
    )
    return [
        [code(*ru.draw(0, step, stream, _SEED)[:2], _DW, _FR) for stream in (0, 1)]
        for step in range(_STEPS)
    ]


@pytest.mark.parametrize("interconnect", ["direct", "folded"])
@pytest.mark.parametrize("precision", ["standard", "exact"])
def test_every_module_is_compiled_in_the_requested_precision(
    interconnect: str, precision: str
) -> None:
    """The neuron module (and the folded PE) equal the single-neuron compile in that mode."""
    result = compile_network_to_fpga(
        _graph(),
        module_name="net",
        data_width=_DW,
        fraction=_FR,
        interconnect=interconnect,
        random_precision=precision,
    )
    neuron = _population_neuron("noise", _graph().populations[0])
    assert result.neuron_modules["noise"] == compile_to_verilog(
        neuron, "sc_nir_noise", _DW, _FR, random_precision=precision
    )
    assert ("exact mode" in result.neuron_modules["noise"]) is (precision == "exact")
    if interconnect == "folded":
        pe = result.neuron_modules["noise_pe"]
        assert ("exact mode" in pe) is (precision == "exact")
        assert pe == compile_to_datapath(
            neuron,
            module_name="sc_nir_noise_pe",
            data_width=_DW,
            fraction=_FR,
            random_precision=precision,
        )


def test_an_unknown_precision_is_refused() -> None:
    """A mode other than standard or exact is refused before any RTL."""
    with pytest.raises(ValueError, match="random_precision"):
        compile_network_to_fpga(_graph(), random_precision="extra")


def _simulate(sources: dict[str, str], testbench: str) -> list[list[int]]:
    """Run Icarus on the sources and the testbench; return the integers of each D line."""
    with tempfile.TemporaryDirectory() as d:
        files = []
        for name, text in {**sources, "tb": testbench}.items():
            path = Path(d) / f"{name}.v"
            path.write_text(text)
            files.append(str(path))
        subprocess.run(
            ["iverilog", "-g2012", "-o", f"{d}/sim", *files],
            check=True,
            capture_output=True,
            timeout=300,
        )
        out = subprocess.run(
            ["vvp", f"{d}/sim"], capture_output=True, text=True, timeout=300
        ).stdout
    return [
        [int(x) for x in line.split()[1:]] for line in out.splitlines() if line.startswith("D ")
    ]


def _testbench(interconnect: str, cycles: int) -> str:
    """Return a testbench printing both neurons' states after every clock edge.

    The direct network steps every cycle. The folded sequencer writes each neuron's
    state word in its own cycle of a tick (the first neuron of the next tick in the
    cycle that commits the spikes), so its states are read as the sequence of
    values each word takes.
    """
    if interconnect == "direct":
        probe = "$signed(uut.p0_n0_v), $signed(uut.p0_n1_v)"
        extra = "wire [1:0] spike_bus;"
        ports = ".spike_bus(spike_bus)"
    else:
        probe = "$signed(uut.state_bram_0[0]), $signed(uut.state_bram_0[1])"
        extra = "wire [1:0] spike_bus; wire tick_done;"
        ports = ".spike_bus(spike_bus), .tick_done(tick_done)"
    return "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb;",
            f"reg clk; reg rst_n; reg en; {extra}",
            f"net uut (.clk(clk), .rst_n(rst_n), .en(en), .I_ext_flat({_DW}'d0), {ports});",
            "initial clk = 0; always #5 clk = ~clk;",
            "integer t;",
            "initial begin",
            "  rst_n = 0; en = 1; #23; rst_n = 1;",
            f"  for (t = 0; t < {cycles}; t = t + 1) begin @(posedge clk); #1;",
            f'    $display("D %0d %0d", {probe});',
            "  end",
            "  $finish;",
            "end",
            "endmodule",
        ]
    )


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
@pytest.mark.parametrize("interconnect", ["direct", "folded"])
def test_both_interconnects_draw_the_exact_reference(interconnect: str) -> None:
    """Every state word equals normal_code_exact of its neuron's draw; standard differs."""
    result = compile_network_to_fpga(
        _graph(),
        module_name="net",
        data_width=_DW,
        fraction=_FR,
        interconnect=interconnect,
        random_precision="exact",
    )
    sources = {**result.neuron_modules, "top": result.top_module}
    if interconnect == "direct":
        rows = _simulate(sources, _testbench("direct", _STEPS))
    else:
        samples = _simulate(sources, _testbench("folded", 8 * _STEPS))
        columns = [_successive_values([row[k] for row in samples])[:_STEPS] for k in (0, 1)]
        rows = [list(pair) for pair in zip(*columns)]
    assert rows == _reference("exact")
    assert rows != _reference("standard")


def _successive_values(column: list[int]) -> list[int]:
    """Return the values a state word takes after reset, each once, in order."""
    values: list[int] = []
    previous = 0  # the reset value
    for value in column:
        if value != previous:
            values.append(value)
            previous = value
    return values
