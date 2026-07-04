# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FPGA synthesis-resource guard tests

"""``compile_network_to_fpga`` fails closed on IR that would exhaust synthesis resources.

The direct and AER interconnects instantiate one module per neuron, so an unbounded neuron
count is a synthesis-time denial of service; the folded interconnect shares one processing
element and is bounded by its state-RAM depth instead, so it is allowed a higher ceiling.
Data widths beyond a hardware-plausible fixed-point datapath are also rejected. Every guard
must raise before any RTL is emitted, so these tests need no HDL toolchain.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.nir_bridge import fpga_compiler
from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga
from sc_neurocore.nir_bridge.neuron_graph import ConnectionSpec, NeuronGraph, NeuronSpec


def _lif_graph(n_neurons: int) -> NeuronGraph:
    """A single connection-less LIF population of ``n_neurons`` neurons."""
    pop = NeuronSpec(name="pop0", neuron_type="lif", n_neurons=n_neurons, params={}, dt=1.0)
    return NeuronGraph(
        populations=[pop], connections=[], input_pop="pop0", output_pop="pop0", dt=1.0
    )


def _connected_graph(n_dst: int, n_src: int) -> NeuronGraph:
    """A population driven by one external connection with an ``n_dst x n_src`` weight ROM."""
    weights = np.ones((n_dst, n_src), dtype=np.float32)
    pop = NeuronSpec(name="pop0", neuron_type="lif", n_neurons=n_dst, params={}, dt=1.0)
    conn = ConnectionSpec(src="stim", dst="pop0", weights=weights)
    return NeuronGraph(
        populations=[pop], connections=[conn], input_pop="stim", output_pop="pop0", dt=1.0
    )


def test_rejects_data_width_above_ceiling() -> None:
    with pytest.raises(ValueError, match="outside the synthesisable range"):
        compile_network_to_fpga(_lif_graph(4), data_width=128, fraction=8)


def test_rejects_zero_data_width() -> None:
    with pytest.raises(ValueError, match="outside the synthesisable range"):
        compile_network_to_fpga(_lif_graph(4), data_width=0, fraction=0)


def test_rejects_fraction_not_below_data_width() -> None:
    # fraction == data_width leaves no integer or sign bit (negative integer bits).
    with pytest.raises(ValueError, match="0 <= fraction < data_width"):
        compile_network_to_fpga(_lif_graph(4), data_width=16, fraction=16)


def test_rejects_negative_fraction() -> None:
    with pytest.raises(ValueError, match="0 <= fraction < data_width"):
        compile_network_to_fpga(_lif_graph(4), data_width=16, fraction=-1)


def test_rejects_unrolled_neuron_count_over_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    # Patch the ceiling small so the guard is exercised without emitting thousands of modules.
    monkeypatch.setattr(fpga_compiler, "_MAX_UNROLLED_NEURONS", 3)
    with pytest.raises(ValueError, match="interconnect='folded'"):
        compile_network_to_fpga(_lif_graph(4), interconnect="direct")


def test_auto_interconnect_is_also_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    # interconnect=None auto-selects direct/AER (both unroll), so the per-neuron cap applies.
    monkeypatch.setattr(fpga_compiler, "_MAX_UNROLLED_NEURONS", 3)
    with pytest.raises(ValueError, match="per-neuron synthesis guard"):
        compile_network_to_fpga(_lif_graph(4), interconnect=None)


def test_folded_exceeds_the_unrolled_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    # The same net the direct/AER guard rejects is accepted by folded (below its own cap).
    monkeypatch.setattr(fpga_compiler, "_MAX_UNROLLED_NEURONS", 3)
    result = compile_network_to_fpga(_lif_graph(4), interconnect="folded")
    assert result.total_neurons == 4
    assert result.interconnect == "folded"
    assert result.top_module


def test_rejects_folded_neuron_count_over_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fpga_compiler, "_MAX_FOLDED_NEURONS", 3)
    with pytest.raises(ValueError, match="folded synthesis guard"):
        compile_network_to_fpga(_lif_graph(4), interconnect="folded")


def test_rejects_synapse_count_over_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fpga_compiler, "_MAX_SYNTHESISABLE_SYNAPSES", 3)
    # A 2 x 2 weight matrix is 4 synapses, over the patched cap.
    with pytest.raises(ValueError, match="weight-ROM synthesis guard"):
        compile_network_to_fpga(_connected_graph(2, 2), interconnect="direct")


def test_synapse_guard_is_interconnect_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    # The weight ROM is shared, so the synapse cap fires for folded too, not only direct/AER.
    monkeypatch.setattr(fpga_compiler, "_MAX_SYNTHESISABLE_SYNAPSES", 3)
    with pytest.raises(ValueError, match="weight-ROM synthesis guard"):
        compile_network_to_fpga(_connected_graph(2, 2), interconnect="folded")


def test_valid_small_network_passes_the_guard() -> None:
    # A normal small net compiles — the guard does not false-fire.
    result = compile_network_to_fpga(_lif_graph(4), data_width=16, fraction=8)
    assert result.total_neurons == 4
    assert result.top_module


def test_valid_connected_network_passes_the_guard() -> None:
    # A small connected net compiles — the synapse guard does not false-fire.
    result = compile_network_to_fpga(_connected_graph(3, 2), interconnect="direct")
    assert result.total_synapses == 6
    assert result.top_module
