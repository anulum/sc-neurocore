# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Address-event FPGA interconnect tests

"""Exercise weighted address-event lowering through public compilation."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.nir_bridge import fpga_compiler
from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga
from sc_neurocore.nir_bridge.neuron_graph import ConnectionSpec, NeuronGraph, NeuronSpec


@pytest.fixture
def force_aer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select AER for every non-delayed, non-thresholded graph in this test."""
    monkeypatch.setattr(fpga_compiler, "_AER_THRESHOLD", 0)


def test_analogue_only_network_emits_empty_event_vector(force_aer: None) -> None:
    graph = NeuronGraph([NeuronSpec("state", "li", 1)], [], "state", "state")

    result = compile_network_to_fpga(graph)

    assert result.interconnect == "aer"
    assert "aer_event_valid = 1'b0" in result.top_module
    assert "aer_addr = 1'b0" in result.top_module


def test_external_weighted_aer_emits_bias_mac_and_skips_zero_weight(force_aer: None) -> None:
    target = NeuronSpec("target", "lif", 2)
    connection = ConnectionSpec(
        "stim",
        "target",
        np.array([[1.0, 0.0], [0.5, -0.25]], dtype=np.float32),
        bias=np.array([0.125, -0.125], dtype=np.float32),
    )
    graph = NeuronGraph([target], [connection], "stim", "target")

    result = compile_network_to_fpga(graph)

    assert result.interconnect == "aer"
    assert "Direct analogue multiply-accumulate terms" in result.top_module
    assert "ext_input_0" in result.top_module
    assert "ext_input_1" in result.top_module
    assert "p0_n0_c0_s1_mul" not in result.top_module


def test_aer_routes_spikes_and_analogue_population_voltage(force_aer: None) -> None:
    populations = [
        NeuronSpec("events", "lif", 1),
        NeuronSpec("state", "li", 1),
        NeuronSpec("target", "lif", 1),
    ]
    connections = [
        ConnectionSpec("events", "target", np.array([[0.5]], dtype=np.float32)),
        ConnectionSpec("state", "target", np.array([[0.25]], dtype=np.float32)),
    ]
    graph = NeuronGraph(populations, connections, "events", "target")

    result = compile_network_to_fpga(graph)

    assert result.interconnect == "aer"
    assert "if (p0_n0_spike)" in result.top_module
    assert "p1_n0_v *" in result.top_module
    assert "aer_event_valid" in result.top_module
