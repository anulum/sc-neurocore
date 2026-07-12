# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Direct FPGA interconnect contract tests

"""Exercise direct-interconnect validation through the public compiler."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.nir_bridge import compile_network_to_fpga
from sc_neurocore.nir_bridge import fpga_compiler
from sc_neurocore.nir_bridge.fpga_direct_interconnect import build_direct_interconnect
from sc_neurocore.nir_bridge.neuron_graph import ConnectionSpec, NeuronGraph, NeuronSpec
from sc_neurocore.nir_bridge.quantise_params import QuantisedGraph


def _two_population_graph(connection: ConnectionSpec) -> NeuronGraph:
    """Build a two-population graph around one internal connection."""
    populations = [NeuronSpec("source", "lif", 2), NeuronSpec("target", "lif", 2)]
    return NeuronGraph(populations, [connection], "source", "target")


@pytest.mark.parametrize(
    ("connection_factory", "message"),
    [
        (
            lambda: ConnectionSpec("source", "target", np.ones((1, 2), dtype=np.float32)),
            "destination rows for 2 destination neurons",
        ),
        (
            lambda: ConnectionSpec("source", "target", np.ones((2, 1), dtype=np.float32)),
            "source columns for 2 source signals",
        ),
        (
            lambda: ConnectionSpec(
                "source",
                "target",
                np.ones((2, 2), dtype=np.float32),
                bias=np.ones(1, dtype=np.float32),
            ),
            "bias length does not match",
        ),
        (
            lambda: ConnectionSpec(
                "source",
                "target",
                np.ones((2, 2), dtype=np.float32),
                source_threshold=np.ones(1, dtype=np.float32),
            ),
            "source_threshold length",
        ),
        (
            lambda: ConnectionSpec(
                "source",
                "target",
                np.ones((2, 2), dtype=np.float32),
                destination_threshold=np.ones(1, dtype=np.float32),
            ),
            "destination_threshold length",
        ),
    ],
)
def test_rejects_inconsistent_direct_connection_contracts(
    connection_factory: Callable[[], ConnectionSpec],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compile_network_to_fpga(
            _two_population_graph(connection_factory()),
            interconnect="direct",
        )


def test_rejects_delay_on_external_source() -> None:
    target = NeuronSpec("target", "lif", 2)
    connection = ConnectionSpec(
        "stim",
        "target",
        np.ones((2, 2), dtype=np.float32),
        delay_steps=1,
    )
    graph = NeuronGraph([target], [connection], "stim", "target")

    with pytest.raises(ValueError, match="does not originate from a neuron population"):
        compile_network_to_fpga(graph, interconnect="direct")


def test_direct_interconnect_emits_delay_threshold_and_bias_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fpga_compiler, "_AER_THRESHOLD", 0)
    connection = ConnectionSpec(
        "source",
        "target",
        np.array([[0.5, 0.0], [-0.25, 0.125]], dtype=np.float32),
        bias=np.array([0.125, -0.125], dtype=np.float32),
        delay_steps=(0, 2),
        source_threshold=np.array([0.0, 0.5], dtype=np.float32),
        destination_threshold=np.array([0.25, 0.5], dtype=np.float32),
    )

    result = compile_network_to_fpga(
        _two_population_graph(connection),
    )

    assert result.interconnect == "direct"
    assert result.top_module.count("_threshold_out") == 4
    assert "p1_n0_c0_raw =" in result.top_module
    assert "p0_n1_spike_d2" in result.top_module
    assert "34'sh000000020 +" in result.top_module
    assert any("delayed recurrent connections" in warning for warning in result.warnings)
    assert any("NIR Threshold transforms" in warning for warning in result.warnings)


def test_unknown_interconnect_name_is_rejected() -> None:
    graph = NeuronGraph([NeuronSpec("layer", "lif", 1)], [], "layer", "layer")

    with pytest.raises(ValueError, match="unknown interconnect"):
        compile_network_to_fpga(graph, interconnect="crossbar")


def test_direct_builder_rejects_empty_graph_and_zero_width_literal() -> None:
    empty = QuantisedGraph([], [], Q88(), "", "", 1.0)
    with pytest.raises(ValueError, match="at least one neuron population"):
        build_direct_interconnect("empty", empty)

    population = NeuronSpec("target", "lif", 1)
    connection = ConnectionSpec("stim", "target", np.ones((1, 1), dtype=np.int64))
    connected = QuantisedGraph(
        [population],
        [connection],
        Q88(),
        "stim",
        "target",
        1.0,
        total_neurons=1,
        total_synapses=1,
    )
    with pytest.raises(ValueError, match="literal width must be positive"):
        build_direct_interconnect("zero_width", connected, data_width=0)
