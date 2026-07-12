# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FPGA connection-routing contract tests

"""Validate connection layouts and delays through the public FPGA compiler."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.nir_bridge import compile_network_to_fpga
from sc_neurocore.nir_bridge.neuron_graph import ConnectionSpec, NeuronGraph, NeuronSpec


def _graph(
    connections: list[ConnectionSpec],
    populations: list[NeuronSpec] | None = None,
) -> NeuronGraph:
    """Build a compiler-ready graph around the supplied connection contracts."""
    pops = populations or [NeuronSpec("dst", "lif", 2)]
    return NeuronGraph(
        populations=pops,
        connections=connections,
        input_pop="stim",
        output_pop=pops[-1].name,
    )


def test_rejects_non_matrix_weights_before_scnir_lowering() -> None:
    graph = _graph([ConnectionSpec("stim", "dst", np.ones(2, dtype=np.float32))])

    with pytest.raises(ValueError, match="weights must be a 2-D matrix"):
        compile_network_to_fpga(graph, interconnect="direct")


def test_rejects_connection_to_unknown_population() -> None:
    connection = ConnectionSpec("stim", "missing", np.ones((2, 1), dtype=np.float32))

    with pytest.raises(ValueError, match="is not a neuron population"):
        compile_network_to_fpga(_graph([connection]), interconnect="direct")


@pytest.mark.parametrize(
    ("delay_steps", "message"),
    [
        (-1, "must be non-negative"),
        (1025, "above the synthesis guard"),
        ((0,), "vector length 1"),
        ((0, -1), "must be non-negative"),
        ((0, 1025), "above the synthesis guard"),
    ],
)
def test_rejects_invalid_delay_vectors(
    delay_steps: int | tuple[int, ...],
    message: str,
) -> None:
    connection = ConnectionSpec(
        "stim",
        "dst",
        np.ones((2, 2), dtype=np.float32),
        delay_steps=delay_steps,
    )

    with pytest.raises(ValueError, match=message):
        compile_network_to_fpga(_graph([connection]), interconnect="direct")


def test_rejects_connections_without_source_columns() -> None:
    connection = ConnectionSpec("stim", "dst", np.ones((2, 0), dtype=np.float32))

    with pytest.raises(ValueError, match="no external source columns"):
        compile_network_to_fpga(_graph([connection]), interconnect="direct")


def test_rejects_zero_width_internal_population_source() -> None:
    populations = [NeuronSpec("source", "lif", 0), NeuronSpec("dst", "lif", 2)]
    connection = ConnectionSpec("source", "dst", np.ones((2, 0), dtype=np.float32))

    with pytest.raises(ValueError, match="source width must be positive"):
        compile_network_to_fpga(_graph([connection], populations), interconnect="direct")


def test_rejects_inconsistent_reused_external_source_width() -> None:
    populations = [NeuronSpec("left", "lif", 1), NeuronSpec("right", "lif", 1)]
    connections = [
        ConnectionSpec("stim", "left", np.ones((1, 1), dtype=np.float32)),
        ConnectionSpec("stim", "right", np.ones((1, 2), dtype=np.float32)),
    ]

    with pytest.raises(ValueError, match="inconsistent widths 1 and 2"):
        compile_network_to_fpga(_graph(connections, populations), interconnect="direct")


def test_external_input_manifest_preserves_first_seen_lane_order() -> None:
    populations = [NeuronSpec("left", "lif", 1), NeuronSpec("right", "lif", 1)]
    connections = [
        ConnectionSpec("beta", "left", np.ones((1, 2), dtype=np.float32)),
        ConnectionSpec("beta", "right", np.ones((1, 2), dtype=np.float32)),
        ConnectionSpec("alpha", "right", np.ones((1, 1), dtype=np.float32)),
    ]

    result = compile_network_to_fpga(_graph(connections, populations), interconnect="direct")

    assert [entry.as_dict() for entry in result.scnir_external_inputs] == [
        {"source": "beta", "offset": 0, "width": 2},
        {"source": "alpha", "offset": 2, "width": 1},
    ]
