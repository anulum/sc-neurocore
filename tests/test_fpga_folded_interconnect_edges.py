# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Folded FPGA interconnect edge contracts

"""Exercise folded eligibility and mixed-population parameter routing."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.nir_bridge import compile_network_to_fpga
from sc_neurocore.nir_bridge.fpga_folded_interconnect import (
    build_folded_interconnect,
    can_fold,
)
from sc_neurocore.nir_bridge.neuron_graph import ConnectionSpec, NeuronGraph, NeuronSpec
from sc_neurocore.nir_bridge.quantise_params import QuantisedGraph

_Q = Q88(data_width=16, fraction=8)


def test_empty_unsupported_and_missing_destination_graphs_do_not_fold() -> None:
    empty = QuantisedGraph([], [], _Q, "", "", 1.0)
    assert can_fold(empty, data_width=16) is False

    unsupported = QuantisedGraph(
        [NeuronSpec("unknown", "not_a_neuron", 1)],
        [],
        _Q,
        "unknown",
        "unknown",
        1.0,
    )
    assert can_fold(unsupported, data_width=16) is False

    population = NeuronSpec("known", "lif", 1)
    missing_destination = QuantisedGraph(
        [population],
        [ConnectionSpec("stim", "missing", np.ones((1, 1), dtype=np.int64))],
        _Q,
        "stim",
        "known",
        1.0,
    )
    assert can_fold(missing_destination, data_width=16) is False
    with pytest.raises(ValueError, match="outside the folded interconnect"):
        build_folded_interconnect("empty", empty)


def test_same_type_uniform_population_uses_shared_heterogeneous_parameter_port() -> None:
    graph = NeuronGraph(
        populations=[
            NeuronSpec(
                "heterogeneous",
                "lif",
                2,
                {"tau": np.array([10.0, 20.0], dtype=np.float64)},
            ),
            NeuronSpec(
                "uniform",
                "lif",
                2,
                {"tau": np.array([10.0], dtype=np.float64)},
            ),
        ],
        connections=[],
        input_pop="heterogeneous",
        output_pop="uniform",
    )

    result = compile_network_to_fpga(graph, interconnect="folded")

    assert "wire signed [DATA_WIDTH - 1:0] param_tau_1" in result.top_module
    assert "pidx == 1'd1 ? param_tau_1" in result.top_module


def test_non_first_connectionless_population_receives_zero_current() -> None:
    graph = NeuronGraph(
        populations=[NeuronSpec("first", "lif", 1), NeuronSpec("second", "if", 1)],
        connections=[],
        input_pop="first",
        output_pop="second",
    )

    result = compile_network_to_fpga(graph, interconnect="folded")

    assert "wire signed [DATA_WIDTH - 1:0] cur_I_1 = 16'sd0" in result.top_module
