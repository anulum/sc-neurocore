# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Per-neuron FPGA RTL contract tests

"""Verify neuron-template and parameter contracts through FPGA compilation."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.nir_bridge import compile_network_to_fpga
from sc_neurocore.nir_bridge.neuron_graph import NeuronGraph, NeuronSpec


def _population_graph(*populations: NeuronSpec) -> NeuronGraph:
    """Build a connectionless graph for per-type RTL generation."""
    return NeuronGraph(
        populations=list(populations),
        connections=[],
        input_pop=populations[0].name,
        output_pop=populations[-1].name,
    )


def test_unknown_neuron_template_fails_closed() -> None:
    graph = _population_graph(NeuronSpec("mystery", "not_a_neuron", 1))

    with pytest.raises(ValueError, match="No ODE template"):
        compile_network_to_fpga(graph, interconnect="direct")


def test_unknown_template_parameter_fails_closed() -> None:
    population = NeuronSpec(
        "layer",
        "lif",
        1,
        params={"not_a_parameter": np.array([1.0], dtype=np.float64)},
    )

    with pytest.raises(ValueError, match="not supported by the 'lif' FPGA template"):
        compile_network_to_fpga(_population_graph(population), interconnect="direct")


def test_empty_parameter_array_is_rejected() -> None:
    population = NeuronSpec(
        "layer",
        "lif",
        1,
        params={"tau": np.array([], dtype=np.float64)},
    )

    with pytest.raises(ValueError, match="Empty parameter array"):
        compile_network_to_fpga(_population_graph(population), interconnect="direct")


def test_same_type_with_different_dynamics_requires_distinct_modules() -> None:
    graph = _population_graph(
        NeuronSpec("fast", "lif", 1, dt=0.5),
        NeuronSpec("slow", "lif", 1, dt=1.0),
    )

    with pytest.raises(ValueError, match="different parameters across populations"):
        compile_network_to_fpga(graph, interconnect="direct")


def test_shared_scalar_parameter_emits_one_module_without_override() -> None:
    population = NeuronSpec(
        "layer",
        "lif",
        2,
        params={"tau": np.array([20.0], dtype=np.float64)},
    )

    result = compile_network_to_fpga(_population_graph(population), interconnect="direct")

    assert list(result.neuron_modules) == ["lif"]
    assert "parameter signed [15:0] P_TAU" in result.neuron_modules["lif"]
    assert "#(" not in result.top_module


def test_explicit_default_on_later_population_needs_no_override() -> None:
    graph = _population_graph(
        NeuronSpec("implicit", "lif", 1),
        NeuronSpec(
            "explicit",
            "lif",
            1,
            params={"tau": np.array([20.0], dtype=np.float64)},
        ),
    )

    result = compile_network_to_fpga(graph, interconnect="direct")

    assert ".P_TAU(" not in result.top_module
