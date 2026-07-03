# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for folded-interconnect rejection of heterogeneous parameters

"""The folded interconnect must refuse a heterogeneous population, not fold it incorrectly.

The shared per-type processing element bakes one representative parameter set (there
is no per-neuron parameter RAM), so a population whose neurons carry different
quantised parameters cannot fold — the fold would apply the first neuron's parameters
to the whole population, diverging from the direct path, which emits a per-neuron
``#(.P_X(...))`` override. These tests pin that ``interconnect="folded"`` raises a
clear error for a heterogeneous population while the direct path keeps emitting the
correct per-neuron overrides, and that a homogeneous population still folds.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("nir")

import nir

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.nir_bridge import (
    compile_network_to_fpga,
    from_nir,
    from_scnetwork,
    quantise_graph,
)
from sc_neurocore.nir_bridge.fpga_compiler import (
    _can_fold,
    _population_params_are_uniform,
)

_Q = Q88(data_width=16, fraction=8)


def _neuron_graph(taus, *, v_leak=None):
    """Build a quantisable LIF network with the given per-neuron ``tau`` values."""
    n = len(taus)
    v_leak = np.zeros(n) if v_leak is None else np.asarray(v_leak, dtype=float)
    graph = nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(weight=np.full((n, 2), 0.5), bias=np.zeros(n)),
            "lif": nir.LIF(
                tau=np.asarray(taus, dtype=float),
                r=np.ones(n),
                v_leak=v_leak,
                v_threshold=np.ones(n),
            ),
            "output": nir.Output(output_type={"output": np.array([n])}),
        },
        edges=[("input", "aff"), ("aff", "lif"), ("lif", "output")],
    )
    return from_scnetwork(from_nir(graph, dt=1.0), dt=1.0)


def _qgraph(taus, *, v_leak=None):
    return quantise_graph(_neuron_graph(taus, v_leak=v_leak), _Q)


def _compile(taus, interconnect, *, v_leak=None):
    return compile_network_to_fpga(
        _neuron_graph(taus, v_leak=v_leak), module_name="m", interconnect=interconnect
    )


class TestPopulationParamsAreUniform:
    """The per-population parameter-uniformity predicate."""

    def test_homogeneous_population_is_uniform(self) -> None:
        pop = _qgraph([10.0, 10.0, 10.0]).populations[0]
        assert _population_params_are_uniform(pop, 16) is True

    def test_heterogeneous_tau_is_not_uniform(self) -> None:
        pop = _qgraph([10.0, 20.0, 30.0]).populations[0]
        assert _population_params_are_uniform(pop, 16) is False

    def test_heterogeneity_in_a_second_parameter_is_detected(self) -> None:
        # tau uniform but v_leak differs — still heterogeneous.
        pop = _qgraph([10.0, 10.0], v_leak=[0.0, -1.0]).populations[0]
        assert _population_params_are_uniform(pop, 16) is False

    def test_values_that_quantise_equal_count_as_uniform(self) -> None:
        # Two floats a fraction apart round to the same Q8.8 literal → uniform.
        step = 1.0 / (1 << 8)
        pop = _qgraph([10.0, 10.0 + step / 4.0]).populations[0]
        assert _population_params_are_uniform(pop, 16) is True

    def test_scalar_shared_parameter_is_uniform(self) -> None:
        # A parameter stored as a single shared value (a length-1 array, not a
        # per-neuron array) is uniform by construction regardless of population size.
        pop = _qgraph([10.0, 10.0, 10.0]).populations[0]
        first_key = next(iter(pop.params))
        shared = dict(pop.params)
        shared[first_key] = np.asarray([np.asarray(shared[first_key]).reshape(-1)[0]])

        class _SharedParamPopulation:
            n_neurons = pop.n_neurons
            params = shared

        assert _population_params_are_uniform(_SharedParamPopulation(), 16) is True


class TestCanFoldHeterogeneity:
    """``_can_fold`` gates on parameter uniformity."""

    def test_homogeneous_graph_folds(self) -> None:
        assert _can_fold(_qgraph([10.0, 10.0, 10.0]), data_width=16) is True

    def test_heterogeneous_graph_does_not_fold(self) -> None:
        assert _can_fold(_qgraph([10.0, 20.0, 30.0]), data_width=16) is False


class TestFoldedSelectionRejectsHeterogeneous:
    """The public compiler entry point refuses to fold a heterogeneous population wrongly."""

    def test_folded_raises_for_heterogeneous_population(self) -> None:
        with pytest.raises(ValueError, match="uniform|heterogeneous"):
            _compile([10.0, 20.0, 30.0], "folded")

    def test_folded_error_points_to_the_direct_path(self) -> None:
        with pytest.raises(ValueError, match="direct"):
            _compile([10.0, 20.0, 30.0], "folded")

    def test_homogeneous_population_still_folds(self) -> None:
        result = _compile([10.0, 10.0, 10.0], "folded")
        assert result.interconnect == "folded"
        assert result.folded_metrics is not None

    def test_direct_path_still_emits_per_neuron_overrides(self) -> None:
        # The correct fallback: the direct interconnect reproduces heterogeneity
        # exactly, so refusing to fold loses no capability.
        result = _compile([10.0, 20.0, 30.0], "direct")
        assert f"16'sd{_Q.encode(20.0)}" in result.top_module
        assert f"16'sd{_Q.encode(30.0)}" in result.top_module
        overrides = [ln for ln in result.top_module.splitlines() if "#(" in ln and "_inst" in ln]
        assert len(overrides) == 2

    def test_auto_path_is_unaffected_by_heterogeneity(self) -> None:
        # ``auto`` never selects folded, so a heterogeneous population compiles as
        # before (direct here, well under the AER threshold).
        result = _compile([10.0, 20.0, 30.0], None)
        assert result.interconnect == "direct"
        assert f"16'sd{_Q.encode(30.0)}" in result.top_module
