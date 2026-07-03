# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for folded-interconnect heterogeneous per-neuron parameters

"""The folded interconnect streams heterogeneous per-neuron parameters from a parameter ROM.

A population whose neurons carry different quantised parameters cannot bake one shared
value into the per-type processing element, so the folded interconnect exposes those
parameters as PE input ports and drives them from a per-neuron ``case(nidx)`` ROM — the
parameter-space analogue of the state BRAM. These tests pin the parameter-uniformity
predicate, the ``_can_fold`` gate, the generated ROM and PE ports, the resource metric,
and — critically — that the PE bakes the *real* parameter value (a regression guard for
the double-encoding that baked ``tau = 0`` into the folded PE for every real graph).
The bit-exact co-simulation against the direct path lives in ``test_folded_interconnect``.
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
    _dequantised_pop,
    _heterogeneous_param_names,
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


class TestHeterogeneousParamNames:
    """Which parameters vary per neuron."""

    def test_homogeneous_population_has_no_heterogeneous_params(self) -> None:
        assert _heterogeneous_param_names(_qgraph([10.0, 10.0, 10.0]).populations[0], 16) == []

    def test_varying_tau_is_reported(self) -> None:
        assert _heterogeneous_param_names(_qgraph([10.0, 20.0, 30.0]).populations[0], 16) == ["tau"]

    def test_multiple_varying_parameters_are_sorted(self) -> None:
        pop = _qgraph([10.0, 20.0], v_leak=[0.0, -1.0]).populations[0]
        assert _heterogeneous_param_names(pop, 16) == ["tau", "v_leak"]

    def test_values_that_quantise_equal_are_not_heterogeneous(self) -> None:
        step = 1.0 / (1 << 8)
        pop = _qgraph([10.0, 10.0 + step / 4.0]).populations[0]
        assert _heterogeneous_param_names(pop, 16) == []


class TestPopulationParamsAreUniform:
    """The per-population parameter-uniformity predicate."""

    def test_homogeneous_population_is_uniform(self) -> None:
        assert (
            _population_params_are_uniform(_qgraph([10.0, 10.0, 10.0]).populations[0], 16) is True
        )

    def test_heterogeneous_tau_is_not_uniform(self) -> None:
        assert (
            _population_params_are_uniform(_qgraph([10.0, 20.0, 30.0]).populations[0], 16) is False
        )


class TestDequantisePop:
    """De-quantising a population's parameters for real-valued PE compilation."""

    def test_rescales_quantised_integers_back_to_real_values(self) -> None:
        pop = _qgraph([10.0, 20.0, 30.0]).populations[0]
        # pop.params['tau'] holds q.encode(10/20/30) = 2560/5120/7680; de-quantising divides
        # by 2**fraction, recovering 10/20/30 (which re-encode losslessly).
        rescaled = _dequantised_pop(pop, 8)
        np.testing.assert_allclose(
            np.asarray(rescaled.params["tau"]).reshape(-1), [10.0, 20.0, 30.0]
        )
        # Round-trips: encoding the rescaled value returns the original quantised integer.
        assert [_Q.encode(v) for v in [10.0, 20.0, 30.0]] == [2560, 5120, 7680]

    def test_empty_params_returns_population_unchanged(self) -> None:
        pop = _qgraph([10.0, 10.0]).populations[0]
        pop.params.clear()
        assert _dequantised_pop(pop, 8) is pop


class TestCanFoldHeterogeneity:
    """``_can_fold`` accepts heterogeneous datapath parameters."""

    def test_homogeneous_graph_folds(self) -> None:
        assert _can_fold(_qgraph([10.0, 10.0, 10.0]), data_width=16) is True

    def test_heterogeneous_graph_folds(self) -> None:
        assert _can_fold(_qgraph([10.0, 20.0, 30.0]), data_width=16) is True


class TestFoldedHeterogeneousParams:
    """The folded interconnect streams per-neuron parameters through a ROM."""

    def test_heterogeneous_population_folds(self) -> None:
        result = _compile([10.0, 20.0, 30.0], "folded")
        assert result.interconnect == "folded"

    def test_folded_pe_exposes_the_varying_parameter_as_a_port(self) -> None:
        result = _compile([10.0, 20.0, 30.0], "folded")
        assert any("input wire signed [15:0] P_TAU" in m for m in result.neuron_modules.values())
        assert ".P_TAU(param_tau_lif)" in result.top_module

    def test_folded_emits_a_per_neuron_parameter_rom(self) -> None:
        result = _compile([10.0, 20.0, 30.0], "folded")
        top = result.top_module
        # The ROM carries each neuron's own quantised tau (2560/5120/7680), addressed by nidx.
        assert f"= {16}'sd{_Q.encode(20.0)}" in top
        assert f"= {16}'sd{_Q.encode(30.0)}" in top
        assert "case (nidx)" in top

    def test_folded_reports_parameter_rom_bits(self) -> None:
        result = _compile([10.0, 20.0, 30.0], "folded")
        assert result.folded_metrics is not None
        # 3 neurons × 1 varying parameter × 16 bits.
        assert result.folded_metrics.param_rom_bits == 3 * 1 * 16
        assert result.folded_metrics.as_dict()["param_rom_bits"] == 48

    def test_homogeneous_population_has_no_parameter_rom(self) -> None:
        result = _compile([10.0, 10.0, 10.0], "folded")
        assert "param_tau" not in result.top_module
        assert result.folded_metrics is not None
        assert result.folded_metrics.param_rom_bits == 0

    def test_folded_pe_bakes_the_real_parameter_not_a_double_encoded_zero(self) -> None:
        # Regression: the folded PE built the per-type neuron from the *quantised* population,
        # so Q88.encode ran twice and baked tau = 5120 × 256 mod 2**16 = 0 into the shared PE
        # for every real graph. A uniform explicit tau must bake q.encode(20) = 5120, not 0.
        result = _compile([20.0, 20.0, 20.0], "folded")
        pe = next(src for key, src in result.neuron_modules.items() if key.endswith("_pe"))
        assert f"P_TAU = 16'sd{_Q.encode(20.0)}" in pe
        assert "P_TAU = 16'sd0" not in pe
