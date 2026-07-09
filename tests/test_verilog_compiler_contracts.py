# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog compiler contract tests

"""Focused contracts for the equation-to-Verilog compiler edge cases."""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.verilog_compiler import compile_to_datapath, compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron, from_equations


def _lif_without_threshold(dt: float = 0.01) -> EquationNeuron:
    """Build a one-state equation neuron without threshold or reset rules."""
    return from_equations(
        "dv/dt = -v/tau + I",
        params={"tau": 10.0},
        init={"v": 0.0},
        dt=dt,
    )


def test_compile_to_verilog_rejects_unknown_overflow_mode() -> None:
    """The registered compiler rejects unsupported overflow policies."""
    neuron = _lif_without_threshold()

    with pytest.raises(ValueError, match="Unknown overflow mode"):
        compile_to_verilog(neuron, overflow="explode")


def test_compile_to_verilog_lowers_map_method_and_piecewise_ifexp() -> None:
    """A discrete-map schema with a piecewise IfExp fast branch lowers to Verilog."""
    neuron = EquationNeuron(
        equations={
            "x": "(alpha / (1.0 - x) + y) if x <= 0 else (alpha + y if x < alpha + y else -1.0)",
            "y": "y - mu * (x + 1.0)",
        },
        parameters={"alpha": 4.0, "mu": 0.001},
        state={"x": -1.0, "y": -3.0},
        threshold="x > 0.0",
        dt=1.0,
        method="map",
    )

    verilog = compile_to_verilog(neuron, module_name="sc_map_ifexp")

    assert "module sc_map_ifexp" in verilog
    assert "?" in verilog  # the IfExp branch lowers to a Verilog ternary select
    # Discrete-map path: the increment is `f(state) - state`, so no forward-Euler
    # `_dt_mul_` derivative-scaling wire is emitted.
    assert "_dt_mul_" not in verilog


def test_compile_to_datapath_rejects_unrepresentable_timestep() -> None:
    """The folded datapath applies the same fixed-point timestep guard."""
    neuron = _lif_without_threshold(dt=0.001)

    with pytest.raises(ValueError, match="underflows in Q8.8"):
        compile_to_datapath(neuron)


def test_compile_to_datapath_rejects_unknown_parameter_ports() -> None:
    """Folded parameter ports must name real neuron parameters."""
    neuron = _lif_without_threshold()

    with pytest.raises(ValueError, match="param_ports names are not parameters"):
        compile_to_datapath(neuron, param_ports=("not_a_parameter",))


def test_compile_to_datapath_carries_named_parameters_on_ports() -> None:
    """A valid ``param_ports`` name is exposed on an input port, not baked as a default.

    A folded population with heterogeneous parameters streams each neuron's value
    through this port from a per-neuron ROM; the same ``P_<NAME>`` identifier is used
    whether baked or ported, so only the declaration moves.
    """
    verilog = compile_to_datapath(
        _lif_without_threshold(), module_name="hetero_pe", param_ports=("tau",)
    )

    assert "input wire signed [15:0] P_TAU," in verilog
    assert "parameter signed [15:0] P_TAU" not in verilog


def test_compile_to_datapath_without_threshold_uses_passthrough_state() -> None:
    """A non-spiking folded datapath drives no spike and passes through next state."""
    verilog = compile_to_datapath(_lif_without_threshold(), module_name="plain_pe")

    assert "assign spike_out = 1'b0;" in verilog
    assert "assign v_next_out = v_next;" in verilog
