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


def test_compile_to_datapath_without_threshold_uses_passthrough_state() -> None:
    """A non-spiking folded datapath drives no spike and passes through next state."""
    verilog = compile_to_datapath(_lif_without_threshold(), module_name="plain_pe")

    assert "assign spike_out = 1'b0;" in verilog
    assert "assign v_next_out = v_next;" in verilog
