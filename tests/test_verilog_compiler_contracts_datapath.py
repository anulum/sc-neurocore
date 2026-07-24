# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (datapath) from former test_verilog_compiler_contracts.py

from __future__ import annotations

from tests.verilog_compiler_contracts_support import *  # noqa: F403


def test_compile_to_datapath_resets_from_the_same_candidate_expression() -> None:
    """The folded PE must expose the same candidate-based post-reset next state."""
    verilog = compile_to_datapath(_candidate_reset_neuron(), module_name="sc_candidate_reset_pe")

    assert "assign a_next_out = spike_out ? ((a_next + P_KICK)) : a_next;" in verilog
    assert "assign v_next_out = spike_out ? (P_V_RESET) : v_next;" in verilog


def test_compile_to_datapath_rejects_unrepresentable_timestep() -> None:
    """The folded datapath applies the same fixed-point timestep guard."""
    neuron = _lif_without_threshold(dt=0.001)

    with pytest.raises(ValueError, match="underflows in Q8.8"):
        compile_to_datapath(neuron)


def test_compile_to_datapath_accepts_zero_timestep() -> None:
    """A zero timestep remains a legal frozen-state folded datapath."""
    verilog = compile_to_datapath(_lif_without_threshold(dt=0.0), module_name="frozen_pe")

    assert "module frozen_pe" in verilog
    assert "_dt_mul_v" in verilog


def test_compile_to_datapath_rejects_unknown_parameter_ports() -> None:
    """Folded parameter ports must name real neuron parameters."""
    neuron = _lif_without_threshold()

    with pytest.raises(ValueError, match="param_ports names are not parameters"):
        compile_to_datapath(neuron, param_ports=("not_a_parameter",))


def test_compile_to_datapath_validates_parameter_port_sequence() -> None:
    """Bare text, non-string entries, and duplicate parameter ports fail closed."""
    neuron = _lif_without_threshold()

    with pytest.raises(TypeError, match="not text"):
        compile_to_datapath(neuron, param_ports="tau")
    with pytest.raises(TypeError, match="entries must all be strings"):
        compile_to_datapath(neuron, param_ports=[cast(str, 1)])
    with pytest.raises(ValueError, match="must not contain duplicates"):
        compile_to_datapath(neuron, param_ports=("tau", "tau"))


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
