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


def _candidate_reset_neuron() -> EquationNeuron:
    """Build a two-state RK4 neuron whose adaptive reset reads the candidate state."""
    return EquationNeuron(
        equations={"v": "I", "a": "-a / tau"},
        parameters={"tau": 10.0, "kick": 2.0, "v_reset": 0.0},
        state={"v": 0.0, "a": 1.0},
        threshold="v >= 1.0",
        reset={"v": "v_reset", "a": "a + kick"},
        dt=1.0,
        method="rk4",
    )


def test_compile_to_verilog_rejects_unknown_overflow_mode() -> None:
    """The registered compiler rejects unsupported overflow policies."""
    neuron = _lif_without_threshold()

    with pytest.raises(ValueError, match="Unknown overflow mode"):
        compile_to_verilog(neuron, overflow="explode")


def test_compile_to_verilog_resets_from_candidate_and_exposes_post_reset_state() -> None:
    """Registered RTL must match EquationNeuron's integrate-detect-reset sequence.

    The adaptive ``a = a + kick`` reset reads ``a_next``, not the pre-step
    ``a_reg``. Both the internal register and public output take that same reset
    value on the spike cycle; a constant voltage reset follows the same public
    post-reset contract.
    """
    verilog = compile_to_verilog(_candidate_reset_neuron(), module_name="sc_candidate_reset")

    assert "a_reg <= (a_next + P_KICK);" in verilog
    assert "a_out <= (a_next + P_KICK);" in verilog
    assert "a_reg <= (a_reg + P_KICK);" not in verilog
    assert "v_reg <= P_V_RESET;" in verilog
    assert "v_out <= P_V_RESET;" in verilog


def test_compile_to_datapath_resets_from_the_same_candidate_expression() -> None:
    """The folded PE must expose the same candidate-based post-reset next state."""
    verilog = compile_to_datapath(_candidate_reset_neuron(), module_name="sc_candidate_reset_pe")

    assert "assign a_next_out = spike_out ? ((a_next + P_KICK)) : a_next;" in verilog
    assert "assign v_next_out = spike_out ? (P_V_RESET) : v_next;" in verilog


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
    # Discrete-map path: the next state is `f(state)` saturated directly, so there is
    # no forward-Euler `_dt_mul_` scaling and no `state_reg + d<var>` increment-add
    # (adding the current state would risk a full-scale `f(state) - state` overflow).
    assert "_dt_mul_" not in verilog
    assert "_reg + d" not in verilog


def test_compile_to_verilog_gauss_seidel_reads_committed_earlier_variable() -> None:
    """Sequential (Gauss-Seidel) lowering reads the freshly-committed earlier variable.

    With state declared ``a`` then ``b`` and ``db/dt`` referencing ``a``, the ``gauss_seidel``
    method emits ``a_next`` inside ``b``'s derivative — the datapath consumes the updated
    ``a`` within the same sub-step — whereas the simultaneous ``euler`` method reads only the
    pre-step ``a_reg`` there. The extra ``a_next`` reference (and the correspondingly fewer
    ``a_reg`` references) in the sequential Verilog pins the commit-before-read ordering, not
    merely that the branch executed.
    """
    equations = {"a": "-a / tau_a", "b": "a - b / tau_b"}
    params = {"tau_a": 10.0, "tau_b": 20.0}
    state = {"a": 1.0, "b": 0.0}

    sequential = compile_to_verilog(
        EquationNeuron(
            equations=equations,
            parameters=params,
            state=dict(state),
            threshold="b > 100.0",
            dt=0.1,
            method="gauss_seidel",
        ),
        module_name="sc_gs_seq",
    )
    simultaneous = compile_to_verilog(
        EquationNeuron(
            equations=equations,
            parameters=params,
            state=dict(state),
            threshold="b > 100.0",
            dt=0.1,
            method="euler",
        ),
        module_name="sc_gs_sim",
    )

    assert "module sc_gs_seq" in sequential
    # ``b``'s derivative reads the committed ``a_next`` under Gauss-Seidel but not under Euler
    # (``a_next`` still appears in the simultaneous Verilog, but only for the state commit).
    assert "a_next" in simultaneous
    assert sequential.count("a_next") > simultaneous.count("a_next")
    assert sequential.count("a_reg") < simultaneous.count("a_reg")


def test_compile_to_verilog_gauss_seidel_pipelined_registers_multiplies() -> None:
    """Sequential-mode compilation with pipelining registers the derivative multiplies.

    ``pipeline_stages > 0`` routes the sub-step dt-scaling multiply through a pipeline
    register (``_dt_mul_<var>_r``) exactly as the Euler path does, exercising the registered
    branch of the sequential emitter's dt-scaling.
    """
    neuron = EquationNeuron(
        equations={"a": "-a / tau_a", "b": "a - b / tau_b"},
        parameters={"tau_a": 10.0, "tau_b": 20.0},
        state={"a": 1.0, "b": 0.0},
        threshold="b > 100.0",
        dt=0.1,
        method="gauss_seidel",
    )

    verilog = compile_to_verilog(neuron, module_name="sc_gs_pipe", pipeline_stages=1)

    assert "module sc_gs_pipe" in verilog
    assert "_dt_mul_a_r" in verilog  # the dt-scaling multiply for ``a`` is registered


def test_compile_to_verilog_rejects_substeps_with_pipelining() -> None:
    """Macro-step sub-stepping and multiply pipelining cannot be combined.

    ``substeps > 1`` folds several integration sub-steps into one macro step whose spike
    decision is taken on the macro boundary; multiply pipelining instead holds the state
    steady across fill cycles, so the two stepping schemes would disagree with the golden. The
    compiler rejects the combination rather than emit a datapath that silently drifts. The
    guard fires only after the edge-crossing check (a resetting model would raise there first),
    so this uses a non-resetting crossing oscillator to reach the pipelining guard.
    """
    neuron = EquationNeuron(
        equations={"v": "v - v * v * v / 3.0 - w + I", "w": "epsilon * (v + a - b * w)"},
        parameters={"a": 0.7, "b": 0.8, "epsilon": 0.08, "v_threshold": 1.0},
        state={"v": -1.0, "w": -0.5},
        threshold="v >= v_threshold",
        detection="crossing",
        dt=0.1,
        method="rk4",
        substeps=2,
    )

    with pytest.raises(NotImplementedError, match="not supported with multiply pipelining"):
        compile_to_verilog(neuron, pipeline_stages=1)


def test_compile_to_verilog_disambiguates_case_colliding_parameters() -> None:
    """Case-distinct parameter names must not collapse onto one Verilog port.

    ``str.upper()`` maps both ``C`` and ``c`` to ``P_C``, which iverilog rejects as a
    duplicate declaration. Verilog identifiers are case-sensitive, so the emitter keeps
    the parameter port map injective by falling back to a case-preserving identifier for
    the collision. A neuron with both a capacitance ``C`` and a reset voltage ``c`` (the
    Izhikevich 2007 naming) must therefore lower to two distinct ports.
    """
    neuron = EquationNeuron(
        equations={"v": "(C * v - c + I) / C"},
        parameters={"C": 2.0, "c": 1.0},
        state={"v": 0.0},
        threshold="v > 100.0",
        dt=1.0,
    )

    verilog = compile_to_verilog(neuron, module_name="sc_case_params")

    # Both case-distinct parameters survive as separate, case-sensitive ports.
    assert "P_C " in verilog  # capacitance, canonical upper-case identifier
    assert "P_c " in verilog  # reset voltage, case-preserved to avoid the collision
    # Each is declared exactly once — no redeclaration for iverilog to reject.
    assert verilog.count("P_C =") == 1
    assert verilog.count("P_c =") == 1


def test_compile_to_verilog_numbers_a_case_preserved_parameter_collision() -> None:
    """When even the case-preserved fallback is taken, a numeric suffix keeps it unique.

    An already-upper-case name (``X``) collides with a lower-case sibling (``x``) on both
    the upper-case form (``P_X``) and the case-preserved fallback (still ``P_X``), so the
    emitter appends a numeric suffix. The pathological triple guarantees the port map is
    injective for any case pattern, not only the common capacitance/reset (``C``/``c``)
    case.
    """
    neuron = EquationNeuron(
        equations={"v": "(x * v - X + I)"},
        parameters={"x": 0.5, "X": 0.25},
        state={"v": 0.0},
        threshold="v > 100.0",
        dt=1.0,
    )

    verilog = compile_to_verilog(neuron, module_name="sc_numbered_params")

    assert verilog.count("P_X =") == 1  # first parameter keeps the canonical form
    assert verilog.count("P_X_2 =") == 1  # the collision is disambiguated numerically


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
