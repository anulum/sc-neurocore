# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExpEulerEmitter from former test_cosim_emitters.py

"""Focused suite: TestExpEulerEmitter from former test_cosim_emitters.py."""

from __future__ import annotations

from tests.cosim_emitters_support import *  # noqa: F403


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestExpEulerEmitter:
    """The schema→Verilog emitter lowers a linearised exponential-Euler step.

    When a schema declares ``method="exp_euler"`` the emitter emits, per variable,
    ``d<var> = f·dt·exprel(A·dt)`` with ``A = ∂f/∂x`` — the *same* symbolic derivative
    string the golden compiled (``EquationNeuron.jacobian_expressions``) lowered by the
    same fixed-point expression emitter as the Euler and RK4 paths, reusing the ``exprel``
    hardware LUT. That reuse makes the integrator agnostic to the number representation,
    so exp-Euler inherits every Q-format for free, and collapses to forward Euler in the
    zero-Jacobian limit (``exprel(0)=1``). Exact spike-count parity holds for the gating,
    linear and transcendental-Jacobian models above; the stiff hybrids (izhikevich,
    quadratic_if) remain a documented per-model range limit, not an emitter defect.
    """

    @pytest.mark.parametrize("model_name,current,n_steps", _EXP_EULER_EXACT_MODELS)
    def test_exp_euler_tracks_python_golden(
        self, model_name: str, current: float, n_steps: int
    ) -> None:
        """Emitted exp-Euler reproduces the Python golden spike count exactly (Q16.16)."""
        py_spikes = _spike_count_method(model_name, n_steps, current, "exp_euler")
        vlog_spikes = _verilog_spike_count_method(model_name, n_steps, current, 32, 16, "exp_euler")
        assert py_spikes > 0, f"Python exp-Euler {model_name} must spike"
        assert vlog_spikes == py_spikes, (
            f"{model_name} exp-Euler mismatch: Python={py_spikes}, Verilog={vlog_spikes}"
        )

    def test_exp_euler_tracks_two_state_linear_oscillator(self) -> None:
        """A synthetic coupled oscillator keeps multi-variable exp-Euler coverage."""
        py_spikes = _linear_oscillator_spike_count("exp_euler", 300, 50.0)
        vlog_spikes = _linear_oscillator_verilog_spike_count("exp_euler", 300, 50.0)
        assert vlog_spikes == py_spikes == 147

    def test_exp_euler_collapses_to_forward_euler_at_zero_jacobian(self) -> None:
        """With A=0 (perfect integrator) exprel(0)=1, so exp-Euler *is* forward Euler.

        The emitted exp-Euler datapath still multiplies by the tabulated ``exprel(0)``,
        so this proves the zero-Jacobian limit survives the LUT: the exp-Euler RTL, the
        Euler RTL and the Python golden all agree exactly. The explicit derivative-form
        fixture avoids overriding the source-faithful map semantics of the maintained
        PerfectIntegratorNeuron schema.
        """
        py_spikes = _zero_jacobian_spike_count("exp_euler", 300, 5.0)
        vlog_exp = _zero_jacobian_verilog_spike_count("exp_euler", 300, 5.0)
        vlog_euler = _zero_jacobian_verilog_spike_count("euler", 300, 5.0)
        assert py_spikes > 0
        assert vlog_exp == vlog_euler == py_spikes, (
            f"A=0 limit broke: exp={vlog_exp}, euler={vlog_euler}, py={py_spikes}"
        )

    def test_exp_euler_path_is_distinct_from_euler(self) -> None:
        """The exp-Euler emitter is a genuine linearised step, not aliased to Euler.

        A synthetic coupled linear oscillator at ``I=30`` is stiff enough that
        forward Euler and exponential-Euler resolve different crossing counts. Keeping
        this fixture local prevents a generic integrator test from overriding a
        production model's maintained exact-map discretisation.
        """
        py_exp = _linear_oscillator_spike_count("exp_euler", 300, 30.0)
        vlog_exp = _linear_oscillator_verilog_spike_count("exp_euler", 300, 30.0)
        vlog_euler = _linear_oscillator_verilog_spike_count("euler", 300, 30.0)
        assert vlog_exp != vlog_euler, "exp-Euler output must differ from Euler for a stiff model"
        gap_pct = abs(py_exp - vlog_exp) / max(py_exp, 1) * 100
        assert gap_pct <= 6.0, f"exp-Euler gap {gap_pct:.1f}% (Python={py_exp}, Verilog={vlog_exp})"

    @pytest.mark.parametrize("mode_name,data_width,fraction", _EXP_EULER_Q_FORMATS)
    def test_exp_euler_is_representation_agnostic(
        self, mode_name: str, data_width: int, fraction: int
    ) -> None:
        """exp-Euler inherits every Q-format for free (integrator ⟂ number representation)."""
        py_spikes = _spike_count_method("lif", 300, 50.0, "exp_euler")
        vlog_spikes = _verilog_spike_count_method(
            "lif", 300, 50.0, data_width, fraction, "exp_euler"
        )
        assert vlog_spikes == py_spikes, (
            f"{mode_name} exp-Euler mismatch: Python={py_spikes}, Verilog={vlog_spikes}"
        )
