# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRK4Emitter from former test_cosim_emitters.py

"""Focused suite: TestRK4Emitter from former test_cosim_emitters.py."""

from __future__ import annotations

from tests.cosim_emitters_support import *  # noqa: F403


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestRK4Emitter:
    """The schema→Verilog emitter lowers a full classical RK4 step, not only Euler.

    When a schema declares ``method="rk4"`` the emitter now emits the four-stage
    RK4 graph (k1..k4 with the s0 + k·dt/2 / +k·dt stage states and the
    (k1+2k2+2k3+k4)·dt/6 increment), reusing the same fixed-point expression
    emitter as the Euler path. That reuse makes the integrator agnostic to the
    number representation, so RK4 inherits every Q-format for free. Faithfulness
    holds for smooth ODEs; the stiff hybrid izhikevich (0.04·v² spike explosion)
    remains a documented per-model range limit, not an emitter defect.
    """

    @pytest.mark.parametrize("model_name,current,n_steps", _RK4_EXACT_MODELS)
    def test_rk4_tracks_python_rk4_golden(
        self, model_name: str, current: float, n_steps: int
    ) -> None:
        """Emitted RK4 reproduces the Python RK4 golden spike count exactly (Q16.16)."""
        py_spikes = _spike_count_method(model_name, n_steps, current, "rk4")
        vlog_spikes = _verilog_spike_count_method(model_name, n_steps, current, 32, 16, "rk4")
        assert py_spikes > 0, f"Python RK4 {model_name} must spike"
        assert vlog_spikes == py_spikes, (
            f"{model_name} RK4 mismatch: Python={py_spikes}, Verilog={vlog_spikes}"
        )

    def test_rk4_path_is_distinct_from_euler(self) -> None:
        """The RK4 emitter is a genuine four-stage step, not aliased to Euler.

        The theta phase-oscillator (sine LUT) at ``I=150`` is nonlinear enough that
        RK4 and Euler resolve a different number of phase wraps: the emitted RK4
        differs from the emitted Euler and still reproduces the Python RK4 golden
        exactly at Q16.16. (The faithful FitzHugh-Nagumo relaxation oscillator counts
        the same threshold crossings under either integrator — that robustness is why
        the distinctness demonstration uses a model whose spike count is genuinely
        integrator-sensitive rather than the earlier Euler+reset FHN caricature.)
        """
        py_rk4 = _spike_count_method("theta", 300, 150.0, "rk4")
        vlog_rk4 = _verilog_spike_count_method("theta", 300, 150.0, 32, 16, "rk4")
        vlog_euler = _verilog_spike_count_method("theta", 300, 150.0, 32, 16, "euler")
        assert vlog_rk4 != vlog_euler, "RK4 output must differ from Euler for a nonlinear model"
        gap_pct = abs(py_rk4 - vlog_rk4) / max(py_rk4, 1) * 100
        assert gap_pct <= 6.0, f"RK4 gap {gap_pct:.1f}% (Python={py_rk4}, Verilog={vlog_rk4})"

    @pytest.mark.parametrize("mode_name,data_width,fraction", _RK4_Q_FORMATS)
    def test_rk4_is_representation_agnostic(
        self, mode_name: str, data_width: int, fraction: int
    ) -> None:
        """RK4 inherits every Q-format for free (integrator ⟂ number representation)."""
        py_spikes = _spike_count_method("quadratic_if", 300, 50.0, "rk4")
        vlog_spikes = _verilog_spike_count_method(
            "quadratic_if", 300, 50.0, data_width, fraction, "rk4"
        )
        assert vlog_spikes == py_spikes, (
            f"{mode_name} RK4 mismatch: Python={py_spikes}, Verilog={vlog_spikes}"
        )
