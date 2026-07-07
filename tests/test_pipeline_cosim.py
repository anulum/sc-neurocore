# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — latency-aware pipelined co-simulation (SR-2)

"""Latency-aware pipelined co-simulation: pipelined RTL stays bit-true to the golden.

Inserting register stages at the multiply outputs of a self-recurrent integration step
would, on its own, corrupt the recurrence — the increment applied each clock would reflect
stale (mid-fill) products, and unreset stages inject X into the state feedback. The
fill-counter FSM in :func:`compile_to_verilog` holds the state steady for ``latency`` cycles
so every stage drains before the state advances (one logical step every ``latency + 1``
clocks), restoring an identical fixed-point sequence. These tests drive the pipelined module
latency-aware and assert its spike count matches the combinational path — and, for the
exact-model sets, the Python golden — across all three integrators (Euler, RK4, exp-Euler).
"""

from __future__ import annotations

import pytest

from tests.cosim_support import (
    HAS_IVERILOG,
    spike_count_method,
    verilog_spike_count_method,
    verilog_spike_count_method_pipelined,
)

# Method × model spread exercised at pipeline_stages=1. The set spans all three integrators
# and both the exact-golden and stiff-hybrid regimes, including izhikevich — which, before the
# reset-to-0 + fill-counter fix, latched to X and produced zero spikes under pipelining.
_PIPELINED_PARITY_CASES = [
    ("lif", "euler", 50.0, 300),
    ("izhikevich", "euler", 50.0, 300),
    ("quadratic_if", "euler", 50.0, 300),
    ("lif", "exp_euler", 50.0, 300),
    ("resonate_fire", "exp_euler", 5.0, 300),
    ("adex", "exp_euler", 1000.0, 500),
    ("quadratic_if", "rk4", 50.0, 300),
    ("theta", "rk4", 50.0, 300),
    ("adex", "rk4", 1000.0, 500),
]

# Models whose combinational exp-Euler tracks the Python golden bit-true at Q16.16; pipelining
# must preserve that match. (Mirrors the exact-model set in the combinational exp-Euler suite.)
_EXP_EULER_EXACT_MODELS = [
    ("lif", 50.0, 300),
    ("lapicque", 50.0, 300),
    ("resonate_fire", 5.0, 300),
    ("adex", 1000.0, 500),
    ("theta", 50.0, 300),
]

# The integrator is a graph rewrite; the Q-format is a separate emission parameter, so the
# pipelined path inherits every representation for free.
_Q_FORMATS = [("Q16.16", 32, 16), ("Q12.12", 24, 12), ("Q18.18", 36, 18), ("Q20.12", 32, 12)]


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestPipelinedCosim:
    """Pipelined RTL reproduces the combinational golden spike count (SR-2, H3 timing spine)."""

    @pytest.mark.parametrize("model_name,method,current,n_steps", _PIPELINED_PARITY_CASES)
    def test_pipelined_matches_combinational(
        self, model_name: str, method: str, current: float, n_steps: int
    ) -> None:
        """pipeline_stages=1 reproduces the combinational (pipeline_stages=0) spike count exactly."""
        comb = verilog_spike_count_method(model_name, n_steps, current, 32, 16, method)
        piped, latency = verilog_spike_count_method_pipelined(
            model_name, n_steps, current, 32, 16, method, pipeline_stages=1
        )
        assert comb > 0, f"{model_name}/{method} combinational must spike"
        assert piped == comb, (
            f"{model_name}/{method} pipeline changed spikes: "
            f"comb={comb}, pipelined={piped} (latency={latency})"
        )

    def test_izhikevich_pipeline_no_longer_latches_x(self) -> None:
        """Regression: pipelined izhikevich once latched X (0 spikes); it now tracks the golden RTL.

        The reset-to-0 of the staging registers plus the fill-counter gate mean an unfilled
        pipeline never reaches the state feedback, so the two-state v*v self-multiply neuron
        produces the same non-zero, defined spike count pipelined as combinational.
        """
        comb = verilog_spike_count_method("izhikevich", 300, 50.0, 32, 16, "euler")
        piped, latency = verilog_spike_count_method_pipelined(
            "izhikevich", 300, 50.0, 32, 16, "euler", pipeline_stages=1
        )
        assert latency > 0, "izhikevich must actually pipeline (registered multiplies present)"
        assert piped == comb > 0, f"pipelined={piped}, combinational={comb}"

    @pytest.mark.parametrize("mode_name,data_width,fraction", _Q_FORMATS)
    def test_pipelined_exp_euler_representation_agnostic(
        self, mode_name: str, data_width: int, fraction: int
    ) -> None:
        """Pipelined exp-Euler inherits every Q-format, matching the combinational path exactly."""
        comb = verilog_spike_count_method("lif", 300, 50.0, data_width, fraction, "exp_euler")
        piped, _ = verilog_spike_count_method_pipelined(
            "lif", 300, 50.0, data_width, fraction, "exp_euler", pipeline_stages=1
        )
        assert piped == comb, f"{mode_name} pipelined exp-Euler: comb={comb}, pipelined={piped}"


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
@pytest.mark.parametrize("model_name,current,n_steps", _EXP_EULER_EXACT_MODELS)
def test_pipelined_exp_euler_tracks_python_golden(
    model_name: str, current: float, n_steps: int
) -> None:
    """Pipelined exp-Euler reproduces the Python golden spike count exactly (Q16.16).

    Transitive parity: the combinational exp-Euler already tracks the golden bit-true for these
    models, and the fill-counter FSM makes the pipelined path bit-true to the combinational one,
    so pipelining preserves the golden match despite the added latency.
    """
    py_spikes = spike_count_method(model_name, n_steps, current, "exp_euler")
    piped, _ = verilog_spike_count_method_pipelined(
        model_name, n_steps, current, 32, 16, "exp_euler", pipeline_stages=1
    )
    assert py_spikes > 0, f"Python exp-Euler {model_name} must spike"
    assert piped == py_spikes, (
        f"{model_name} pipelined exp-Euler mismatch: Python={py_spikes}, Verilog={piped}"
    )
