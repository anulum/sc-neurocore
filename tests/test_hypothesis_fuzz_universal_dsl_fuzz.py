# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestUniversalDSLFuzz from former test_hypothesis_fuzz.py

"""Focused suite: TestUniversalDSLFuzz from former test_hypothesis_fuzz.py."""

from __future__ import annotations

from tests.hypothesis_fuzz_support import *  # noqa: F403


class TestUniversalDSLFuzz:
    """Property: random parameter values don't cause NaN or crash."""

    @given(
        current=st.floats(min_value=-1000, max_value=1000, allow_nan=False),
        steps=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_lif_never_produces_nan(self, current: float, steps: int) -> None:
        """LIF with random current should never produce NaN state."""
        neuron = UniversalNeuron.from_schema("lif")
        for _ in range(steps):
            neuron.step(I=current)
        v = neuron.state["v"]
        assert not math.isnan(v), f"LIF produced NaN with I={current}"

    @given(
        a=st.floats(min_value=0.01, max_value=2.0),
        b=st.floats(min_value=0.01, max_value=2.0),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_izhikevich_parameter_sweep(self, a: float, b: float) -> None:
        """Izhikevich with swept parameters should not crash."""
        neuron = UniversalNeuron.from_schema(
            "izhikevich",
            parameter_overrides={"a": a, "b": b},
        )
        for _ in range(50):
            neuron.step(I=10.0)
        v = neuron.state["v"]
        assert not math.isnan(v), f"Izhikevich NaN with a={a}, b={b}"

    @given(dt=st.floats(min_value=0.001, max_value=2.0))
    @settings(max_examples=50)
    def test_fitzhugh_nagumo_dt_sweep(self, dt: float) -> None:
        """Faithful FHN (RK4, no reset) either stays finite or fails closed.

        The re-enrolled FitzHugh-Nagumo is an unbounded cubic relaxation oscillator,
        so a large enough step size genuinely diverges (unlike the earlier bounded
        reset caricature). The runner fails closed on a non-finite state
        (``FloatingPointError``, matching the hand models) rather than silently
        propagating NaN, so a completed 100-step sweep must leave a finite state and a
        divergent step raises the controlled error instead of corrupting the trace.
        """
        neuron = UniversalNeuron.from_schema("fitzhugh_nagumo", dt_override=dt)
        try:
            for _ in range(100):
                neuron.step(I=0.5)
        except (OverflowError, ValueError, FloatingPointError):
            return  # controlled fail-closed divergence for an extreme step size
        v = neuron.state["v"]
        assert math.isfinite(v), f"FHN left a non-finite state without failing closed (dt={dt})"
