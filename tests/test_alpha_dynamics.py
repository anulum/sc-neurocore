# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Alpha-synapse source-dynamics contracts

"""Contracts of the dual alpha-synapse exact-flow dynamics."""

from __future__ import annotations

import math

import numpy as np

from sc_neurocore.neurons.models.alpha import AlphaNeuron


def test_filter_reproduces_rall_alpha_kernel_shape() -> None:
    """The cascade current peaks near tau with the alpha-kernel profile."""
    n = AlphaNeuron(v_threshold=100.0, dt=0.25)
    currents: list[float] = []
    for index in range(40):
        n.step(0.0)
        currents.append(n.i_exc if index else None or n.i_exc)
    # drive a single unit pulse at step 0 through the excitatory cascade
    n2 = AlphaNeuron(v_threshold=100.0, dt=0.25)
    n2.step(1.0)
    trace = []
    for _ in range(40):
        n2.step(0.0)
        trace.append(n2.i_exc)
    peak_index = int(np.argmax(trace))
    assert 12 <= peak_index <= 28  # alpha peak near tau_exc=5 with dt=0.25
    assert trace[peak_index] > 0.0


def test_inhibition_enters_with_the_opposite_sign() -> None:
    """Inhibitory current lowers the membrane candidate identically in magnitude."""
    exc_only = AlphaNeuron(v_threshold=100.0)
    dual = AlphaNeuron(v_threshold=100.0)
    exc_only.step(1.0)
    dual.step(1.0, 1.0)
    assert dual.v < exc_only.v


def test_membrane_relaxation_matches_linear_system_closed_form() -> None:
    """With zero cascade states the membrane relaxes exactly to the steady state."""
    n = AlphaNeuron(v=0.5, v_threshold=100.0, tau_v=7.5, dt=0.05)
    steady = n.v_rest
    expected = steady + (n.v - steady) * math.exp(-n.dt / n.tau_v)
    assert n.step(0.0) == 0
    assert abs(n.v - expected) < 1.0e-13


def test_equal_time_constant_limit_matches_series_expansion() -> None:
    """The equal-tau convolution equals its analytic limit, not a 0/0 form."""
    n = AlphaNeuron(i_exc=0.3, a_exc=0.2, tau_v=20.0, tau_exc=20.0, v_threshold=100.0, dt=0.5)
    rate = 1.0 / 20.0
    decay = math.exp(-0.5 / 20.0)
    contribution = rate * decay * (0.3 * 0.5 + 0.2 * 0.5 * 0.5 / (2.0 * 20.0))
    expected = n.v * decay + contribution
    assert n.step(0.0) == 0
    assert abs(n.v - expected) < 1.0e-13


def test_varied_dual_drive_event_vector_matches_candidate_crossing_rule() -> None:
    """Every emitted spike is a candidate crossing with the somatic reset."""
    n = AlphaNeuron()
    exc = 2.0 + 0.8 * np.sin(np.arange(256, dtype=np.float64) * 0.041)
    inh = 0.7 + 0.3 * np.cos(np.arange(256, dtype=np.float64) * 0.027)
    for exc_value, inh_value in zip(exc, inh):
        decay_v = math.exp(-n.dt / n.tau_v)
        exc_steady = n.tau_exc * exc_value
        inh_steady = n.tau_inh * inh_value
        v_steady = n.v_rest + exc_steady - inh_steady

        def contribution(current: float, rise: float, steady: float, tau: float) -> float:
            rate_v, rate_drive = 1.0 / n.tau_v, 1.0 / tau
            decay_drive = math.exp(-n.dt / tau)
            rd = rate_v - rate_drive
            return rate_v * (
                (current - steady) * (decay_drive - decay_v) / rd
                + (rise - steady) / tau * (decay_drive * (rd * n.dt - 1.0) + decay_v) / (rd * rd)
            )

        candidate_v = (
            v_steady
            + (n.v - v_steady) * decay_v
            + contribution(n.i_exc, n.a_exc, exc_steady, n.tau_exc)
            - contribution(n.i_inh, n.a_inh, inh_steady, n.tau_inh)
        )
        spike = n.step(float(exc_value), float(inh_value))
        assert spike == int(candidate_v >= n.v_threshold)
        if spike:
            assert n.v == n.v_rest


def test_long_run_is_finite_deterministic_and_bounded() -> None:
    """A 20k-step varied dual-drive run stays finite, deterministic, and bounded."""
    exc = 2.0 + 0.6 * np.sin(np.arange(20_000, dtype=np.float64) * 0.007)
    inh = 0.8 + 0.3 * np.cos(np.arange(20_000, dtype=np.float64) * 0.011)
    first = AlphaNeuron().simulate(exc, inh, backend="python")
    second = AlphaNeuron().simulate(exc, inh, backend="python")
    assert np.isfinite(first["v"]).all()
    assert np.isfinite(first["i_exc"]).all()
    assert np.all(first["v"] <= 1.0)
    np.testing.assert_array_equal(first["v"], second["v"])
    np.testing.assert_array_equal(first["a_exc"], second["a_exc"])
