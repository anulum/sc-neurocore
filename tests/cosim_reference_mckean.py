# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — McKean co-simulation references

"""Independent McKean spike-count and RK4 reference contracts."""

from __future__ import annotations

from sc_neurocore.neurons.models.sc_triangular_mckean import SCTriangularMcKeanNeuron
from tests.cosim_reference_statistics import _summarise


# Retained project parameters mirrored by ``sc_triangular_mckean``. This helper
# verifies the preserved recurrence; the source McKean identity has a separate
# receipt and must not be blended back into this compatibility oracle.
_MCKEAN_PARAMS = {"a": 0.25, "epsilon": 0.01, "gamma": 0.5, "v_peak": 0.8}


def _mckean_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored McKean (RK4, rising-edge crossing) spike count."""
    neuron = SCTriangularMcKeanNeuron(dt=0.1, v=0.0, w=0.0, **_MCKEAN_PARAMS)
    return sum(neuron.step(current) for _ in range(n_steps))


def _mckean_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact classical-RK4 features for the driven McKean oscillator.

    The McKean (1970) piecewise-linear FitzHugh-Nagumo caricature replaces the cubic
    membrane nullcline with the three-branch function ``f(v) = min(max(-v, v - a),
    1 - v)`` (min/max are supported by the schema DSL). The membrane and linear
    recovery equations are advanced with the same four-stage RK4 step and rising-edge
    ``v >= v_peak`` crossing detection the faithful schema runner applies, with **no
    reset**. The retained project point uses ``epsilon = 0.01``, ``gamma = 0.5``, and
    ``I = 0.6``; its initial upward threshold crossing is an observational event. The
    right-hand side is exact arithmetic (comparisons and linear pieces, no cube or
    transcendental), so the reference is an independent re-derivation of the committed
    trace, not a copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v`` and ``w`` state variables plus
        spike-count and first-spike-step features.
    """
    a = 0.25
    epsilon = 0.01
    gamma = 0.5
    v_peak = 0.8
    v = 0.0
    w = 0.0
    v_values: list[float] = []
    w_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, w_state: float) -> tuple[float, float]:
        f_v = min(max(-v_state, v_state - a), 1.0 - v_state)
        return f_v - w_state + current, epsilon * (v_state - gamma * w_state)

    for _ in range(steps):
        v_prev = v
        k1v, k1w = deriv(v, w)
        k2v, k2w = deriv(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w)
        k3v, k3w = deriv(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w)
        k4v, k4w = deriv(v + dt * k3v, w + dt * k3w)
        v = v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        w = w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0
        spikes.append(1 if (v >= v_peak and v_prev < v_peak) else 0)
        v_values.append(v)
        w_values.append(w)

    return _summarise({"v": v_values, "w": w_values}, spikes)
