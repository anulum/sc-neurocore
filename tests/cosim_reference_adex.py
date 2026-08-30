# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AdEx co-simulation reference

"""Independent AdEx subthreshold Euler reference contract."""

from __future__ import annotations

import math

from tests.cosim_reference_statistics import _summarise


def _adex_published_euler_trace(
    *, current: float, dt: float, steps: int
) -> tuple[list[float], list[float], list[int]]:
    """Evaluate the source-paper regular-spiking fit without runtime imports.

    This uses the dimensional equations and parameter fit printed in Brette and
    Gerstner (2005), including the numerical ``Vpeak=20 mV`` cutoff, ``Vr=EL``
    reset, and ``w <- w+b`` event update.  Explicit Euler and ``dt`` are stated
    evidence-protocol choices rather than source-model claims.
    """
    capacitance = 281.0
    leak_conductance = 30.0
    rest = -70.6
    rheobase = -50.4
    slope = 2.0
    tau_w = 144.0
    adaptation = 4.0
    spike_increment = 80.5
    peak = 20.0
    v = rest
    w = 0.0
    v_trace: list[float] = []
    w_trace: list[float] = []
    events: list[int] = []
    for _ in range(steps):
        dv = (
            -leak_conductance * (v - rest)
            + leak_conductance * slope * math.exp((v - rheobase) / slope)
            - w
            + current
        ) / capacitance
        dw = (adaptation * (v - rest) - w) / tau_w
        v_next = v + dt * dv
        w_next = w + dt * dw
        event = int(v_next >= peak)
        if event:
            v_next = rest
            w_next += spike_increment
        v, w = v_next, w_next
        v_trace.append(v)
        w_trace.append(w)
        events.append(event)
    return v_trace, w_trace, events


def _adex_subthreshold_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact explicit-Euler features for the subthreshold AdEx recurrence.

    The Brette-Gerstner (2005) exponential membrane and linear adaptation equations
    are advanced with the same simultaneous explicit-Euler update the schema runner
    applies. For the resting zero-current protocol the ``v > -50`` threshold is never
    reached, so the ``v = v_reset``, ``w = w + b`` reset stays inactive and the
    reference is an independent re-derivation of the committed quiet trajectory.

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
    v_rest = -65.0
    v_reset = -68.0
    v_rh = -55.0
    delta_t = 2.0
    tau = 20.0
    tau_w = 100.0
    a = 0.5
    b_adapt = 7.0
    capacitance = 200.0
    v = -65.0
    w = 0.0
    v_values: list[float] = []
    w_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        dv = (-(v - v_rest) + delta_t * math.exp((v - v_rh) / delta_t)) / tau + (
            -w + current
        ) / capacitance
        dw = (a * (v - v_rest) - w) / tau_w
        v_next = v + dv * dt
        w_next = w + dw * dt
        if v_next > -50:
            spikes.append(1)
            v_next = v_reset
            w_next = w_next + b_adapt
        else:
            spikes.append(0)
        v, w = v_next, w_next
        v_values.append(v)
        w_values.append(w)

    return _summarise({"v": v_values, "w": w_values}, spikes)
