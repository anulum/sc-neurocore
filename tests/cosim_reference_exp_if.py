# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ExpIF co-simulation reference

"""Independent exponential integrate-and-fire RK4 reference contract."""

from __future__ import annotations

import math

from tests.cosim_reference_statistics import _summarise


def _exp_if_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return independent RK4 features for the source-bound driven EIF recurrence.

    Fourcaud-Trocmé et al. (2003), Equations 6 and 10, define the leak plus
    exponential current. This re-derivation uses the fitted ``V_T``, slope,
    leak and reset with the SC profile's ``+30 mV`` finite event surface. RK4 stages
    are bounded at that event surface, matching the maintained deterministic
    recurrence without importing the hand model or the schema runner.

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
        Reference feature map for voltage, spike count, and first-spike step.
    """
    v_rest = -65.0
    v_reset = -68.0
    v_threshold = 30.0
    v_rh = -59.9
    delta_t = 3.48
    tau = 10.0
    v = -65.0
    v_values: list[float] = []
    spikes: list[int] = []

    def rhs(stage_v: float) -> float:
        bounded_v = min(stage_v, v_threshold)
        return (
            -(bounded_v - v_rest) + delta_t * math.exp((bounded_v - v_rh) / delta_t) + current
        ) / tau

    for _ in range(steps):
        k1 = rhs(v)
        k2 = rhs(v + 0.5 * dt * k1)
        k3 = rhs(v + 0.5 * dt * k2)
        k4 = rhs(v + dt * k3)
        v_next = v + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if v_next >= v_threshold:
            spikes.append(1)
            v_next = v_reset
        else:
            spikes.append(0)
        v = v_next
        v_values.append(v)

    return _summarise({"v": v_values}, spikes)
