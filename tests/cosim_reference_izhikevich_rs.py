# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Izhikevich regular-spiking co-simulation reference

"""Independent Izhikevich regular-spiking explicit-Euler reference contract."""

from __future__ import annotations

from tests.cosim_reference_statistics import _summarise


def _izhikevich_rs_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact explicit-Euler features for the regular-spiking Izhikevich recurrence.

    The Izhikevich (2003) quadratic membrane and linear recovery equations are
    advanced with the same simultaneous explicit-Euler update the schema runner
    applies, and the ``v = c``, ``u = u + d`` reset fires whenever the post-update
    membrane crosses the ``v > 30`` peak. The reference is therefore an independent
    re-derivation of the committed spike-bearing trace, not a copy of the runner.

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
        Reference feature map for the ``v`` and ``u`` state variables plus
        spike-count and first-spike-step features.
    """
    a = 0.02
    b = 0.2
    c = -65.0
    d = 8.0
    v = -65.0
    u = -14.0
    v_values: list[float] = []
    u_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        dv = 0.04 * v**2 + 5 * v + 140 - u + current
        du = a * (b * v - u)
        v_next = v + dv * dt
        u_next = u + du * dt
        if v_next > 30:
            spikes.append(1)
            v_next = c
            u_next = u_next + d
        else:
            spikes.append(0)
        v, u = v_next, u_next
        v_values.append(v)
        u_values.append(u)

    return _summarise({"v": v_values, "u": u_values}, spikes)
