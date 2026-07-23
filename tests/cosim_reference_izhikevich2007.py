# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Izhikevich 2007 co-simulation references

"""Independent Izhikevich 2007 spike-count and Euler reference contracts."""

from __future__ import annotations

from sc_neurocore.neurons.models.izhikevich2007 import Izhikevich2007Neuron
from tests.cosim_reference_statistics import _summarise


def _izhikevich2007_hand_euler_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored Izhikevich 2007 (Euler) spike count for comparison."""
    neuron = Izhikevich2007Neuron(integrator="euler")
    return sum(neuron.step(current) for _ in range(n_steps))


def _izhikevich2007_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact explicit-Euler features for the Izhikevich 2007 recurrence.

    The Izhikevich (2007) biophysical quadratic membrane ``C dv/dt =
    k (v - vr) (v - vt) - u + I`` and linear recovery ``du/dt = a (b (v - vr) - u)``
    are advanced with the same simultaneous explicit-Euler update the schema runner
    applies, and the ``v = c``, ``u = u + d`` reset fires whenever the post-update
    membrane reaches the ``v >= vpeak`` peak. The right-hand side is polynomial, so
    the recurrence reproduces the schema runner bit-for-bit — an independent
    re-derivation of the committed regular-spiking trace, not a copy of the runner.

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
    c_m = 100.0
    k = 0.7
    vr = -60.0
    vt = -40.0
    vpeak = 35.0
    a = 0.03
    b = -2.0
    c = -50.0
    d = 100.0
    v = -60.0
    u = 0.0
    v_values: list[float] = []
    u_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        dv = (k * (v - vr) * (v - vt) - u + current) / c_m
        du = a * (b * (v - vr) - u)
        v_next = v + dv * dt
        u_next = u + du * dt
        if v_next >= vpeak:
            spikes.append(1)
            v_next = c
            u_next = u_next + d
        else:
            spikes.append(0)
        v, u = v_next, u_next
        v_values.append(v)
        u_values.append(u)

    return _summarise({"v": v_values, "u": u_values}, spikes)
