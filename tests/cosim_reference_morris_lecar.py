# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Morris-Lecar co-simulation references

"""Independent Morris-Lecar spike-count and RK4 reference contracts."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.morris_lecar import MorrisLecarNeuron
from tests.cosim_reference_statistics import _summarise


def _morris_lecar_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored Morris-Lecar (RK4, rising-edge crossing) spike count.

    The bundled ``morris_lecar`` schema mirrors ``MorrisLecarNeuron``'s maintained
    defaults exactly (RK4 integrator, no reset, ``v >= v_threshold`` upward crossing,
    ``phi = 1/15``), so the default construction is the enrolled operating point.
    """
    neuron = MorrisLecarNeuron()
    return sum(neuron.step(current) for _ in range(n_steps))


def _morris_lecar_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact classical-RK4 features for the driven Morris-Lecar oscillator.

    The Morris-Lecar (1981) calcium-potassium oscillator is the faithful
    conductance model: a genuine relaxation oscillator whose spikes are upward
    ``v >= v_threshold`` crossings, integrated with the same four-stage classical
    RK4 step the maintained ``MorrisLecarNeuron`` uses, with **no reset**. The
    sigmoidal calcium activation and potassium gating rate functions are transcribed
    verbatim from the schema, reusing ``numpy.tanh`` and ``numpy.cosh`` so the
    recurrence reproduces the schema runner bit-for-bit (the input current enters at
    every RK4 stage). The reference is an independent re-derivation of the committed
    driven-oscillation trace, not a copy of the runner.

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
    c_m = 20.0
    g_ca = 4.0
    g_k = 8.0
    g_l = 2.0
    e_ca = 120.0
    e_k = -84.0
    e_l = -60.0
    v1 = -1.2
    v2 = 18.0
    v3 = 12.0
    v4 = 17.4
    phi = 0.06666666666666667
    v_threshold = 0.0
    v = -60.0
    w = 0.0
    v_values: list[float] = []
    w_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, w_state: float) -> tuple[float, float]:
        dv = (
            -g_ca * 0.5 * (1 + float(np.tanh((v_state - v1) / v2))) * (v_state - e_ca)
            - g_k * w_state * (v_state - e_k)
            - g_l * (v_state - e_l)
            + current
        ) / c_m
        dw = (
            phi
            * float(np.cosh((v_state - v3) / (2 * v4)))
            * (0.5 * (1 + float(np.tanh((v_state - v3) / v4))) - w_state)
        )
        return dv, dw

    for _ in range(steps):
        v_prev = v
        k1v, k1w = deriv(v, w)
        k2v, k2w = deriv(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w)
        k3v, k3w = deriv(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w)
        k4v, k4w = deriv(v + dt * k3v, w + dt * k3w)
        v = v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        w = w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0
        # Rising-edge crossing: fires when the post-step membrane is at/above threshold
        # and the previous committed membrane was below it (matching the hand model's
        # ``v >= thr and v_prev < thr`` edge test); no reset for this oscillator.
        spikes.append(1 if (v >= v_threshold and v_prev < v_threshold) else 0)
        v_values.append(v)
        w_values.append(w)

    return _summarise({"v": v_values, "w": w_values}, spikes)
