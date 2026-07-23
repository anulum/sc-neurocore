# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DPI neuron co-simulation references

"""Independent DPI neuron spike-count and driven-Euler reference contracts."""

from __future__ import annotations

import math

from sc_neurocore.neurons.models.dpi_neuron import DPINeuron
from tests.cosim_reference_statistics import _summarise


def _dpi_neuron_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored DPI (current-mode Euler) spike count for comparison."""
    neuron = DPINeuron()
    return sum(neuron.step(current) for _ in range(n_steps))


def _dpi_neuron_driven_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return independent features for the coupled published DPI equations.

    Indiveri, Stefanini, and Chicca (2010), Eqs. (2)–(3), define the nonlinear
    positive-feedback membrane current and the spike-triggered adaptation DPI.
    This helper re-derives both right-hand sides directly, advances them
    simultaneously with explicit Euler, holds ``i_mem`` at reset during the
    refractory pulse, and applies the post-update threshold/reset ordering. It
    intentionally does not call the maintained model or schema runner.

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
        Reference feature map for all three states plus event features.
    """
    if steps <= 0:
        raise ValueError("DPI reference trace requires at least one step")
    i_threshold = 1.0
    i_reset = 0.01
    i_rest = 0.1
    i_tau = 1.0
    i_g = 1.0
    i_tau_ahp = 0.1
    i_ga = 1.0
    i_spike = 5.0
    i_0 = 0.01
    kappa = 0.7
    alpha = 10.0
    tau = 20.0
    tau_ahp = 100.0
    refractory_period = 2.0
    i_mem = 0.01
    i_ahp = 0.01
    refractory_time = 0.0
    i_mem_values: list[float] = []
    i_ahp_values: list[float] = []
    refractory_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        spike_active = refractory_time > 0.0
        spike_current = i_spike if spike_active else 0.0
        d_i_ahp = i_ahp / (tau_ahp * i_tau_ahp) * (spike_current / (1.0 + i_ahp / i_ga) - i_tau_ahp)
        next_i_ahp = i_ahp + dt * d_i_ahp
        if spike_active:
            event = 0
            next_i_mem = i_reset
            next_refractory = max(0.0, refractory_time - dt)
        else:
            log_current = (math.log(i_0) + kappa * math.log(i_mem)) / (kappa + 1.0)
            # Every candidate at or above threshold resets before the next iteration.
            gate_argument = alpha * (i_mem - i_threshold)
            exponential = math.exp(gate_argument)
            gate = exponential / (1.0 + exponential)
            i_fb = math.exp(log_current) * gate
            d_i_mem = (
                i_mem
                / (tau * i_tau)
                * ((i_rest + current) / (1.0 + i_mem / i_g) - i_tau + i_fb - i_ahp)
            )
            next_i_mem = i_mem + dt * d_i_mem
            event = int(next_i_mem >= i_threshold)
            next_refractory = 0.0
            if event:
                next_i_mem = i_reset
                next_refractory = refractory_period
        i_mem, i_ahp, refractory_time = next_i_mem, next_i_ahp, next_refractory
        spikes.append(event)
        i_mem_values.append(i_mem)
        i_ahp_values.append(i_ahp)
        refractory_values.append(refractory_time)

    return _summarise(
        {
            "i_mem": i_mem_values,
            "i_ahp": i_ahp_values,
            "refractory_time": refractory_values,
        },
        spikes,
    )
