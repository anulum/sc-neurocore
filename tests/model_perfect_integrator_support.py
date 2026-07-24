# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_perfect_integrator.py

from __future__ import annotations

"""Full pipeline test for PerfectIntegratorNeuron (Lapicque 1907, no leak).

dV/dt = I / C — voltage accumulates without decay.
Analytically: ISI = C·θ / (I·dt) steps, firing rate f = I / (C·θ)."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.perfect_integrator import PerfectIntegratorNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _collect_spike_times(neuron: PerfectIntegratorNeuron, current: float, steps: int) -> list[int]:
    """Run neuron and return list of step indices where spikes occurred."""
    return [t for t in range(steps) if neuron.step(current) == 1]


def _analytical_isi_steps(
    current: float, c_m: float, threshold: float, v_reset: float, dt: float
) -> float:
    """Exact ISI in steps: voltage ramp from v_reset to threshold.

    Each step adds dV = I/C * dt.  Steps to threshold = (θ - v_reset) / dV.
    """
    dv_per_step = current / c_m * dt
    if dv_per_step <= 0:
        return float("inf")
    return (threshold - v_reset) / dv_per_step


__all__ = [
    "np",
    "pytest",
    "PerfectIntegratorNeuron",
    "Population",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "spike_count",
    "_collect_spike_times",
    "_analytical_isi_steps",
    "__all__",
]
