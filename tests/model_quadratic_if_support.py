# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_quadratic_if.py

from __future__ import annotations

"""Full pipeline test for QuadraticIFNeuron (QIF).

dV/dt = V² + I. Canonical Type-I excitability.
Saddle-node bifurcation at I=0: I<0 stable, I>0 → periodic spiking."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.quadratic_if import QuadraticIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run(neuron: QuadraticIFNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


def _exact_qif_candidate(neuron: QuadraticIFNeuron, current: float) -> tuple[float, bool]:
    if current > 0.0:
        root_i = np.sqrt(current)
        phase = np.arctan(neuron.v / root_i)
        peak_phase = np.arctan(neuron.v_peak / root_i)
        next_phase = phase + root_i * neuron.dt
        if next_phase >= peak_phase or next_phase >= np.pi / 2.0:
            return neuron.v_reset, True
        return float(root_i * np.tan(next_phase)), False
    if current == 0.0:
        denominator = 1.0 - neuron.v * neuron.dt
        if denominator <= 0.0:
            return neuron.v_reset, True
        next_v = neuron.v / denominator
        return (neuron.v_reset, True) if next_v >= neuron.v_peak else (float(next_v), False)

    root_i = np.sqrt(-current)
    if abs(neuron.v + root_i) <= 1e-15:
        return neuron.v, False
    numerator_ratio = (neuron.v - root_i) / (neuron.v + root_i)
    evolved_ratio = numerator_ratio * np.exp(2.0 * root_i * neuron.dt)
    denominator = 1.0 - evolved_ratio
    if denominator <= 0.0:
        return neuron.v_reset, True
    next_v = root_i * (1.0 + evolved_ratio) / denominator
    return (neuron.v_reset, True) if next_v >= neuron.v_peak else (float(next_v), False)


def _euler_candidate(neuron: QuadraticIFNeuron, current: float) -> float:
    return neuron.v + (neuron.v * neuron.v + current) * neuron.dt


__all__ = [
    "np",
    "pytest",
    "QuadraticIFNeuron",
    "Population",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "spike_count",
    "_run",
    "_exact_qif_candidate",
    "_euler_candidate",
]
