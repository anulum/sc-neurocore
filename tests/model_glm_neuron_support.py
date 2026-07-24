# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_glm_neuron.py

from __future__ import annotations

"""Full pipeline test for GLMNeuron (Pillow et al. 2008).

Point-process generalised linear model:
λ(t) = exp(k·stim_buf + h·spike_buf + μ)
P(spike) = min(λ·dt_ms/1000, 1)

k: stimulus filter (n_k=10, exponential decay)
h: post-spike filter (n_h=20, negative=refractoriness + slow excitation)
μ=-3.0 (baseline log-rate). Stochastic — Bernoulli sampling.
Circular buffers for stimulus history and spike history.
log_rate clipped to [-20, 20] to prevent overflow.
FULL PIPELINE WIRED + PERFORMANCE."""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.glm_neuron import GLMNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: GLMNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


__all__ = [
    "time",
    "np",
    "pytest",
    "GLMNeuron",
    "Population",
    "Projection",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "spike_count",
    "firing_rate",
    "isi",
    "_run",
]
