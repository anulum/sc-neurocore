# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_sigmoid_rate.py

from __future__ import annotations

"""Full pipeline tests for the reduced sigmoid-rate relaxation model.

The maintained scalar equation is ``τ dr/dt = -r + σ(β(I-θ))``. It is inspired
by the population-rate motif in Wilson and Cowan (1972), not their full coupled
excitatory/inhibitory system.
"""
import time
from typing import cast
import numpy as np
import pytest
from sc_neurocore.neurons.model_catalogue import load_descriptor_payload
from sc_neurocore.neurons.models.sigmoid_rate import SigmoidRateNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from sc_neurocore.network.population import Population
def _stable_sigmoid(beta: float, current: float, theta: float) -> float:
    z = beta * (current - theta)
    if z >= 0.0:
        return 1.0 / (1.0 + np.exp(-z))
    exp_z = np.exp(z)
    return exp_z / (1.0 + exp_z)
def _exact_rate(r: float, sigma: float, dt: float, tau: float) -> float:
    decay = np.exp(-dt / tau)
    return decay * r + (1.0 - decay) * sigma

__all__ = ['time', 'cast', 'np', 'pytest', 'load_descriptor_payload', 'SigmoidRateNeuron', 'UniversalNeuron', 'Population', '_stable_sigmoid', '_exact_rate']
