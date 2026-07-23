# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_ai_optimized.py

from __future__ import annotations

"""Full pipeline test for all 8 AI-optimised neuron models.

MultiTimescale, AttentionGated, PredictiveCoding, SelfReferential,
CompositionalBinding, DifferentiableSurrogate, ContinuousAttractor,
MetaPlastic. All return int, all fire at I≥2.0.
Performance range: 4K–880K steps/s. All pipeline-wired."""
import time
import numpy as np
import pytest
from tests.performance_guard import assert_throughput_guard
from sc_neurocore.neurons.models.ai_optimized import (
    AttentionGatedNeuron,
    CompositionalBindingNeuron,
    ContinuousAttractorNeuron,
    DifferentiableSurrogateNeuron,
    MetaPlasticNeuron,
    MultiTimescaleNeuron,
    PredictiveCodingNeuron,
    SelfReferentialNeuron,
)
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count
ALL_CLASSES = [
    MultiTimescaleNeuron,
    AttentionGatedNeuron,
    PredictiveCodingNeuron,
    SelfReferentialNeuron,
    CompositionalBindingNeuron,
    DifferentiableSurrogateNeuron,
    ContinuousAttractorNeuron,
    MetaPlasticNeuron,
]

__all__ = ['time', 'np', 'pytest', 'assert_throughput_guard', 'AttentionGatedNeuron', 'CompositionalBindingNeuron', 'ContinuousAttractorNeuron', 'DifferentiableSurrogateNeuron', 'MetaPlasticNeuron', 'MultiTimescaleNeuron', 'PredictiveCodingNeuron', 'SelfReferentialNeuron', 'Population', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', 'ALL_CLASSES']
