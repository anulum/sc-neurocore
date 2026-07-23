# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_zoo.py

from __future__ import annotations

"""Full pipeline test for sc_neurocore.model_zoo.

10 pre-configured network architectures + 3 pretrained weight loaders.
Each config is tested for:
  1. Construction — factory returns a Network with correct topology
  2. Topology — population counts, projection wiring, monitor counts
  3. Dynamics — network produces spikes under Poisson drive
  4. Analytical — neuron model types match published references
  5. Scaling — parameter sweeps verify config scales correctly
  6. Performance — network throughput (neuron-steps/s)
  7. Pipeline — spike_count, firing_rate, ISI from monitor data
FULL PIPELINE WIRED + PERFORMANCE."""
import time
import numpy as np
import pytest
from sc_neurocore.model_zoo import (
    mnist_classifier,
    dvs_gesture_classifier,
    shd_speech_classifier,
    brunel_balanced_network,
    cortical_column,
    central_pattern_generator,
    decision_making_circuit,
    working_memory_circuit,
    auditory_processing,
    visual_cortex_v1,
)
from sc_neurocore.model_zoo.pretrained import load_pretrained
from sc_neurocore.network.network import Network
from sc_neurocore.neurons import StochasticLIFNeuron
from sc_neurocore.neurons.models import (
    CompteWMNeuron,
    GolombFSNeuron,
    HindmarshRoseNeuron,
    HodgkinHuxleyNeuron,
    PospischilNeuron,
    WangBuzsakiNeuron,
)
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi
def _total_spikes(net: Network) -> int:
    return sum(m.count for m in net.spike_monitors)
def _total_neurons(net: Network) -> int:
    return sum(p.n for p in net.populations)
def _run_and_count(net: Network, duration: float = 0.1) -> int:
    net.run(duration, dt=0.001, backend="python")
    return _total_spikes(net)
_ALL_BUILDERS = [
    ("mnist", lambda: mnist_classifier(n_hidden=16)),
    ("dvs", lambda: dvs_gesture_classifier(n_classes=4)),
    ("shd", lambda: shd_speech_classifier()),
    ("brunel", lambda: brunel_balanced_network(n_exc=50, n_inh=12)),
    ("cortical", lambda: cortical_column(n_layers=2)),
    ("cpg", lambda: central_pattern_generator(n_oscillators=2)),
    ("decision", lambda: decision_making_circuit(n_per_pool=10)),
    ("wm", lambda: working_memory_circuit(n_neurons=50)),
    ("auditory", lambda: auditory_processing(n_channels=8)),
    ("v1", lambda: visual_cortex_v1(n_orientation=2, n_per_orientation=5)),
]

__all__ = ['time', 'np', 'pytest', 'mnist_classifier', 'dvs_gesture_classifier', 'shd_speech_classifier', 'brunel_balanced_network', 'cortical_column', 'central_pattern_generator', 'decision_making_circuit', 'working_memory_circuit', 'auditory_processing', 'visual_cortex_v1', 'load_pretrained', 'Network', 'StochasticLIFNeuron', 'CompteWMNeuron', 'GolombFSNeuron', 'HindmarshRoseNeuron', 'HodgkinHuxleyNeuron', 'PospischilNeuron', 'WangBuzsakiNeuron', 'spike_count', 'firing_rate', 'isi', '_total_spikes', '_total_neurons', '_run_and_count', '_ALL_BUILDERS', '__all__']
