# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for model_zoo pre-configured network architectures

"""Tests for sc_neurocore.model_zoo: one test per config, run 100ms, verify spikes."""

from __future__ import annotations

import os

os.environ.setdefault("SC_NEUROCORE_NO_RUST", "1")

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

DURATION = 0.1  # 100 ms


def _total_spikes(net):
    return sum(m.count for m in net.spike_monitors)


def test_mnist_classifier():
    net = mnist_classifier(n_hidden=32)
    net.run(DURATION)
    assert _total_spikes(net) > 0


def test_dvs_gesture_classifier():
    net = dvs_gesture_classifier(n_classes=4)
    net.run(DURATION)
    assert _total_spikes(net) > 0


def test_shd_speech_classifier():
    net = shd_speech_classifier()
    net.run(DURATION)
    assert _total_spikes(net) > 0


def test_brunel_balanced_network():
    net = brunel_balanced_network(n_exc=100, n_inh=25)
    net.run(DURATION)
    assert _total_spikes(net) > 0


def test_cortical_column():
    net = cortical_column(n_layers=4)
    net.run(DURATION)
    assert _total_spikes(net) > 0


def test_central_pattern_generator():
    net = central_pattern_generator(n_oscillators=2)
    net.run(DURATION)
    assert _total_spikes(net) > 0


def test_decision_making_circuit():
    net = decision_making_circuit(n_per_pool=30)
    net.run(DURATION)
    assert _total_spikes(net) > 0


def test_working_memory_circuit():
    net = working_memory_circuit(n_neurons=80)
    net.run(DURATION)
    assert _total_spikes(net) > 0


def test_auditory_processing():
    net = auditory_processing(n_channels=8)
    net.run(DURATION)
    assert _total_spikes(net) > 0


def test_visual_cortex_v1():
    net = visual_cortex_v1(n_orientation=4, n_per_orientation=10)
    net.run(DURATION)
    assert _total_spikes(net) > 0
