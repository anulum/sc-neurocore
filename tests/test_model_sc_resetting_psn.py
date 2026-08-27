# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC resetting windowed neuron frozen anchors

"""Frozen bit-exact anchors for the preserved repository recurrence."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest

from sc_neurocore.neurons.models import SCResettingParallelSpikingNeuron as PublicSC
from sc_neurocore.neurons.models.sc_resetting_psn import SCResettingParallelSpikingNeuron


def test_registry_exports_the_preserved_class() -> None:
    assert PublicSC is SCResettingParallelSpikingNeuron


def test_frozen_anchor_varied_drive_64_steps() -> None:
    """Anchor A1: defaults, varied sub-threshold drive, bit-exact buffer."""
    neuron = SCResettingParallelSpikingNeuron()
    drive = [0.4 + 0.3 * math.sin(index * 0.17) for index in range(64)]
    spikes = [neuron.step(current) for current in drive]
    assert spikes == [0] * 64
    assert [repr(float(value)) for value in neuron.buffer] == [
        "0.3714765387020912",
        "0.32136293497363344",
        "0.27351647448898186",
        "0.22931659300901405",
        "0.19003759372193693",
        "0.15681190848831542",
        "0.1305974493026139",
        "0.11214999124383584",
    ]
    assert neuron._ptr == 64


def test_frozen_anchor_constant_drive_alternates_after_reset() -> None:
    """Anchor A2: spike-triggered buffer reset yields the alternating train."""
    neuron = SCResettingParallelSpikingNeuron(kernel_size=4, v_threshold=0.5)
    spikes = [neuron.step(1.0) for _ in range(20)]
    assert "".join(str(spike) for spike in spikes) == "01010101010101010101"
    assert [float(value) for value in neuron.buffer] == [0.0, 0.0, 0.0, 0.0]


def test_frozen_anchor_non_uniform_kernel_circular_pairing() -> None:
    """Anchor A3: a replaced kernel pairs circular slots, preserved verbatim."""
    neuron = SCResettingParallelSpikingNeuron(kernel_size=4, v_threshold=0.9)
    neuron.kernel = np.array([0.4, 0.3, 0.2, 0.1])
    drive = [0.9, 1.1, 0.7, 1.3, 0.2, 1.9, 0.8, 0.6, 1.4, 0.3, 1.0, 1.2]
    spikes = [neuron.step(current) for current in drive]
    assert "".join(str(spike) for spike in spikes) == "000100001000"
    assert [float(value) for value in neuron.buffer] == [0.0, 0.3, 1.0, 1.2]
    assert neuron._ptr == 12


def test_frozen_anchor_warm_up_divides_by_full_kernel_size() -> None:
    """Anchor A4: warm-up buffer occupancy and score normalisation."""
    neuron = SCResettingParallelSpikingNeuron()
    for current in (0.5, 0.25, 0.125):
        assert neuron.step(current) == 0
    assert [float(value) for value in neuron.buffer[:4]] == [0.5, 0.25, 0.125, 0.0]
    assert neuron._ptr == 3


def test_reset_clears_buffer_and_pointer() -> None:
    neuron = SCResettingParallelSpikingNeuron(kernel_size=4, v_threshold=0.5)
    neuron.step(1.0)
    neuron.reset()
    assert [float(value) for value in neuron.buffer] == [0.0, 0.0, 0.0, 0.0]
    assert neuron._ptr == 0


@pytest.mark.parametrize("bad", (math.nan, math.inf, -math.inf))
def test_non_finite_input_is_rejected_atomically(bad: float) -> None:
    neuron = SCResettingParallelSpikingNeuron()
    neuron.step(0.7)
    before = (list(neuron.buffer), neuron._ptr)
    with pytest.raises(ValueError, match="current"):
        neuron.step(bad)
    assert (list(neuron.buffer), neuron._ptr) == before


@pytest.mark.parametrize("kernel_size", (0, -3))
def test_non_positive_kernel_size_is_rejected(kernel_size: int) -> None:
    with pytest.raises(ValueError, match="kernel_size"):
        SCResettingParallelSpikingNeuron(kernel_size=kernel_size)


def test_non_integer_kernel_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="kernel_size"):
        SCResettingParallelSpikingNeuron(kernel_size=cast(int, 4.0))


def test_non_finite_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="v_threshold"):
        SCResettingParallelSpikingNeuron(v_threshold=math.nan)


def test_mutated_kernel_size_is_rejected_before_stepping() -> None:
    neuron = SCResettingParallelSpikingNeuron(kernel_size=2)
    neuron.kernel_size = 0
    with pytest.raises(ValueError, match="kernel_size"):
        neuron.step(0.1)


def test_corrupted_kernel_is_rejected_before_stepping() -> None:
    neuron = SCResettingParallelSpikingNeuron(kernel_size=2)
    neuron.kernel = np.array([0.5, math.nan])
    with pytest.raises(ValueError, match="kernel"):
        neuron.step(0.1)


def test_reshaped_kernel_is_rejected_before_stepping() -> None:
    neuron = SCResettingParallelSpikingNeuron(kernel_size=2)
    neuron.kernel = np.array([0.5])
    with pytest.raises(ValueError, match="kernel"):
        neuron.step(0.1)


def test_corrupted_buffer_is_rejected_before_stepping() -> None:
    neuron = SCResettingParallelSpikingNeuron(kernel_size=2)
    neuron.buffer = np.array([0.0, math.inf])
    with pytest.raises(ValueError, match="buffer"):
        neuron.step(0.1)


def test_reshaped_buffer_is_rejected_before_stepping() -> None:
    neuron = SCResettingParallelSpikingNeuron(kernel_size=2)
    neuron.buffer = np.array([0.0])
    with pytest.raises(ValueError, match="buffer"):
        neuron.step(0.1)
