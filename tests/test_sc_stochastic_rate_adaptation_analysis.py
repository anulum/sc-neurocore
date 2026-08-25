# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC stochastic rate-adaptation analysis tests

"""Analysis contracts for the retained stochastic adaptation model."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count
from sc_neurocore.neurons.models.sc_stochastic_rate_adaptation import (
    SCStochasticRateAdaptationNeuron,
)


def _binary_train() -> np.ndarray:
    neuron = SCStochasticRateAdaptationNeuron(seed=11)
    return np.fromiter((neuron.step(50.0) for _ in range(10_000)), dtype=np.int8)


def test_firing_rate_matches_count_and_duration() -> None:
    train = _binary_train()
    count = spike_count(train)

    assert count > 0
    assert firing_rate(train, dt=0.001) == pytest.approx(count / 10.0)
