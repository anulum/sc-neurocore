# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hodgkin-Huxley integrator path contracts

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron
from tests.neuron_integrator_paths_support import as_untyped, count_spikes


def test_hodgkin_huxley_integrator_validation() -> None:
    with pytest.raises(ValueError, match="Unsupported integrator"):
        HodgkinHuxleyNeuron(integrator=as_untyped("bad-path"))


@pytest.mark.parametrize("current", [math.nan, math.inf, -math.inf])
def test_hodgkin_huxley_invalid_current_is_failure_atomic(current: float) -> None:
    neuron = HodgkinHuxleyNeuron()
    before = (neuron.v, neuron.m, neuron.h, neuron.n)

    with pytest.raises(ValueError, match="current must be finite"):
        neuron.step(current)

    assert (neuron.v, neuron.m, neuron.h, neuron.n) == before


def test_hodgkin_huxley_invalid_candidate_is_failure_atomic() -> None:
    neuron = HodgkinHuxleyNeuron(v=249.0)
    before = (neuron.v, neuron.m, neuron.h, neuron.n)

    with pytest.raises(FloatingPointError, match="candidate left finite physical bounds"):
        neuron.step(2.0e4)

    assert (neuron.v, neuron.m, neuron.h, neuron.n) == before


def test_hodgkin_huxley_rk4_path_stays_finite_and_tracks_baseline() -> None:
    baseline = HodgkinHuxleyNeuron(dt=0.01, integrator="baseline_euler")
    candidate = HodgkinHuxleyNeuron(dt=0.01, integrator="rk4")

    baseline_spikes = count_spikes(baseline, 10.0, 1000)
    candidate_spikes = count_spikes(candidate, 10.0, 1000)

    for value in [candidate.v, candidate.m, candidate.h, candidate.n]:
        assert np.isfinite(value)
    assert baseline_spikes > 0
    assert candidate_spikes > 0
    assert abs(candidate_spikes - baseline_spikes) <= 20
    assert abs(candidate.v - baseline.v) < 15.0


def test_hodgkin_huxley_rosenbrock_path_tracks_rk4_and_keeps_gates_bounded() -> None:
    reference = HodgkinHuxleyNeuron(dt=0.02, integrator="rk4")
    candidate = HodgkinHuxleyNeuron(dt=0.02, integrator="rosenbrock")

    reference_spikes = count_spikes(reference, 10.0, 200)
    candidate_spikes = count_spikes(candidate, 10.0, 200)

    for value in [candidate.v, candidate.m, candidate.h, candidate.n]:
        assert np.isfinite(value)
    assert 0.0 <= candidate.m <= 1.0
    assert 0.0 <= candidate.h <= 1.0
    assert 0.0 <= candidate.n <= 1.0
    assert reference_spikes > 0
    assert abs(candidate_spikes - reference_spikes) <= 1
    assert abs(candidate.v - reference.v) < 1.0
