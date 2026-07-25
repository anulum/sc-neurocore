# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — adaptive-exponential integrator path contracts

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.adex import AdExNeuron
from tests.neuron_integrator_paths_support import as_untyped, count_spikes


def test_adex_integrator_validation() -> None:
    with pytest.raises(ValueError, match="Unsupported integrator"):
        AdExNeuron(integrator=as_untyped("bad-path"))


def test_adex_rk4_path_stays_finite_and_tracks_baseline() -> None:
    baseline = AdExNeuron(dt=0.1, integrator="baseline_euler")
    candidate = AdExNeuron(dt=0.1, integrator="rk4")

    baseline_spikes = count_spikes(baseline, 500.0, 3000)
    candidate_spikes = count_spikes(candidate, 500.0, 3000)

    assert np.isfinite(candidate.v)
    assert np.isfinite(candidate.w)
    assert baseline_spikes > 0
    assert candidate_spikes > 0
    assert abs(candidate_spikes - baseline_spikes) <= 10
    assert abs(candidate.w - baseline.w) < 20.0


def test_adex_rosenbrock_path_tracks_rk4_and_stays_finite() -> None:
    reference = AdExNeuron(dt=0.2, integrator="rk4")
    candidate = AdExNeuron(dt=0.2, integrator="rosenbrock")

    reference_spikes = count_spikes(reference, 500.0, 500)
    candidate_spikes = count_spikes(candidate, 500.0, 500)

    assert np.isfinite(candidate.v)
    assert np.isfinite(candidate.w)
    assert reference_spikes > 0
    assert abs(candidate_spikes - reference_spikes) <= 1
    assert abs(candidate.v - reference.v) < 1.0
    assert abs(candidate.w - reference.w) < 1.0
