# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FitzHugh-Nagumo integrator path contracts

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron
from tests.neuron_integrator_paths_support import as_untyped, count_spikes


def test_fitzhugh_nagumo_integrator_validation() -> None:
    with pytest.raises(ValueError, match="Unsupported integrator"):
        FitzHughNagumoNeuron(integrator=as_untyped("bad-path"))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dt": 0.0}, "dt"),
        ({"dt": np.nan}, "dt"),
        ({"epsilon": 0.0}, "epsilon"),
        ({"b": -0.1}, "b"),
        ({"v": np.inf}, "v"),
        ({"w": np.nan}, "w"),
    ],
)
def test_fitzhugh_nagumo_rejects_invalid_numerical_configuration(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        FitzHughNagumoNeuron(**as_untyped(kwargs))


@pytest.mark.parametrize("integrator", ["baseline_euler", "rk4", "rosenbrock"])
def test_fitzhugh_nagumo_rejects_non_finite_input_current(integrator: str) -> None:
    neuron = FitzHughNagumoNeuron(dt=0.05, integrator=as_untyped(integrator))

    with pytest.raises(ValueError, match="current"):
        neuron.step(np.inf)


def test_fitzhugh_nagumo_rk4_path_stays_finite_and_tracks_baseline() -> None:
    baseline = FitzHughNagumoNeuron(dt=0.05, integrator="baseline_euler")
    candidate = FitzHughNagumoNeuron(dt=0.05, integrator="rk4")

    baseline_spikes = count_spikes(baseline, 0.8, 2000)
    candidate_spikes = count_spikes(candidate, 0.8, 2000)

    assert np.isfinite(candidate.v)
    assert np.isfinite(candidate.w)
    assert baseline_spikes > 0
    assert candidate_spikes > 0
    assert abs(candidate_spikes - baseline_spikes) <= 2
    assert abs(candidate.v - baseline.v) < 0.3
    assert abs(candidate.w - baseline.w) < 0.3


def test_fitzhugh_nagumo_rosenbrock_path_tracks_rk4_and_stays_finite() -> None:
    reference = FitzHughNagumoNeuron(dt=0.05, integrator="rk4")
    candidate = FitzHughNagumoNeuron(dt=0.05, integrator="rosenbrock")

    reference_spikes = count_spikes(reference, 0.8, 1000)
    candidate_spikes = count_spikes(candidate, 0.8, 1000)

    assert np.isfinite(candidate.v)
    assert np.isfinite(candidate.w)
    assert reference_spikes > 0
    assert candidate_spikes > 0
    assert abs(candidate_spikes - reference_spikes) <= 1
    assert abs(candidate.v - reference.v) < 0.2
    assert abs(candidate.w - reference.w) < 0.2
