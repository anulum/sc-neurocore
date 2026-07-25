# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Morris-Lecar integrator path contracts

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.morris_lecar import MorrisLecarNeuron
from tests.neuron_integrator_paths_support import as_untyped, count_spikes


def test_morris_lecar_integrator_validation() -> None:
    with pytest.raises(ValueError, match="Unsupported integrator"):
        MorrisLecarNeuron(integrator=as_untyped("bad-path"))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"c_m": 0.0}, "c_m"),
        ({"dt": 0.0}, "dt"),
        ({"phi": -0.1}, "phi"),
        ({"g_ca": np.nan}, "g_ca"),
        ({"v": np.inf}, "v"),
    ],
)
def test_morris_lecar_rejects_invalid_numerical_configuration(
    kwargs: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        MorrisLecarNeuron(**as_untyped(kwargs))


@pytest.mark.parametrize("integrator", ["baseline_euler", "rk4", "rosenbrock"])
def test_morris_lecar_rejects_non_finite_input_current(integrator: str) -> None:
    neuron = MorrisLecarNeuron(dt=0.05, integrator=as_untyped(integrator))

    with pytest.raises(ValueError, match="current"):
        neuron.step(np.nan)


def test_morris_lecar_rk4_path_stays_finite_and_tracks_baseline() -> None:
    baseline = MorrisLecarNeuron(dt=0.05, integrator="baseline_euler")
    candidate = MorrisLecarNeuron(dt=0.05, integrator="rk4")

    baseline_spikes = count_spikes(baseline, 100.0, 2000)
    candidate_spikes = count_spikes(candidate, 100.0, 2000)

    assert np.isfinite(candidate.v)
    assert np.isfinite(candidate.w)
    assert baseline_spikes > 0
    assert candidate_spikes > 0
    assert abs(candidate_spikes - baseline_spikes) <= 5
    assert abs(candidate.v - baseline.v) < 5.0
    assert abs(candidate.w - baseline.w) < 0.1


def test_morris_lecar_rosenbrock_path_tracks_rk4_and_keeps_gate_bounded() -> None:
    reference = MorrisLecarNeuron(dt=0.05, integrator="rk4")
    candidate = MorrisLecarNeuron(dt=0.05, integrator="rosenbrock")

    reference_spikes = count_spikes(reference, 100.0, 1000)
    candidate_spikes = count_spikes(candidate, 100.0, 1000)

    assert np.isfinite(candidate.v)
    assert np.isfinite(candidate.w)
    assert 0.0 <= candidate.w <= 1.0
    assert reference_spikes > 0
    assert candidate_spikes > 0
    assert abs(candidate_spikes - reference_spikes) <= 2
    assert abs(candidate.v - reference.v) < 5.0
    assert abs(candidate.w - reference.w) < 0.1
