# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Explicit baseline/candidate integrator path tests for selected neurons

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.adex import AdExNeuron
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron
from sc_neurocore.neurons.sc_izhikevich import SCIzhikevichNeuron


def _count_spikes(neuron, current: float, steps: int) -> int:
    return sum(int(neuron.step(current)) for _ in range(steps))


def test_izhikevich_integrator_validation():
    with pytest.raises(ValueError, match="Unsupported integrator"):
        SCIzhikevichNeuron(integrator="bad-path")  # type: ignore[arg-type]


def test_hodgkin_huxley_integrator_validation():
    with pytest.raises(ValueError, match="Unsupported integrator"):
        HodgkinHuxleyNeuron(integrator="bad-path")  # type: ignore[arg-type]


def test_adex_integrator_validation():
    with pytest.raises(ValueError, match="Unsupported integrator"):
        AdExNeuron(integrator="bad-path")  # type: ignore[arg-type]


def test_izhikevich_rk4_regular_spiking_and_default_preserved():
    baseline = SCIzhikevichNeuron(noise_std=0.0, dt=0.5, integrator="baseline_half_euler")
    candidate = SCIzhikevichNeuron(noise_std=0.0, dt=0.5, integrator="rk4")
    default = SCIzhikevichNeuron(noise_std=0.0, dt=0.5)

    baseline_spikes = _count_spikes(baseline, 10.0, 1000)
    candidate_spikes = _count_spikes(candidate, 10.0, 1000)
    default_spikes = _count_spikes(default, 10.0, 1000)

    assert baseline_spikes == default_spikes
    assert baseline_spikes >= 5
    assert candidate_spikes >= 5
    assert abs(candidate_spikes - baseline_spikes) <= 5


def test_hodgkin_huxley_rk4_path_stays_finite_and_tracks_baseline():
    baseline = HodgkinHuxleyNeuron(dt=0.01, integrator="baseline_euler")
    candidate = HodgkinHuxleyNeuron(dt=0.01, integrator="rk4")

    baseline_spikes = _count_spikes(baseline, 10.0, 1000)
    candidate_spikes = _count_spikes(candidate, 10.0, 1000)

    for value in [candidate.v, candidate.m, candidate.h, candidate.n]:
        assert np.isfinite(value)
    assert baseline_spikes > 0
    assert candidate_spikes > 0
    assert abs(candidate_spikes - baseline_spikes) <= 20
    assert abs(candidate.v - baseline.v) < 15.0


def test_adex_rk4_path_stays_finite_and_tracks_baseline():
    baseline = AdExNeuron(dt=0.1, integrator="baseline_euler")
    candidate = AdExNeuron(dt=0.1, integrator="rk4")

    baseline_spikes = _count_spikes(baseline, 500.0, 3000)
    candidate_spikes = _count_spikes(candidate, 500.0, 3000)

    assert np.isfinite(candidate.v)
    assert np.isfinite(candidate.w)
    assert baseline_spikes > 0
    assert candidate_spikes > 0
    assert abs(candidate_spikes - baseline_spikes) <= 10
    assert abs(candidate.w - baseline.w) < 20.0


def test_default_integrators_match_historical_paths():
    hh_default = HodgkinHuxleyNeuron(dt=0.01)
    hh_baseline = HodgkinHuxleyNeuron(dt=0.01, integrator="baseline_euler")
    adex_default = AdExNeuron(dt=0.1)
    adex_baseline = AdExNeuron(dt=0.1, integrator="baseline_euler")

    hh_default_spikes = _count_spikes(hh_default, 10.0, 200)
    hh_baseline_spikes = _count_spikes(hh_baseline, 10.0, 200)
    adex_default_spikes = _count_spikes(adex_default, 500.0, 2000)
    adex_baseline_spikes = _count_spikes(adex_baseline, 500.0, 2000)

    assert hh_default_spikes == hh_baseline_spikes
    assert adex_default_spikes == adex_baseline_spikes


def test_izhikevich_noise_injection_is_reachable_on_both_paths():
    """``noise_std > 0`` must perturb membrane on both integrator paths."""
    for integrator in ("baseline_half_euler", "rk4"):
        neuron = SCIzhikevichNeuron(noise_std=0.5, dt=0.5, seed=42, integrator=integrator)
        v0 = neuron.v
        neuron.step(5.0)
        assert neuron.v != v0


def test_izhikevich_get_state_reflects_running_v_and_u():
    neuron = SCIzhikevichNeuron(noise_std=0.0, dt=0.5)
    state = neuron.get_state()
    assert set(state) == {"v", "u"}
    assert state["v"] == float(neuron.v)
    assert state["u"] == float(neuron.u)
