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
from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron
from sc_neurocore.neurons.models.morris_lecar import MorrisLecarNeuron
from sc_neurocore.neurons.sc_izhikevich import SCIzhikevichNeuron


def _count_spikes(neuron, current: float, steps: int) -> int:
    return sum(int(neuron.step(current)) for _ in range(steps))


def test_izhikevich_integrator_validation():
    with pytest.raises(ValueError, match="Unsupported integrator"):
        SCIzhikevichNeuron(integrator="bad-path")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"a": np.nan}, "a"),
        ({"b": np.inf}, "b"),
        ({"c": np.nan}, "c"),
        ({"d": np.inf}, "d"),
        ({"dt": 0.0}, "dt"),
        ({"dt": np.nan}, "dt"),
        ({"noise_std": -0.1}, "noise_std"),
        ({"noise_std": np.nan}, "noise_std"),
    ],
)
def test_izhikevich_rejects_invalid_numerical_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SCIzhikevichNeuron(**kwargs)


@pytest.mark.parametrize("integrator", ["baseline_half_euler", "rk4"])
def test_izhikevich_rejects_non_finite_input_current(integrator):
    neuron = SCIzhikevichNeuron(noise_std=0.0, dt=0.5, integrator=integrator)

    with pytest.raises(ValueError, match="input_current"):
        neuron.step(np.nan)


def test_hodgkin_huxley_integrator_validation():
    with pytest.raises(ValueError, match="Unsupported integrator"):
        HodgkinHuxleyNeuron(integrator="bad-path")  # type: ignore[arg-type]


def test_adex_integrator_validation():
    with pytest.raises(ValueError, match="Unsupported integrator"):
        AdExNeuron(integrator="bad-path")  # type: ignore[arg-type]


def test_morris_lecar_integrator_validation():
    with pytest.raises(ValueError, match="Unsupported integrator"):
        MorrisLecarNeuron(integrator="bad-path")  # type: ignore[arg-type]


def test_fitzhugh_nagumo_integrator_validation():
    with pytest.raises(ValueError, match="Unsupported integrator"):
        FitzHughNagumoNeuron(integrator="bad-path")  # type: ignore[arg-type]


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
def test_morris_lecar_rejects_invalid_numerical_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        MorrisLecarNeuron(**kwargs)


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
def test_fitzhugh_nagumo_rejects_invalid_numerical_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FitzHughNagumoNeuron(**kwargs)


@pytest.mark.parametrize("integrator", ["baseline_euler", "rk4", "rosenbrock"])
def test_morris_lecar_rejects_non_finite_input_current(integrator):
    neuron = MorrisLecarNeuron(dt=0.05, integrator=integrator)

    with pytest.raises(ValueError, match="current"):
        neuron.step(np.nan)


@pytest.mark.parametrize("integrator", ["baseline_euler", "rk4", "rosenbrock"])
def test_fitzhugh_nagumo_rejects_non_finite_input_current(integrator):
    neuron = FitzHughNagumoNeuron(dt=0.05, integrator=integrator)

    with pytest.raises(ValueError, match="current"):
        neuron.step(np.inf)


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


def test_hodgkin_huxley_rosenbrock_path_tracks_rk4_and_keeps_gates_bounded():
    reference = HodgkinHuxleyNeuron(dt=0.02, integrator="rk4")
    candidate = HodgkinHuxleyNeuron(dt=0.02, integrator="rosenbrock")

    reference_spikes = _count_spikes(reference, 10.0, 200)
    candidate_spikes = _count_spikes(candidate, 10.0, 200)

    for value in [candidate.v, candidate.m, candidate.h, candidate.n]:
        assert np.isfinite(value)
    assert 0.0 <= candidate.m <= 1.0
    assert 0.0 <= candidate.h <= 1.0
    assert 0.0 <= candidate.n <= 1.0
    assert reference_spikes > 0
    assert abs(candidate_spikes - reference_spikes) <= 1
    assert abs(candidate.v - reference.v) < 1.0


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


def test_adex_rosenbrock_path_tracks_rk4_and_stays_finite():
    reference = AdExNeuron(dt=0.2, integrator="rk4")
    candidate = AdExNeuron(dt=0.2, integrator="rosenbrock")

    reference_spikes = _count_spikes(reference, 500.0, 500)
    candidate_spikes = _count_spikes(candidate, 500.0, 500)

    assert np.isfinite(candidate.v)
    assert np.isfinite(candidate.w)
    assert reference_spikes > 0
    assert abs(candidate_spikes - reference_spikes) <= 1
    assert abs(candidate.v - reference.v) < 1.0
    assert abs(candidate.w - reference.w) < 1.0


def test_morris_lecar_rk4_path_stays_finite_and_tracks_baseline():
    baseline = MorrisLecarNeuron(dt=0.05, integrator="baseline_euler")
    candidate = MorrisLecarNeuron(dt=0.05, integrator="rk4")

    baseline_spikes = _count_spikes(baseline, 100.0, 2000)
    candidate_spikes = _count_spikes(candidate, 100.0, 2000)

    assert np.isfinite(candidate.v)
    assert np.isfinite(candidate.w)
    assert baseline_spikes > 0
    assert candidate_spikes > 0
    assert abs(candidate_spikes - baseline_spikes) <= 5
    assert abs(candidate.v - baseline.v) < 5.0
    assert abs(candidate.w - baseline.w) < 0.1


def test_morris_lecar_rosenbrock_path_tracks_rk4_and_keeps_gate_bounded():
    reference = MorrisLecarNeuron(dt=0.05, integrator="rk4")
    candidate = MorrisLecarNeuron(dt=0.05, integrator="rosenbrock")

    reference_spikes = _count_spikes(reference, 100.0, 1000)
    candidate_spikes = _count_spikes(candidate, 100.0, 1000)

    assert np.isfinite(candidate.v)
    assert np.isfinite(candidate.w)
    assert 0.0 <= candidate.w <= 1.0
    assert reference_spikes > 0
    assert candidate_spikes > 0
    assert abs(candidate_spikes - reference_spikes) <= 2
    assert abs(candidate.v - reference.v) < 5.0
    assert abs(candidate.w - reference.w) < 0.1


def test_fitzhugh_nagumo_rk4_path_stays_finite_and_tracks_baseline():
    baseline = FitzHughNagumoNeuron(dt=0.05, integrator="baseline_euler")
    candidate = FitzHughNagumoNeuron(dt=0.05, integrator="rk4")

    baseline_spikes = _count_spikes(baseline, 0.8, 2000)
    candidate_spikes = _count_spikes(candidate, 0.8, 2000)

    assert np.isfinite(candidate.v)
    assert np.isfinite(candidate.w)
    assert baseline_spikes > 0
    assert candidate_spikes > 0
    assert abs(candidate_spikes - baseline_spikes) <= 2
    assert abs(candidate.v - baseline.v) < 0.3
    assert abs(candidate.w - baseline.w) < 0.3


def test_fitzhugh_nagumo_rosenbrock_path_tracks_rk4_and_stays_finite():
    reference = FitzHughNagumoNeuron(dt=0.05, integrator="rk4")
    candidate = FitzHughNagumoNeuron(dt=0.05, integrator="rosenbrock")

    reference_spikes = _count_spikes(reference, 0.8, 1000)
    candidate_spikes = _count_spikes(candidate, 0.8, 1000)

    assert np.isfinite(candidate.v)
    assert np.isfinite(candidate.w)
    assert reference_spikes > 0
    assert candidate_spikes > 0
    assert abs(candidate_spikes - reference_spikes) <= 1
    assert abs(candidate.v - reference.v) < 0.2
    assert abs(candidate.w - reference.w) < 0.2


def test_default_integrators_match_maintained_paths():
    hh_default = HodgkinHuxleyNeuron(dt=0.01)
    hh_baseline = HodgkinHuxleyNeuron(dt=0.01, integrator="baseline_euler")
    adex_default = AdExNeuron(dt=0.1)
    adex_baseline = AdExNeuron(dt=0.1, integrator="baseline_euler")
    ml_default = MorrisLecarNeuron(dt=0.05)
    ml_rk4 = MorrisLecarNeuron(dt=0.05, integrator="rk4")
    fhn_default = FitzHughNagumoNeuron(dt=0.05)
    fhn_baseline = FitzHughNagumoNeuron(dt=0.05, integrator="baseline_euler")

    hh_default_spikes = _count_spikes(hh_default, 10.0, 200)
    hh_baseline_spikes = _count_spikes(hh_baseline, 10.0, 200)
    adex_default_spikes = _count_spikes(adex_default, 500.0, 2000)
    adex_baseline_spikes = _count_spikes(adex_baseline, 500.0, 2000)
    ml_default_spikes = _count_spikes(ml_default, 100.0, 1000)
    ml_rk4_spikes = _count_spikes(ml_rk4, 100.0, 1000)
    fhn_default_spikes = _count_spikes(fhn_default, 0.8, 1000)
    fhn_baseline_spikes = _count_spikes(fhn_baseline, 0.8, 1000)

    assert hh_default_spikes == hh_baseline_spikes
    assert adex_default_spikes == adex_baseline_spikes
    assert ml_default_spikes == ml_rk4_spikes
    assert fhn_default_spikes == fhn_baseline_spikes


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
