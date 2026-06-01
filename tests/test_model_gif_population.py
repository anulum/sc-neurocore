# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GIFPopulationNeuron physics contracts

"""Model-specific contracts for Mensi et al. GIF escape-rate dynamics."""

from __future__ import annotations

import math

from sc_neurocore.neurons.models.gif_population import GIFPopulationNeuron


def test_exact_coupled_subthreshold_reference_point() -> None:
    neuron = GIFPopulationNeuron(v=-68.0, eta=0.4, seed=7)

    assert neuron.step(4.0) == 0

    assert math.isclose(neuron.v, -67.8370206677805, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(neuron.eta, 0.398004991677073, rel_tol=0.0, abs_tol=1e-15)


def test_equal_time_constant_limit_is_finite() -> None:
    neuron = GIFPopulationNeuron(v=-63.0, eta=1.5, tau_m=20.0, tau_eta=20.0)

    assert neuron.step(2.0) == 0

    assert math.isfinite(neuron.v)
    assert math.isclose(neuron.eta, 1.4629648680424988, rel_tol=0.0, abs_tol=1e-15)


def test_forced_spike_resets_voltage_and_adds_decayed_adaptation() -> None:
    neuron = GIFPopulationNeuron(v=-51.0, eta=0.3, theta=-90.0, lambda_0=1.0e9)

    assert neuron.step(0.0) == 1

    assert neuron.v == neuron.v_reset
    assert math.isclose(neuron.eta, 5.298503743757805, rel_tol=0.0, abs_tol=1e-15)


def test_zero_baseline_hazard_never_spikes() -> None:
    neuron = GIFPopulationNeuron(theta=-1000.0, lambda_0=0.0)

    spikes = sum(neuron.step(100.0) for _ in range(128))

    assert spikes == 0
    assert neuron._spike_probability(neuron.v) == 0.0


def test_invalid_input_and_invalid_parameters_preserve_state() -> None:
    neuron = GIFPopulationNeuron(v=-62.0, eta=0.75)
    before = (neuron.v, neuron.eta)

    assert neuron.step(float("nan")) == 0
    assert (neuron.v, neuron.eta) == before

    neuron.tau_m = 0.0
    assert neuron.step(1.0) == 0
    assert (neuron.v, neuron.eta) == before


def test_nonfinite_candidate_preserves_state() -> None:
    neuron = GIFPopulationNeuron(v=1.0e308, v_rest=1.0e308, eta=0.0)
    before = (neuron.v, neuron.eta)

    assert neuron.step(1.0e308) == 0

    assert (neuron.v, neuron.eta) == before


def test_seeded_reproducibility_and_seed_separation() -> None:
    left = GIFPopulationNeuron(seed=123)
    right = GIFPopulationNeuron(seed=123)
    other = GIFPopulationNeuron(seed=999)

    left_train = [left.step(60.0) for _ in range(256)]
    right_train = [right.step(60.0) for _ in range(256)]
    other_train = [other.step(60.0) for _ in range(256)]

    assert left_train == right_train
    zero_seed = GIFPopulationNeuron(seed=0)
    one_seed = GIFPopulationNeuron(seed=1)
    assert [zero_seed.step(60.0) for _ in range(32)] == [one_seed.step(60.0) for _ in range(32)]
    assert left_train != other_train


def test_reset_restores_state_and_rng_sequence() -> None:
    neuron = GIFPopulationNeuron(seed=123, theta=-90.0, lambda_0=1.0e9)

    first = [neuron.step(0.0) for _ in range(3)]
    neuron.reset()
    replay = [neuron.step(0.0) for _ in range(3)]

    assert first == replay
    assert neuron.v == neuron.v_reset
    assert neuron.eta > neuron.eta_increment
