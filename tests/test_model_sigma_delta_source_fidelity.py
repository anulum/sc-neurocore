# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

import math

import pytest

from sc_neurocore.neurons.models.sigma_delta import SigmaDeltaNeuron


def test_one_step_matches_disclosed_sampled_apsdm_equations() -> None:
    neuron = SigmaDeltaNeuron(sigma=0.4, reconstruction=0.2, delta=1.0, dt=0.1)
    expected_sigma = 0.4 + 0.1 * 2.0
    expected_reconstruction = 0.2 * math.exp(-0.1 / 10.0)
    assert neuron.step(2.0) == 0
    assert neuron.sigma == expected_sigma
    assert neuron.reconstruction == expected_reconstruction


def test_upper_threshold_is_unipolar_and_feedback_adds_delta() -> None:
    neuron = SigmaDeltaNeuron(sigma=0.49, reconstruction=0.0)
    assert neuron.step(0.2) == 1
    assert neuron.sigma == pytest.approx(0.51)
    assert neuron.reconstruction == pytest.approx(1.0)
    for _ in range(20):
        assert neuron.step(-10.0) in (0, 1)


def test_invalid_candidate_is_atomic() -> None:
    neuron = SigmaDeltaNeuron(sigma=0.25, reconstruction=0.125)
    before = (neuron.sigma, neuron.reconstruction)
    with pytest.raises(ValueError):
        neuron.step(float("nan"))
    assert (neuron.sigma, neuron.reconstruction) == before


def test_reset_clears_both_dynamic_states() -> None:
    neuron = SigmaDeltaNeuron(sigma=2.0, reconstruction=1.0)
    neuron.reset()
    assert (neuron.sigma, neuron.reconstruction) == (0.0, 0.0)
