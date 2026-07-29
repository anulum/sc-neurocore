# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Aihara model contracts

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.neurons.models.aihara_map_neuron import AiharaMapNeuron


class TestAiharaMapNeuron:
    def test_source_anchored_defaults_and_first_step(self) -> None:
        neuron = AiharaMapNeuron()
        assert neuron.y == 0.1
        assert neuron.output() == pytest.approx(1.0 / (1.0 + math.exp(-10.0)))
        expected = 0.7 * 0.1 - neuron.output() + 0.3968
        assert neuron.step(0.0) == 0
        assert neuron.y == pytest.approx(expected, rel=0.0, abs=1.0e-15)

    def test_stable_logistic_at_extreme_arguments(self) -> None:
        assert AiharaMapNeuron._logistic(1.0e308, 1.0) == 1.0
        assert AiharaMapNeuron._logistic(-1.0e308, 1.0) == 0.0

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("y", np.nan),
            ("k", -0.1),
            ("k", 1.0),
            ("alpha", 0.0),
            ("bias", np.inf),
            ("epsilon", 0.0),
        ],
    )
    def test_rejects_invalid_configuration(self, field: str, value: float) -> None:
        with pytest.raises((ValueError, FloatingPointError)):
            AiharaMapNeuron(**{field: value})

    def test_rejects_input_and_candidate_without_mutation(self) -> None:
        neuron = AiharaMapNeuron()
        before = neuron.y
        with pytest.raises(ValueError, match="current"):
            neuron.step(np.nan)
        assert neuron.y == before
        neuron.y = 1.7e308
        neuron.k = 0.99
        neuron.bias = 1.7e308
        before = neuron.y
        with pytest.raises(FloatingPointError, match="candidate"):
            neuron.step(0.0)
        assert neuron.y == before

    def test_eq12_is_level_not_crossing(self) -> None:
        neuron = AiharaMapNeuron(y=-0.1, k=0.0, alpha=0.01, bias=0.2)
        assert neuron.step(0.0) == 1
        assert neuron.step(0.0) == 1

    def test_reset_restores_only_source_initial_state(self) -> None:
        neuron = AiharaMapNeuron(k=0.6, alpha=2.0, bias=0.5, epsilon=0.015)
        neuron.step(0.0)
        neuron.reset()
        assert neuron.y == 0.1
        assert (neuron.k, neuron.alpha, neuron.bias, neuron.epsilon) == (0.6, 2.0, 0.5, 0.015)
