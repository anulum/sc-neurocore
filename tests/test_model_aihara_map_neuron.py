# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Module-specific test: AiharaMapNeuron

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.aihara_map_neuron import AiharaMapNeuron


class TestAiharaMapNeuron:
    def test_defaults_and_binary_step(self):
        n = AiharaMapNeuron()
        assert n.x == 0.0
        assert n.y == 0.0
        assert n.step(0.0) in (0, 1)

    def test_stable_sigmoid_for_extreme_finite_state(self):
        assert AiharaMapNeuron._sigmoid(1.0e308) == 1.0
        assert AiharaMapNeuron._sigmoid(-1.0e308) == 0.0

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("x", np.nan),
            ("y", np.inf),
            ("k_f", -1.0),
            ("k_s", np.nan),
            ("alpha", np.inf),
            ("delta", -1.0),
            ("x_threshold", np.nan),
        ],
    )
    def test_rejects_invalid_numerical_configuration(self, field: str, value: float):
        with pytest.raises(ValueError):
            AiharaMapNeuron(**{field: value})

    def test_rejects_non_finite_current_before_state_mutation(self):
        n = AiharaMapNeuron()
        before = (n.x, n.y)
        with pytest.raises(ValueError, match="current"):
            n.step(np.nan)
        assert (n.x, n.y) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self):
        n = AiharaMapNeuron()
        n.y = np.inf
        before = (n.x, n.y)
        with pytest.raises(FloatingPointError, match="state"):
            n.step(0.0)
        assert (n.x, n.y) == before

    def test_rejects_non_finite_candidate_before_state_mutation(self):
        n = AiharaMapNeuron(x=1.0e308, y=0.0, delta=1.0e308)
        before = (n.x, n.y)
        with pytest.raises(FloatingPointError, match="candidate"):
            n.step(0.0)
        assert (n.x, n.y) == before

    def test_clamps_finite_candidates_to_documented_box(self):
        n = AiharaMapNeuron()
        n.step(1.0e6)
        assert n.x == 10.0
        assert -10.0 <= n.y <= 10.0

    def test_reset_restores_state(self):
        n = AiharaMapNeuron()
        n.step(2.0)
        n.reset()
        assert (n.x, n.y) == (0.0, 0.0)
