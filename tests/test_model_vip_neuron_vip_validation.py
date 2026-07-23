# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVIPValidation from former test_model_vip_neuron.py

"""Focused suite: TestVIPValidation from former test_model_vip_neuron.py."""

from __future__ import annotations

from tests.model_vip_neuron_support import *  # noqa: F403

class TestVIPValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"g_na": -1.0},
            {"g_k": 0.0},
            {"g_l": -0.1},
            {"c_m": 0.0},
            {"dt": 0.0},
            {"g_a": -1.0},
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs: dict[str, float]):
        with pytest.raises(ValueError):
            VIPNeuron(**kwargs)

    def test_accepts_zero_a_conductance(self):
        assert VIPNeuron(g_a=0.0).g_a == 0.0

    @pytest.mark.parametrize("field", ["v", "e_na", "e_k", "e_l"])
    def test_rejects_non_finite_field(self, field: str):
        with pytest.raises(ValueError, match="must be finite"):
            VIPNeuron(**{field: float("nan")})

    def test_rejects_boolean_field(self):
        with pytest.raises(ValueError, match="must be finite"):
            VIPNeuron(v=True)  # type: ignore[arg-type]

    def test_rejects_non_finite_current(self):
        with pytest.raises(ValueError, match="must be finite"):
            VIPNeuron().step(float("inf"))

    def test_runtime_validation_catches_corrupted_state(self):
        n = VIPNeuron()
        n.dt = -1.0
        with pytest.raises(ValueError, match="dt must be positive"):
            n.step(0.0)

    def test_non_finite_candidate_fails_closed(self):
        n = VIPNeuron()
        with pytest.raises((FloatingPointError, OverflowError)):
            for _ in range(60):
                n.step(1e308)
