# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPospischilValidation from former test_model_pospischil.py

"""Focused suite: TestPospischilValidation from former test_model_pospischil.py."""

from __future__ import annotations

from tests.model_pospischil_support import *  # noqa: F403

class TestPospischilValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"g_na": -1.0},
            {"g_kd": 0.0},
            {"g_l": -0.1},
            {"c_m": 0.0},
            {"dt": 0.0},
            {"dt": -0.025},
            {"g_m": -0.01},
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs: dict[str, float]):
        with pytest.raises(ValueError):
            PospischilNeuron(**kwargs)

    def test_accepts_zero_m_current_conductance(self):
        # The fast-spiking variant legitimately sets g_m = 0.
        assert PospischilNeuron(g_m=0.0).g_m == 0.0

    @pytest.mark.parametrize("field", ["v", "vt", "e_na", "e_k"])
    def test_rejects_non_finite_field(self, field: str):
        with pytest.raises(ValueError, match="must be finite"):
            PospischilNeuron(**{field: float("nan")})

    def test_rejects_boolean_field(self):
        with pytest.raises(ValueError, match="must be finite"):
            PospischilNeuron(v=True)  # type: ignore[arg-type]

    def test_rejects_non_finite_current(self):
        n = PospischilNeuron()
        with pytest.raises(ValueError, match="must be finite"):
            n.step(float("inf"))

    def test_runtime_validation_catches_corrupted_state(self):
        n = PospischilNeuron()
        n.dt = -1.0  # corrupt a positive parameter after construction
        with pytest.raises(ValueError, match="dt must be positive"):
            n.step(0.0)

    def test_non_finite_candidate_fails_closed(self):
        # A colossal stimulus overflows the membrane derivative; the candidate
        # guard raises rather than committing a non-finite state.
        n = PospischilNeuron()
        with pytest.raises((FloatingPointError, OverflowError)):
            for _ in range(20):
                n.step(1e308)
