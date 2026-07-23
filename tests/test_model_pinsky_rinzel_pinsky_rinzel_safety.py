# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPinskyRinzelSafety from former test_model_pinsky_rinzel.py

"""Focused suite: TestPinskyRinzelSafety from former test_model_pinsky_rinzel.py."""

from __future__ import annotations

from tests.model_pinsky_rinzel_support import *  # noqa: F403

class TestPinskyRinzelSafety:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("dt", 0.0),
            ("cm", 0.0),
            ("gc", 0.0),
            ("p", 0.0),
            ("p", 1.0),
            ("g_na", 0.0),
            ("g_kdr", 0.0),
            ("g_ca", 0.0),
            ("g_kahp", 0.0),
            ("g_kc", 0.0),
            ("g_l", 0.0),
            ("h", -0.01),
            ("n", 1.01),
            ("s", float("nan")),
            ("c", 1.01),
            ("q", 1.01),
            ("ca", -0.01),
        ],
    )
    def test_rejects_invalid_configuration(self, field: str, value: float):
        with pytest.raises(ValueError):
            PinskyRinzelNeuron(**{field: value})

    def test_rejects_runtime_parameter_corruption_before_mutation(self):
        n = PinskyRinzelNeuron()
        n.p = 1.0
        before = (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q, n.ca)
        with pytest.raises(ValueError):
            n.step(30.0)
        assert (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q, n.ca) == before

    def test_rejects_non_finite_input_before_mutation(self):
        n = PinskyRinzelNeuron()
        before = (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q, n.ca)
        with pytest.raises(ValueError):
            n.step(float("nan"))
        assert (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q, n.ca) == before
        with pytest.raises(ValueError):
            n.step(0.0, float("inf"))
        assert (n.v_s, n.v_d, n.h, n.n, n.s, n.c, n.q, n.ca) == before

    def test_extreme_timestep_fails_closed(self):
        with pytest.raises(FloatingPointError):
            PinskyRinzelNeuron(dt=10.0).step(30.0)

    def test_validate_candidate_rejects_non_finite_state(self):
        with pytest.raises(FloatingPointError):
            PinskyRinzelNeuron._validate_candidate(
                (float("nan"), -60.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2)
            )

    def test_validate_candidate_clamps_gates_and_calcium(self):
        v_s, v_d, h, n, s, c, q, ca = PinskyRinzelNeuron._validate_candidate(
            (-60.0, -60.0, 1.5, -0.2, 0.5, 2.0, -0.3, -4.0)
        )
        assert (h, n, s, c, q, ca) == (1.0, 0.0, 0.5, 1.0, 0.0, 0.0)
