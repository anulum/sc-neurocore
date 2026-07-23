# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestYamadaValidation from former test_model_yamada.py

"""Focused suite: TestYamadaValidation from former test_model_yamada.py."""

from __future__ import annotations

from tests.model_yamada_support import *  # noqa: F403

class TestYamadaValidation:
    @pytest.mark.parametrize(
        "field",
        [
            "v",
            "n",
            "q",
            "g_na",
            "g_k",
            "g_q",
            "g_l",
            "e_na",
            "e_k",
            "e_q",
            "e_l",
            "tau_q",
            "dt",
            "v_threshold",
        ],
    )
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            YamadaNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["g_na", "g_k", "g_q", "g_l"])
    def test_rejects_negative_conductance(self, field: str):
        with pytest.raises(ValueError, match=field):
            YamadaNeuron(**{field: -0.1})

    @pytest.mark.parametrize("field", ["tau_q", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_rejects_non_positive_timescale(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            YamadaNeuron(**{field: value})

    @pytest.mark.parametrize(
        ("field", "value"), [("n", -0.01), ("n", 1.01), ("q", -0.01), ("q", 1.01)]
    )
    def test_rejects_gates_outside_unit_interval(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            YamadaNeuron(**{field: value})

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = YamadaNeuron(v=-55.0, n=0.2, q=0.1)
        before = (n.v, n.n, n.q)

        with pytest.raises(ValueError, match="current"):
            n.step(current)

        assert (n.v, n.n, n.q) == before

    def test_rejects_non_finite_candidate_update_before_state_mutation(self):
        n = YamadaNeuron(v=-55.0, n=0.2, q=0.1, dt=1.0e308)
        before = (n.v, n.n, n.q)

        with pytest.raises(ValueError, match="Yamada RK4"):
            n.step(1.0e308)

        assert (n.v, n.n, n.q) == before
