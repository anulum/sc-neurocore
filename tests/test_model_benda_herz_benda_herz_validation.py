# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBendaHerzValidation from former test_model_benda_herz.py

"""Focused suite: TestBendaHerzValidation from former test_model_benda_herz.py."""

from __future__ import annotations

from tests.model_benda_herz_support import *  # noqa: F403


class TestBendaHerzValidation:
    @pytest.mark.parametrize("a", [-1.0, np.nan, np.inf, -np.inf])
    def test_rejects_negative_or_non_finite_adaptation_state(self, a: float):
        with pytest.raises(ValueError, match="a"):
            BendaHerzNeuron(a=a)

    @pytest.mark.parametrize("field", ["f_max", "beta", "tau_a", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_scale_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            BendaHerzNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["i_half", "delta_a"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_threshold_and_adaptation_gain(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            BendaHerzNeuron(**{field: value})

    def test_rejects_negative_adaptation_gain(self):
        with pytest.raises(ValueError, match="delta_a"):
            BendaHerzNeuron(delta_a=-1.0)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = BendaHerzNeuron(a=0.5)
        before = n.a
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.a == before

    def test_rejects_non_finite_adaptation_update_before_state_mutation(self):
        n = BendaHerzNeuron(f_max=1.0e-306, delta_a=1.0e308, dt=1.0e308, a=0.5)
        before = n.a

        with pytest.raises(ValueError, match="adaptation RK4"):
            n.step(100.0)

        assert n.a == before

    @pytest.mark.parametrize("seed", [np.nan, np.inf, -1, True, 2**64])
    def test_rejects_invalid_seed(self, seed):
        with pytest.raises(ValueError, match="seed"):
            BendaHerzNeuron(seed=seed)
