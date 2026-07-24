# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLNMValidation from former test_model_lnm.py

"""Focused suite: TestLNMValidation from former test_model_lnm.py."""

from __future__ import annotations

from tests.model_lnm_support import *  # noqa: F403


class TestLNMValidation:
    @pytest.mark.parametrize(
        "field",
        ["v", "alpha", "beta", "gamma", "v_reset", "f_shift"],
    )
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_state_and_trainable_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            LearnableNeuronModel(**{field: value})

    @pytest.mark.parametrize("field", ["v_threshold", "f_slope"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_rejects_non_positive_or_non_finite_physical_scales(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            LearnableNeuronModel(**{field: value})

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = LearnableNeuronModel(v=0.25)
        before = n.v
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.v == before
