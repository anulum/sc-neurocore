# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSSTSingularRates from former test_model_sst_neuron.py

"""Focused suite: TestSSTSingularRates from former test_model_sst_neuron.py."""

from __future__ import annotations

from tests.model_sst_neuron_support import *  # noqa: F403

class TestSSTSingularRates:
    def test_alpha_singular_returns_limit_at_singularity(self):
        from sc_neurocore.neurons.models.sst_neuron import _alpha_singular

        # At numerator 0 the closed-form L'Hôpital limit (equal to slope) is used.
        assert _alpha_singular(0.0, -4.0, -4.0) == -4.0
        assert _alpha_singular(0.0, 5.0, 5.0) == 5.0

    def test_alpha_singular_continuous_across_singularity(self):
        from sc_neurocore.neurons.models.sst_neuron import _alpha_singular

        left = _alpha_singular(-1e-7, 5.0, 5.0)
        right = _alpha_singular(1e-7, 5.0, 5.0)
        assert abs(left - 5.0) < 1e-3
        assert abs(right - 5.0) < 1e-3
