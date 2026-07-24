# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPospischilAlphaSingular from former test_model_pospischil.py

"""Focused suite: TestPospischilAlphaSingular from former test_model_pospischil.py."""

from __future__ import annotations

from tests.model_pospischil_support import *  # noqa: F403


class TestPospischilAlphaSingular:
    def test_limit_returned_at_singularity(self):
        from sc_neurocore.neurons.models.pospischil import _alpha_singular

        assert _alpha_singular(0.0, -4.0, -4.0) == -4.0
        assert _alpha_singular(5e-7, 5.0, 5.0) == 5.0

    def test_regular_branch_matches_hodgkin_huxley_ratio(self):
        from sc_neurocore.neurons.models.pospischil import _alpha_singular

        expected = 2.0 / (np.exp(2.0 / -4.0) - 1.0)
        assert _alpha_singular(2.0, -4.0, -4.0) == pytest.approx(expected)

    def test_neuron_runs_through_gating_singularity(self):
        # V_T + 13 = -43.2 puts the m-activation numerator exactly on its
        # removable singularity at the start of a sub-step.
        n = PospischilNeuron(v=-43.2)
        n.step(0.0)
        assert np.isfinite(n.v)
