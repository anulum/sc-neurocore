# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPinskyRinzelRateBranches from former test_model_pinsky_rinzel.py

"""Focused suite: TestPinskyRinzelRateBranches from former test_model_pinsky_rinzel.py."""

from __future__ import annotations

from tests.model_pinsky_rinzel_support import *  # noqa: F403


class TestPinskyRinzelRateBranches:
    @pytest.mark.parametrize("v_s", [-46.9, -19.9, -24.9])
    def test_somatic_rate_singularities_are_finite(self, v_s: float):
        """αm/βm/αn evaluate their removable limit at the singular voltage."""
        n = PinskyRinzelNeuron(v_s=v_s)
        n.step(0.0)
        assert np.isfinite(n.v_s)

    def test_dendritic_beta_s_singularity_is_finite(self):
        n = PinskyRinzelNeuron(v_d=-8.9)
        n.step(0.0)
        assert np.isfinite(n.v_d)

    def test_depolarised_dendrite_uses_alternate_c_branch(self):
        """Vd > −10 mV selects the βc = 0 branch of the K-C activation rate."""
        n = PinskyRinzelNeuron(v_d=0.0)
        n.step(0.0)
        assert np.isfinite(n.v_d)
