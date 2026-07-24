# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlasticityRuleSet from former test_meta_plasticity.py

"""Focused suite: TestPlasticityRuleSet from former test_meta_plasticity.py."""

from __future__ import annotations

from meta_plasticity_support import *  # noqa: F403


class TestPlasticityRuleSet:
    def test_to_vector(self):
        rs = PlasticityRuleSet()
        v = rs.to_vector()
        assert len(v) == rs.vector_dim

    def test_from_vector_roundtrip(self):
        rs = PlasticityRuleSet()
        v = rs.to_vector()
        rs2 = PlasticityRuleSet.from_vector(v)
        assert abs(rs2.stdp.tau_plus - rs.stdp.tau_plus) < 1e-6
        assert abs(rs2.stp.u_base - rs.stp.u_base) < 1e-6

    def test_copy_independent(self):
        rs = PlasticityRuleSet()
        rs2 = rs.copy()
        rs2.stdp.lr = 999.0
        assert rs.stdp.lr != 999.0
