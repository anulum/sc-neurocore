# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEWCProtection from former test_meta_plasticity.py

"""Focused suite: TestEWCProtection from former test_meta_plasticity.py."""

from __future__ import annotations

from meta_plasticity_support import *  # noqa: F403

class TestEWCProtection:
    def test_no_anchor(self):
        ewc = EWCProtection()
        rs = PlasticityRuleSet()
        assert ewc.penalty(rs) == 0.0

    def test_penalty_after_consolidation(self):
        ewc = EWCProtection(importance=100.0)
        rs = PlasticityRuleSet()
        ewc.consolidate(rs)
        modified = rs.copy()
        modified.stdp.lr = 0.05
        assert ewc.penalty(modified) > 0

    def test_regularise_pulls_back(self):
        ewc = EWCProtection(importance=10000.0)
        rs = PlasticityRuleSet()
        ewc.consolidate(rs)
        modified = rs.copy()
        modified.stdp.lr = 0.1
        regularised = ewc.regularise(modified, max_penalty=0.01)
        assert abs(regularised.stdp.lr - rs.stdp.lr) < abs(modified.stdp.lr - rs.stdp.lr)

    def test_regularise_without_anchor_is_identity(self):
        ewc = EWCProtection()
        rs = PlasticityRuleSet()
        assert ewc.regularise(rs) is rs

    def test_regularise_below_threshold_is_identity(self):
        ewc = EWCProtection()
        rs = PlasticityRuleSet()
        ewc.consolidate(rs)  # anchor == current, so penalty is 0 <= max_penalty
        assert ewc.regularise(rs, max_penalty=10.0) is rs
