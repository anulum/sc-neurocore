# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestContextRuleBank from former test_meta_plasticity.py

"""Focused suite: TestContextRuleBank from former test_meta_plasticity.py."""

from __future__ import annotations

from meta_plasticity_support import *  # noqa: F403


class TestContextRuleBank:
    def test_store_and_switch(self):
        bank = ContextRuleBank()
        rs = PlasticityRuleSet()
        rs.stdp.lr = 0.05
        bank.store("task_A", rs)
        restored = bank.switch("task_A")
        assert restored is not None
        assert restored.stdp.lr == 0.05

    def test_missing_context(self):
        bank = ContextRuleBank()
        assert bank.switch("missing") is None

    def test_contexts_list(self):
        bank = ContextRuleBank()
        bank.store("A", PlasticityRuleSet())
        bank.store("B", PlasticityRuleSet())
        assert set(bank.contexts()) == {"A", "B"}
        assert bank.num_contexts == 2
