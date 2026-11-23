# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRuleConstraints from former test_meta_plasticity.py

"""Focused suite: TestRuleConstraints from former test_meta_plasticity.py."""

from __future__ import annotations

from sc_neurocore.meta_plasticity.meta_plasticity import (
    PlasticityRuleSet,
    RuleConstraints,
)


class TestRuleConstraints:
    """Valid ranges of plasticity rules."""

    def test_valid_rules(self) -> None:
        """Default rules are valid."""
        rc = RuleConstraints()
        rs = PlasticityRuleSet()
        assert rc.is_valid(rs)

    def test_invalid_lr(self) -> None:
        """An out-of-range learning rate is invalid."""
        rc = RuleConstraints()
        rs = PlasticityRuleSet()
        rs.stdp.lr = 999.0
        assert not rc.is_valid(rs)

    def test_invalid_tau_with_valid_lr(self) -> None:
        """An out-of-range time constant is invalid."""
        # lr passes its range check so validation proceeds to the tau check,
        # which an out-of-range tau_plus fails.
        rc = RuleConstraints()
        rs = PlasticityRuleSet()
        rs.stdp.lr = 0.01
        rs.stdp.tau_plus = 1000.0
        assert not rc.is_valid(rs)

    def test_enforce_clamps(self) -> None:
        """Enforcing the constraints clamps rate, time constant and length."""
        rc = RuleConstraints()
        rs = PlasticityRuleSet()
        rs.stdp.lr = 999.0
        rs.stdp.tau_plus = 0.001
        rs.bitstream.length = 1
        rc.enforce(rs)
        assert rs.stdp.lr <= 0.1
        assert rs.stdp.tau_plus >= 1.0
        assert rs.bitstream.length >= 32
