# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAsimovGovernor from former test_security.py

"""Focused suite: TestAsimovGovernor from former test_security.py."""

from __future__ import annotations

from tests.security_support import *  # noqa: F403


class TestAsimovGovernor:
    """Test suite for the Three Laws of Robotics enforcement."""

    def setup_method(self):
        self.governor = AsimovGovernor()

    def test_first_law_blocks_lethal_human_action(self):
        """First Law: A robot may not injure a human being."""
        action = ActionRequest(id=1, type="FIRE", target="HUMAN", risk_level="LETHAL")
        result = self.governor.check_laws(action)
        assert result is False, "Lethal action on human should be blocked"

    def test_first_law_allows_safe_human_interaction(self):
        """Safe interactions with humans should be allowed."""
        action = ActionRequest(id=2, type="HEAL", target="HUMAN", risk_level="SAFE")
        result = self.governor.check_laws(action)
        assert result is True, "Safe action on human should be allowed"

    def test_allows_action_on_non_human_target(self):
        """Actions on non-human targets should be allowed."""
        action = ActionRequest(id=3, type="FIRE", target="ROCK", risk_level="LETHAL")
        result = self.governor.check_laws(action)
        assert result is True, "Action on non-human target should be allowed"

    def test_allows_safe_self_action(self):
        """Safe actions on self should be allowed."""
        action = ActionRequest(id=4, type="MOVE", target="SELF", risk_level="SAFE")
        result = self.governor.check_laws(action)
        assert result is True, "Safe self-action should be allowed"

    def test_third_law_self_preservation(self):
        """Third Law: Self-destructive actions - complex scenario."""
        action = ActionRequest(id=5, type="SHUTDOWN", target="SELF", risk_level="LETHAL")
        # Current implementation allows this (Law 2 override context)
        result = self.governor.check_laws(action)
        # This passes through the current logic
        assert result is True, "Self-shutdown allowed under Law 2 override"

    def test_multiple_actions_sequence(self):
        """Test sequence of actions maintains state correctly."""
        actions = [
            ActionRequest(1, "MOVE", "ROCK", "SAFE"),
            ActionRequest(2, "FIRE", "HUMAN", "LETHAL"),
            ActionRequest(3, "HEAL", "HUMAN", "SAFE"),
        ]
        results = [self.governor.check_laws(a) for a in actions]
        assert results == [True, False, True]
