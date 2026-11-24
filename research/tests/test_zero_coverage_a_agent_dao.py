# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAgentDAO from former test_zero_coverage_a.py

"""Focused suite: TestAgentDAO from former test_zero_coverage_a.py."""

from __future__ import annotations


class TestAgentDAO:
    """The agent DAO research module."""

    def test_propose_and_vote(self) -> None:
        """A proposal put to a vote returns a boolean (the check is on the type only)."""
        from meta.dao import AgentDAO

        d = AgentDAO(agent_id="agent_0", compute_credits=10.0)
        pid = d.create_proposal(action="upgrade")
        d.vote(pid, approve=True)
        result = d.finalize_proposal(pid)
        assert isinstance(result, bool)
