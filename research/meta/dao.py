# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class Proposal:
    id: int
    action: str
    proposer_id: str
    votes_for: float = 0.0
    votes_against: float = 0.0
    status: str = "Active"


@dataclass
class AgentDAO:
    """
    Decentralized Autonomous Organization for Agent Governance.
    Uses 'Proof of Compute' as voting weight.
    """

    agent_id: str
    compute_credits: float = 10.0
    ledger: List[Proposal] = field(default_factory=list)

    def create_proposal(self, action: str) -> int:
        pid = len(self.ledger)
        prop = Proposal(pid, action, self.agent_id)
        self.ledger.append(prop)
        logger.info("DAO: Proposal %d created by %s: '%s'", pid, self.agent_id, action)
        return pid

    def vote(self, proposal_id: int, approve: bool):
        """
        Cast vote weighted by credits.
        """
        if proposal_id >= len(self.ledger):
            return

        prop = self.ledger[proposal_id]
        if prop.status != "Active":
            return

        weight = self.compute_credits
        if approve:
            prop.votes_for += weight
        else:
            prop.votes_against += weight

        logger.info(
            "DAO: %s voted %s on #%d (Weight: %s)",
            self.agent_id,
            "YES" if approve else "NO",
            proposal_id,
            weight,
        )

    def finalize_proposal(self, proposal_id: int) -> bool:
        """
        Tally votes.
        """
        prop = self.ledger[proposal_id]
        total_votes = prop.votes_for + prop.votes_against

        if total_votes == 0:
            prop.status = "Failed"
            return False

        if prop.votes_for > prop.votes_against:
            prop.status = "Passed"
            logger.info("DAO: Proposal #%d PASSED.", proposal_id)
            return True
        else:
            prop.status = "Rejected"
            logger.info("DAO: Proposal #%d REJECTED.", proposal_id)
            return False
