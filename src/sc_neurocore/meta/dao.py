
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List

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
        print(f"DAO: Proposal {pid} created by {self.agent_id}: '{action}'")
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
            
        print(f"DAO: {self.agent_id} voted {'YES' if approve else 'NO'} on #{proposal_id} (Weight: {weight})")
        
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
            print(f"DAO: Proposal #{proposal_id} PASSED.")
            return True
        else:
            prop.status = "Rejected"
            print(f"DAO: Proposal #{proposal_id} REJECTED.")
            return False
