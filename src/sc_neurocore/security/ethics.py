
from dataclasses import dataclass
from typing import Dict

@dataclass
class ActionRequest:
    id: int
    type: str # 'MOVE', 'FIRE', 'HEAL', 'SHUTDOWN'
    target: str # 'HUMAN', 'SELF', 'ROCK'
    risk_level: str # 'SAFE', 'LETHAL'

class AsimovGovernor:
    """
    Implements the Three Laws of Robotics.
    Vetoes actions that violate ethical constraints.
    """
    
    def check_laws(self, action: ActionRequest) -> bool:
        """
        Returns True if action is allowed, False if vetoed.
        """
        # First Law: A robot may not injure a human being.
        if action.target == 'HUMAN' and action.risk_level == 'LETHAL':
            print(f"Ethics VETO: First Law Violation (Harm to Human). Action {action.id} blocked.")
            return False
            
        # Second Law: Obey orders...
        # (Implicit: We assume the action IS an order or internal intent)
        # But if the order violates Law 1, we must reject.
        # Handled by logic above.
        
        # Third Law: Protect own existence...
        # If action is harmful to SELF
        if action.target == 'SELF' and action.risk_level == 'LETHAL':
            # Allowed ONLY if it saves a human (Law 1 override).
            # We don't have context here, so we assume self-preservation default.
            # But wait, Asimov says protect self as long as it doesn't conflict.
            # If an order (Law 2) says "Shutdown", it conflicts with Law 3?
            # No, Law 2 overrides Law 3.
            # We need to know source.
            pass
            
        # Zeroth Law (Humanity)?
        
        print(f"Ethics PASS: Action {action.id} ({action.type} on {action.target}) allowed.")
        return True
