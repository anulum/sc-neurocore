
import numpy as np
from dataclasses import dataclass, field
from typing import List

@dataclass
class DigitalImmuneSystem:
    """
    Artificial Immune System (AIS) for Agent Security.
    Detects anomalies (Non-Self) and neutralizes threats.
    """
    self_patterns: List[np.ndarray] = field(default_factory=list)
    tolerance: float = 0.2
    
    def train_self(self, normal_state: np.ndarray):
        """
        Learn a 'Self' pattern (Normal behavior).
        """
        # Store representative vectors (Antibodies)
        if len(self.self_patterns) < 100:
            self.self_patterns.append(normal_state)
            
    def scan(self, current_state: np.ndarray) -> bool:
        """
        Check if current state matches 'Self'.
        Returns True if Healthy, False if Infected (Anomaly).
        """
        if not self.self_patterns:
            return True # No training yet
            
        # Distance to nearest Self pattern
        distances = [np.linalg.norm(current_state - p) for p in self.self_patterns]
        min_dist = min(distances)
        
        if min_dist > self.tolerance:
            print(f"Immune System: ANOMALY DETECTED! Deviation: {min_dist:.4f}")
            self._trigger_response()
            return False
            
        return True
        
    def _trigger_response(self):
        print("Immune System: Initiating Quarantine Protocol...")
        # Action: Disable compromised modules (Simulation)
