from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from ..neurons.homeostatic_lif import HomeostaticLIFNeuron

@dataclass
class StochasticCPG:
    """
    Central Pattern Generator using two mutually inhibiting neurons.
    Generates rhythmic alternating outputs (e.g., Left/Right leg).
    """
    drive_current: float = 2.0
    inhibition_weight: float = 2.0
    
    def __post_init__(self):
        # High adaptation rate to force switching
        self.n1 = HomeostaticLIFNeuron(v_threshold=1.0, adaptation_rate=0.1, target_rate=0.3)
        self.n2 = HomeostaticLIFNeuron(v_threshold=1.0, adaptation_rate=0.1, target_rate=0.3)
        
        self.s1_trace = 0.0
        self.s2_trace = 0.0
        self.decay = 0.8
        
    def step(self) -> tuple[int, int]:
        # Inhibition logic:
        # Input to N1 = Drive - Weight * N2_Activity
        # Input to N2 = Drive - Weight * N1_Activity
        
        # We use a trace of spikes for inhibition "potential"
        i1 = self.drive_current - self.inhibition_weight * self.s2_trace
        i2 = self.drive_current - self.inhibition_weight * self.s1_trace
        
        spike1 = self.n1.step(i1)
        spike2 = self.n2.step(i2)
        
        # Update traces
        self.s1_trace = self.s1_trace * self.decay + spike1
        self.s2_trace = self.s2_trace * self.decay + spike2
        
        return spike1, spike2
