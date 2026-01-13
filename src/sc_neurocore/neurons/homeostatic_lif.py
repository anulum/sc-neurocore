
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from .stochastic_lif import StochasticLIFNeuron

@dataclass
class HomeostaticLIFNeuron(StochasticLIFNeuron):
    """
    LIF Neuron with Homeostatic Threshold Adaptation.
    Self-regulates firing rate to a target setpoint.
    """
    target_rate: float = 0.1     # Desired firing probability
    adaptation_rate: float = 0.01 # How fast threshold changes
    rate_trace: float = 0.0
    trace_decay: float = 0.95
    
    def step(self, input_current: float) -> int:
        spike = super().step(input_current)
        
        # Update Rate Trace (Low-pass filter of spikes)
        self.rate_trace = self.rate_trace * self.trace_decay + spike * (1.0 - self.trace_decay)
        
        # Homeostatic Control
        # If trace > target, threshold increases (harder to fire)
        # If trace < target, threshold decreases (easier to fire)
        
        error = self.rate_trace - self.target_rate
        
        # Adjust threshold
        self.v_threshold += self.adaptation_rate * error
        
        # Safety limits for threshold
        self.v_threshold = max(0.1, self.v_threshold)
        
        return spike
        
    def get_state(self) -> Dict[str, Any]:
        s = super().get_state()
        s['threshold'] = float(self.v_threshold)
        s['rate_trace'] = float(self.rate_trace)
        return s
