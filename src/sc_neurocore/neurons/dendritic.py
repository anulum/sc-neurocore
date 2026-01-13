
from dataclasses import dataclass
import numpy as np
from typing import List

@dataclass
class StochasticDendriticNeuron:
    """
    Two-Compartment Neuron (Soma + 2 Dendrites).
    Can solve non-linear problems (XOR) singly.
    
    Structure:
    Input A -> Dendrite 1
    Input B -> Dendrite 2
    Dendrite Output = NonLinear(Input)
    Soma = Integrate(D1 + D2)
    """
    threshold: float = 1.5
    
    def step(self, input_a: float, input_b: float) -> int:
        """
        Inputs are probabilities/currents.
        """
        # Dendrite 1: Active if A is high (Excites)
        d1 = input_a 
        
        # Dendrite 2: Active if B is high (Excites)
        d2 = input_b
        
        # Soma Interaction:
        # Simplest non-linear interaction:
        # If we want XOR: 
        # Fire if (A & !B) OR (!A & B)
        # That logic usually requires inhibitory weights.
        
        # Let's implement a "Active Dendrite" model.
        # Soma Current = D1 + D2 - Interaction(D1*D2) (Shunting Inhibition)
        
        current = d1 + d2 - 2.0 * (d1 * d2) 
        # Logic:
        # 0,0 -> 0
        # 1,0 -> 1
        # 0,1 -> 1
        # 1,1 -> 1+1 - 2 = 0
        
        if current > 0.5: # Simple threshold
            return 1
        return 0
