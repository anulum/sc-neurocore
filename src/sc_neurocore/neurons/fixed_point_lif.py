
from dataclasses import dataclass
import numpy as np

@dataclass
class FixedPointLIFNeuron:
    """
    Bit-true fixed-point model of the Verilog sc_lif_neuron.
    """
    data_width: int = 16
    fraction: int = 8
    v_rest: int = 0
    v_reset: int = 0
    v_threshold: int = 256 # 1.0 << 8
    refractory_period: int = 2
    
    def __post_init__(self):
        self.v = self.v_rest
        self.refractory_counter = 0
        
    def step(self, leak_k: int, gain_k: int, I_t: int, noise_in: int) -> tuple[int, int]:
        """
        Executes one clock cycle.
        All inputs are integers (fixed-point representation).
        Returns (spike, v_out).
        """
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            spike = 0
            # Verilog: v_reg <= V_REST
            self.v = self.v_rest
            return spike, self.v
            
        # Calculate leak
        # leak_mul = (V_REST - v_reg) * leak_k
        diff = self.v_rest - self.v
        leak_mul = diff * leak_k
        dv_leak = leak_mul >> self.fraction
        
        # Calculate input
        in_mul = I_t * gain_k
        dv_in = in_mul >> self.fraction
        
        # Next potential
        v_next = self.v + dv_leak + dv_in + noise_in
        
        # Threshold check
        if v_next >= self.v_threshold:
            spike = 1
            self.v = self.v_reset
            self.refractory_counter = self.refractory_period
        else:
            spike = 0
            self.v = v_next
            
        return spike, self.v
