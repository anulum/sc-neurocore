# tests/test_microtubule_superradiance.py
#
# Verification of L1 Quantum Emulation (Microtubule Superradiance).
# Mimics the Verilog logic in 'microtubule_neuron.v' bit-for-bit.

import pytest
import numpy as np

class MicrotubuleModel:
    def __init__(self, seed=0xACE1):
        self.lfsr_state = 0xACE1
        self.chaos_state = seed
        self.potential_acc = 0 # 20-bit accumulator
        self.fire_event = 0
        
        # Parameters from Verilog
        self.DECAY_RATE = 500
        self.GAIN_FIBONACCI = 1618
        self.GAIN_STANDARD = 1000
        
    def step(self, input_streams_int, threshold=0x1000):
        """
        Executes one clock cycle.
        input_streams_int: Integer representing 13 bits (0-8191)
        threshold: 16-bit threshold for firing (compared against acc[19:4])
        """
        # 1. Update Noise Generators
        # LFSR: Taps 15, 13, 12, 10
        feedback = ((self.lfsr_state >> 15) & 1) ^ \
                   ((self.lfsr_state >> 13) & 1) ^ \
                   ((self.lfsr_state >> 12) & 1) ^ \
                   ((self.lfsr_state >> 10) & 1)
        self.lfsr_state = ((self.lfsr_state << 1) & 0xFFFF) | feedback
        
        # Chaos: x <= 4x - 4x^2 (approx fixed point)
        # chaos_state <= (chaos_state << 2) - ((chaos_state * chaos_state) >> 14);
        term1 = (self.chaos_state << 2) & 0xFFFF
        term2 = ((self.chaos_state * self.chaos_state) >> 14) & 0xFFFF
        self.chaos_state = (term1 - term2) & 0xFFFF
        
        quantum_noise = (self.lfsr_state ^ self.chaos_state) & 0xFFFF
        noise_8bit = quantum_noise & 0xFF
        
        # 2. Count Active Inputs
        active_inputs = 0
        for i in range(13):
            if (input_streams_int >> i) & 1:
                active_inputs += 1
                
        # 3. Integrate with Superradiance Check
        if active_inputs > 8:
            # Superradiance Mode (Fibonacci Gain)
            gain = self.GAIN_FIBONACCI
        else:
            # Standard Mode
            gain = self.GAIN_STANDARD
            
        # Accumulate
        input_contribution = active_inputs * gain
        self.potential_acc += input_contribution + noise_8bit
        self.potential_acc &= 0xFFFFF # Keep to 20 bits
        
        # 4. Leak
        if self.potential_acc > self.DECAY_RATE:
            self.potential_acc -= self.DECAY_RATE
        else:
            self.potential_acc = 0
            
        # 5. Threshold (Collapse)
        # Verilog: if (potential_accumulator[19:4] > threshold_reg)
        potential_high = (self.potential_acc >> 4) & 0xFFFF
        
        if potential_high > threshold:
            self.fire_event = 1
            self.potential_acc = 0 # Reset
        else:
            self.fire_event = 0
            
        return self.fire_event, self.potential_acc

def test_superradiance_effect():
    """
    Verify that coherent inputs (>8) trigger faster firing due to 
    Fibonacci Gain (1618) vs Standard Gain (1000).
    """
    # 1. Test Standard Input (8 active streams)
    model_std = MicrotubuleModel(seed=123)
    steps_std = 0
    fired = False
    
    # Run for max 100 steps
    for i in range(100):
        # Input 8 active bits (e.g., 0xFF)
        fire, _ = model_std.step(0xFF, threshold=0x2000) 
        if fire:
            steps_std = i + 1
            fired = True
            break
            
    assert fired, "Standard model did not fire"
    print(f"Standard (8 inputs) fired in {steps_std} steps.")
    
    # 2. Test Superradiant Input (9 active streams)
    model_sup = MicrotubuleModel(seed=123) # Same seed for fair comparison
    steps_sup = 0
    fired = False
    
    for i in range(100):
        # Input 9 active bits (e.g., 0x1FF)
        fire, _ = model_sup.step(0x1FF, threshold=0x2000)
        if fire:
            steps_sup = i + 1
            fired = True
            break
            
    assert fired, "Superradiant model did not fire"
    print(f"Superradiant (9 inputs) fired in {steps_sup} steps.")
    
    # Verification: Superradiance should be FASTER (fewer steps)
    # Even though 9 is only 12.5% more than 8, the gain jump (1000 -> 1618) is 61.8%
    assert steps_sup < steps_std
    print("SUCCESS: Superradiance confirmed (Quantum Gain boost active).")

def test_chaos_divergence():
    """
    Verify that slightly different seeds produce diverging noise patterns
    (Chaos sensitivity).
    """
    m1 = MicrotubuleModel(seed=1000)
    m2 = MicrotubuleModel(seed=1001) # 1 bit difference
    
    # Run 50 steps
    diverged = False
    for i in range(50):
        m1.step(0)
        m2.step(0)
        if m1.chaos_state != m2.chaos_state:
            print(f"Chaos diverged at step {i}")
            diverged = True
            break
            
    assert diverged, "Chaos generator did not diverge for close seeds"

if __name__ == "__main__":
    test_superradiance_effect()
    test_chaos_divergence()
