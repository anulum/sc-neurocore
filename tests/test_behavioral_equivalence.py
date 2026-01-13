# tests/test_behavioral_equivalence.py
#
# Bit-true behavioral model of the sc-neurocore hardware.
# Verifies the logic of LFSR, Stochastic Generation, and LIF Update
# against the Verilog specification.

import pytest
import numpy as np

def lfsr_16bit_step(reg):
    """
    Python model of the 16-bit LFSR in sc_bitstream_encoder.v
    Polynomial: x^16 + x^14 + x^13 + x^11 + 1
    Taps at: 15, 13, 12, 10 (0-indexed)
    """
    # Feedback = reg[15] ^ reg[13] ^ reg[12] ^ reg[10]
    feedback = ((reg >> 15) & 1) ^ ((reg >> 13) & 1) ^ ((reg >> 12) & 1) ^ ((reg >> 10) & 1)
    
    # Shift left and insert feedback at LSB
    # Note: Verilog implementation was:
    # lfsr_reg <= {lfsr_reg[LFSR_WIDTH-2:0], feedback};
    # This means SHIFT LEFT by 1, insert feedback at 0.
    
    new_reg = ((reg << 1) & 0xFFFF) | feedback
    return new_reg

def test_lfsr_sequence():
    """Verify LFSR sequence generation matches hardware logic."""
    seed = 0xACE1
    reg = seed
    
    # Run for a few steps
    sequence = []
    for _ in range(10):
        reg = lfsr_16bit_step(reg)
        sequence.append(reg)
        
    # Expected sequence (manual verification or cross-check)
    # 0xACE1 = 1010 1100 1110 0001
    # Taps: 15(1), 13(1), 12(0), 10(1) -> 1^1^0^1 = 1
    # Next: (0xACE1 << 1) | 1 = 0x59C3 | 1 = 0x59C3
    
    # Let's check first step logic explicitly
    # 1010... -> 1 ^ 1 ^ 0 ^ 1 = 1.
    # Shift: 0101 1001 1100 0010 (0x59C2). Add 1 -> 0x59C3.
    
    assert sequence[0] == 0x59C3
    print(f"LFSR Step 1: {hex(sequence[0])} (Expected 0x59C3)")

def fixed_point_lif_step(v, I_t, params):
    """
    Python model of sc_lif_neuron.v
    
    DATA_WIDTH = 16
    FRACTION = 8
    """
    DATA_WIDTH = 16
    FRACTION = 8
    ONE = 1 << FRACTION
    
    V_REST = params.get('V_REST', 0)
    V_RESET = params.get('V_RESET', 0)
    V_THRESHOLD = params.get('V_THRESHOLD', ONE)
    ALPHA_LEAK = params.get('ALPHA_LEAK', 10) # 10/256 approx 0.04
    GAIN_IN = params.get('GAIN_IN', ONE)
    
    # Verilog Logic:
    # assign leak_mul = (V_REST - v_reg) * ALPHA_LEAK;
    # assign dv_leak  = leak_mul >>> FRACTION;
    leak_mul = (V_REST - v) * ALPHA_LEAK
    dv_leak = leak_mul >> FRACTION # Python >> is arithmetic shift for signed integers
    
    # assign in_mul = I_t * GAIN_IN;
    # assign dv_in  = in_mul >>> FRACTION;
    in_mul = I_t * GAIN_IN
    dv_in = in_mul >> FRACTION
    
    v_next = v + dv_leak + dv_in
    
    # Threshold check
    spike = 0
    if v_next >= V_THRESHOLD:
        spike = 1
        v_next = V_RESET
        
    return v_next, spike

def test_lif_neuron_dynamics():
    """Verify LIF neuron integration and spiking."""
    params = {
        'V_THRESHOLD': 256, # 1.0 in Q8.8
        'ALPHA_LEAK': 10,
        'GAIN_IN': 256
    }
    
    # Case 1: Constant input, integration
    v = 0
    I_t = 10 # Small input (10/256)
    
    # Step 1
    # dv_leak = (0-0)*10 >> 8 = 0
    # dv_in = 10*256 >> 8 = 10
    # v_next = 10
    v, s = fixed_point_lif_step(v, I_t, params)
    assert v == 10
    assert s == 0
    
    # Case 2: Leak
    v = 100
    I_t = 0
    # dv_leak = (0-100)*10 = -1000. -1000 >> 8 = -3 (approx -3.9) -> -4? 
    # Python integer division floor: -1000 // 256 = -4.
    # Verilog >>> checks: -1000 is ...1111110000011000. Shift right 8 -> ...11111100 = -4.
    # Matches.
    v, s = fixed_point_lif_step(v, I_t, params)
    assert v == 100 - 4 # 96
    
    print("LIF dynamics verified.")

if __name__ == "__main__":
    test_lfsr_sequence()
    test_lif_neuron_dynamics()
