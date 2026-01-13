
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sc_neurocore.meta.time_crystal import TimeCrystalLayer
from sc_neurocore.meta.vacuum import VacuumNoiseSource
from sc_neurocore.meta.hyper_turing import OracleLayer
from sc_neurocore.meta.black_hole import EventHorizonLayer

def run_meta_demo():
    print("--- META-COMPUTING SPECULATIVE DEMO ---")
    
    # 1. Time Crystal
    print("\n[1] Testing Time Crystal Memory...")
    tc = TimeCrystalLayer(n_spins=4)
    bits = tc.get_bitstream(cycles=10)
    print(f"    Temporal Oscillations (Bits): {bits}")
    
    # 2. Vacuum
    print("\n[2] Testing Vacuum Fluctuation Harvester...")
    vac = VacuumNoiseSource(dimension=2, plate_distance=0.5)
    virtual_bits = vac.generate_virtual_bits(length=10)
    print(f"    Harvested Virtual Bits:\n{virtual_bits}")
    
    # 3. Hyper-Turing
    print("\n[3] Testing Hyper-Turing Oracle...")
    oracle = OracleLayer()
    halting_stream = np.zeros(200) # Settle to zero
    random_stream = np.random.randint(0, 2, 200)
    print(f"    Halting Predict (Zeros): {oracle.solve_halting(halting_stream)}")
    print(f"    Halting Predict (Random): {oracle.solve_halting(random_stream)}")
    
    # 4. Black Hole
    print("\n[4] Testing Black Hole Scrambler...")
    bh = EventHorizonLayer(n_inputs=2, n_outputs=4)
    in_bits = np.array([[1, 1, 1, 1], [0, 0, 0, 0]], dtype=np.uint8)
    scrambled = bh.scramble(in_bits)
    print(f"    Input (Polarized):\n{in_bits}")
    print(f"    Scrambled (Entropic):\n{scrambled}")

if __name__ == "__main__":
    run_meta_demo()
