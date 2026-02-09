
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sc_neurocore.interfaces.bci import BCIDecoder
from sc_neurocore.quantum.hybrid import QuantumStochasticLayer
from sc_neurocore.robotics.cpg import StochasticCPG
from sc_neurocore.bio.grn import GeneticRegulatoryLayer
from sc_neurocore.hdl_gen.spice_generator import SpiceGenerator
from sc_neurocore.learning.lifelong import EWC_SCLayer

def run_frontier_demo():
    print("--- FINAL FRONTIER DEMO ---")
    
    # 1. BCI
    print("\n[1] Testing BCI Decoder...")
    bci = BCIDecoder(channels=2)
    signal = np.array([[0.1, 0.2, 0.5, 0.8], [0.9, 0.8, 0.2, 0.1]])
    bits = bci.encode_to_bitstream(signal, length=10)
    print(f"    Encoded BCI: {bits.shape}")
    
    # 2. Quantum
    print("\n[2] Testing Quantum Layer...")
    q_layer = QuantumStochasticLayer(n_qubits=2, length=10)
    in_bits = np.ones((2, 10), dtype=np.uint8) # High prob -> Pi rotation
    out_bits = q_layer.forward(in_bits)
    # Cos(pi/2)^2 = 0. Output should be low?
    # Wait, theta = p * pi. If p=1, theta=pi.
    # Prob = cos(pi/2)^2 = 0. Correct.
    print(f"    Quantum Output (expect 0s): {out_bits}")
    
    # 3. CPG
    print("\n[3] Testing Robotic CPG...")
    cpg = StochasticCPG()
    print("    Running CPG Step...")
    s1, s2 = cpg.step()
    print(f"    Spikes: {s1}, {s2}")
    
    # 4. GRN
    print("\n[4] Testing Gene Regulatory Network...")
    grn = GeneticRegulatoryLayer(n_neurons=2)
    grn.step(np.array([1, 1])) # High activity
    print(f"    Protein Levels: {grn.protein_levels}")
    
    # 5. SPICE
    print("\n[5] Testing SPICE Gen...")
    w = np.random.random((2, 2))
    SpiceGenerator.generate_crossbar(w, "memristor.sp")
    
    # 6. EWC
    print("\n[6] Testing Lifelong Learning...")
    ewc = EWC_SCLayer(n_inputs=2, n_neurons=2)
    ewc.consolidate_task()
    print("    Task Consolidated. Fisher Info stored.")

if __name__ == "__main__":
    run_frontier_demo()
