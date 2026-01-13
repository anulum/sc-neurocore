
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sc_neurocore.exotic.fungal import MyceliumLayer
from sc_neurocore.exotic.chemical import ReactionDiffusionSolver
from sc_neurocore.exotic.mechanical import MechanicalLatticeLayer
from sc_neurocore.exotic.space import RadHardLayer
from sc_neurocore.analysis.consciousness import PhiEvaluator

def run_blue_sky_demo():
    print("--- BLUE SKY COMPUTING DEMO ---")
    
    # 1. Fungal
    print("\n[1] Testing Fungal Mycelium Network...")
    fungus = MyceliumLayer(n_nodes=5)
    inputs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    # Run growth
    for _ in range(10):
        out = fungus.step(inputs)
    print(f"    Conductance Matrix (Path Formed):\n{fungus.conductance}")
    
    # 2. Chemical
    print("\n[2] Testing Chemical Reaction-Diffusion...")
    chem = ReactionDiffusionSolver(width=20, height=20)
    for _ in range(50):
        chem.step()
    print(f"    Turing Pattern Max: {np.max(chem.B)}")
    
    # 3. Mechanical
    print("\n[3] Testing Mechanical Neural Network...")
    mech = MechanicalLatticeLayer(n_nodes=5)
    # Apply force to node 0
    forces = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    mech.relax(forces, clamped_nodes=[4])
    print(f"    Displacements: {mech.x}")
    
    # 4. Space
    print("\n[4] Testing Rad-Hard TMR Layer...")
    space = RadHardLayer(n_inputs=10, n_neurons=5, seu_rate=0.1) # High radiation
    out = space.forward(np.random.random(10))
    print(f"    Output (TMR Corrected): {out}")
    
    # 5. Consciousness
    print("\n[5] Testing Consciousness (Phi)...")
    # Generate correlated bitstreams (High Phi)
    bits = np.zeros((3, 100), dtype=np.uint8)
    bits[0] = np.random.randint(0, 2, 100)
    bits[1] = bits[0] # Clone
    bits[2] = bits[0] # Clone
    
    phi = PhiEvaluator.calculate_phi(bits)
    print(f"    Phi (Highly Integrated): {phi:.4f}")
    
    # Random (Low Phi)
    bits_rand = np.random.randint(0, 2, (3, 100)).astype(np.uint8)
    phi_rand = PhiEvaluator.calculate_phi(bits_rand)
    print(f"    Phi (Random): {phi_rand:.4f}")

if __name__ == "__main__":
    run_blue_sky_demo()
