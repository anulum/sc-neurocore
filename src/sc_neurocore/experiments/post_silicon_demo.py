
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sc_neurocore.post_silicon.reversible import ReversibleLayer
from sc_neurocore.post_silicon.femto import FemtoSwitch
from sc_neurocore.post_silicon.synthetic_cell import CellularComputer
from sc_neurocore.post_silicon.claytronics import CatomLattice

def run_post_silicon_demo():
    print("--- POST-SILICON PARADIGMS DEMO ---")
    
    # 1. Reversible
    print("\n[1] Testing Reversible (Adiabatic) Logic...")
    rev = ReversibleLayer()
    a = np.array([1, 0, 1, 0], dtype=np.uint8)
    b = np.array([1, 1, 0, 0], dtype=np.uint8)
    res, garbage = rev.forward(a, b)
    print(f"    AND Result: {res}")
    print(f"    Garbage (Reversible Info): {garbage}")
    
    # 2. Femto
    print("\n[2] Testing Femto-Computing (Quarks)...")
    femto = FemtoSwitch()
    q1 = np.array([0, 1, 2], dtype=np.uint8) # R, G, B
    q2 = np.array([1, 2, 0], dtype=np.uint8) # G, B, R
    # R+G->B(2), G+B->R(0), B+R->G(1)
    out = femto.interact(q1, q2)
    print(f"    Quark Interaction: {out}")
    
    # 3. Cell
    print("\n[3] Testing Synthetic Cell...")
    cell = CellularComputer(n_molecules_a=100, n_molecules_b=100)
    product = cell.step(0, 0)
    print(f"    Product C Produced: {product}")
    
    # 4. Claytronics
    print("\n[4] Testing Programmable Matter...")
    clay = CatomLattice(size=5)
    clay.load = np.array([0.1, 0.9, 0.2, 0.8, 0.5])
    print(f"    Initial Load: {clay.load}")
    clay.reconfigure() # Bubble sort step
    print(f"    Reconfigured Load: {clay.load}")

if __name__ == "__main__":
    run_post_silicon_demo()
