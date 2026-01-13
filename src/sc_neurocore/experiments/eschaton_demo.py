import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sc_neurocore.eschaton.heat_death import HeatDeathLayer
from sc_neurocore.eschaton.computronium import PlanckGrid
from sc_neurocore.eschaton.holographic import HolographicBoundary
from sc_neurocore.eschaton.simulation import NestedUniverse

def run_eschaton_demo():
    print("--- ESCHATOLOGICAL FRONTIERS DEMO ---")
    
    # 1. Heat Death
    print("\n[1] Testing Heat Death Survival...")
    hd = HeatDeathLayer(initial_energy=0.001, entropy_rate=0.5)
    bits = np.ones(10, dtype=np.uint8)
    out = hd.compute_step(bits)
    print(f"    Output (Low Energy): {out}")
    print(f"    {hd.status()}")
    
    # 2. Planck
    print("\n[2] Testing Planck-Level Computronium...")
    pg = PlanckGrid(volume_cm3=1.0, mass_kg=1.0)
    print(f"    {pg.simulate_step()}")
    
    # 3. Holographic
    print("\n[3] Testing Holographic Boundary Mapping...")
    hb = HolographicBoundary(grid_size=4)
    bulk = np.random.randint(0, 2, (4, 4, 4)).astype(np.uint8)
    bound = hb.encode_to_boundary(bulk)
    print(f"    2D Boundary Representation (Parity):\n{bound}")
    
    # 4. Simulation
    print("\n[4] Testing Simulation Hypothesis (Recursion)...")
    base_reality = NestedUniverse(id=0, computing_resources=100.0)
    sim1 = base_reality.spawn_simulation()
    if sim1:
        sim2 = sim1.spawn_simulation()
        if sim2:
            sim3 = sim2.spawn_simulation()

if __name__ == "__main__":
    run_eschaton_demo()
