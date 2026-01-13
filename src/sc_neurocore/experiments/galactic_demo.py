
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sc_neurocore.interfaces.interstellar import InterstellarDTN, Packet
from sc_neurocore.exotic.dyson_grid import DysonPowerGrid
from sc_neurocore.analysis.kardashev import KardashevEstimator
from sc_neurocore.meta.fermi_game import DarkForestAgent

def run_galactic_demo():
    print("--- GALACTIC SCALE DEMO ---")
    
    # 1. DTN
    print("\n[1] Testing Interstellar DTN...")
    dtn = InterstellarDTN(node_id="Alpha-Centauri")
    pkt = Packet(id=1, data=np.array([1,0,1]))
    dtn.receive(pkt)
    # Simulate time steps
    forwarded = None
    for _ in range(20):
        res = dtn.step()
        if res: 
            forwarded = res
            break
    print(f"    Packet Forwarded: {forwarded is not None}")
    
    # 2. Dyson
    print("\n[2] Testing Dyson Power Grid...")
    grid = DysonPowerGrid(n_collectors=1000, n_consumers=50)
    # Sun output ~ 3.8e26 W
    satisfaction = grid.step(solar_output=3.8e26)
    print(f"    Grid Satisfaction: {satisfaction*100:.2f}%")
    
    # 3. Kardashev
    print("\n[3] Testing Kardashev Estimator...")
    k_type = KardashevEstimator.calculate_type(3.8e26)
    print(f"    Civilization Type (Sun): {k_type:.2f}")
    
    # 4. Fermi
    print("\n[4] Testing Dark Forest Game...")
    agent = DarkForestAgent()
    action = agent.decide(alien_signal_strength=0.8)
    print(f"    Alien Signal Strong! Action: {action}")

if __name__ == "__main__":
    run_galactic_demo()
