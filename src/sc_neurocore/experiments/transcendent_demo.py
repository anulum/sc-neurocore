
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sc_neurocore.transcendent.multiverse import EverettTreeLayer
from sc_neurocore.transcendent.spacetime import SpinNetwork
from sc_neurocore.transcendent.vacuum_decay import FalseVacuumField
from sc_neurocore.transcendent.noetic import SemioticTriad, Sign

def run_transcendent_demo():
    print("--- TRANSCENDENT COMPUTING DEMO ---")
    
    # 1. Multiverse
    print("\n[1] Testing Everettian Branching Solver...")
    solver = EverettTreeLayer(max_depth=5)
    # Task: find bit sequence summing to 3
    def goal(s): return s == 3
    def transition(s, choice): return s + choice
    path = solver.solve(0, goal, transition)
    print(f"    Solution Timeline Found: {path}")
    
    # 2. Spacetime
    print("\n[2] Testing Sub-Planckian Spin Network...")
    net = SpinNetwork(n_nodes=3)
    print(f"    Initial Volume: {net.calculate_volume():.4f}")
    net.pachner_move_1_3(0)
    print(f"    Volume after 1->3 Move: {net.calculate_volume():.4f}")
    
    # 3. Vacuum Decay
    print("\n[3] Testing Vacuum Decay Engineer...")
    vacuum = FalseVacuumField(size=10)
    vacuum.nucleate(5, 5) # Input pulse
    print(f"    Initial Energy: {vacuum.measure_energy()}")
    for _ in range(3): vacuum.step()
    print(f"    Energy after 3 steps (Expansion): {vacuum.measure_energy()}")
    
    # 4. Noetic
    print("\n[4] Testing Semiotic AI (Meaning Shift)...")
    triad = SemioticTriad()
    triad.learn_association("Fire", "Heat")
    triad.learn_association("Heat", "Life")
    
    s1 = Sign("Spark", "Fire", "Heat")
    print(f"    Initial Sign: {s1}")
    s2 = triad.interpret(s1)
    print(f"    Shifted Sign (Semiosis): {s2}")

if __name__ == "__main__":
    run_transcendent_demo()
