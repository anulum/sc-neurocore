
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sc_neurocore.meta.singularity import RecursiveSelfImprover
from sc_neurocore.exotic.matrioshka import DysonSwarmNet
from sc_neurocore.meta.omega import OmegaIntegrator
from sc_neurocore.exotic.constructor import ConstructorCell
from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer

def run_terminal_demo():
    print("--- TERMINAL HORIZON DEMO ---")
    
    # 1. Singularity
    print("\n[1] Testing Recursive Self-Improvement...")
    layer = VectorizedSCLayer(n_inputs=10, n_neurons=5)
    improver = RecursiveSelfImprover()
    imp_val = improver.improve(layer)
    print(f"    Intelligence Growth Factor: {imp_val:.6f}")
    
    # 2. Matrioshka
    print("\n[2] Testing Matrioshka Brain (Dyson Swarm)...")
    brain = DysonSwarmNet(n_shells=3, n_nodes_per_shell=10)
    input_energy = np.random.rand(10)
    final_output = brain.process(input_energy)
    print(f"    Outer Shell Reasoning Output: {final_output[:4]}")
    
    # 3. Omega
    print("\n[3] Testing Omega Point Integration...")
    omega = OmegaIntegrator()
    states = [np.random.rand(10) for _ in range(5)]
    unified = omega.unify(states)
    print(f"    Unified State Vector Magnitude: {np.linalg.norm(unified):.4f}")
    
    # 4. Constructor
    print("\n[4] Testing Universal Constructor (Self-Replication)...")
    blueprint = np.array([1, 0, 1, 1], dtype=np.uint8)
    cell = ConstructorCell(id=0, blueprint=blueprint)
    offspring = cell.replicate()
    print(f"    Original ID: {cell.id}, Offspring ID: {offspring.id}")
    print(f"    Blueprint Integrity: {np.array_equal(cell.blueprint, offspring.blueprint)}")

if __name__ == "__main__":
    run_terminal_demo()
