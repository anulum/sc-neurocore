
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sc_neurocore.math.category_theory import CategoryTheoryBridge
from sc_neurocore.verification.formal_proofs import FormalVerifier, Interval
from sc_neurocore.physics.wolfram_hypergraph import WolframHypergraph
from sc_neurocore.interfaces.symbiosis import SymbiosisProtocol

def run_unified_demo():
    print("--- UNIFIED REALITY DEMO ---")
    
    # 1. Category Theory
    print("\n[1] Testing Category Theory Bridge...")
    bridge = CategoryTheoryBridge()
    # Stochastic Bitstream -> Quantum State
    bits = np.array([1, 1, 0, 1])
    f_sto_quant = bridge.get_functor("Stochastic", "Quantum")
    obj_sto = type('obj', (object,), {'data': bits})
    obj_quant = f_sto_quant(obj_sto)
    print(f"    Mapped Bitstream {bits} -> Quantum Amps {obj_quant.data}")
    
    # 2. Formal Verification
    print("\n[2] Testing Formal Verifier...")
    i1 = Interval(0.0, 0.5)
    i2 = Interval(0.5, 1.0)
    FormalVerifier.verify_probability_bounds(i1, i2)
    FormalVerifier.verify_energy_safety(10.0, 5.0)
    
    # 3. Wolfram Physics
    print("\n[3] Testing Wolfram Hypergraph...")
    # Initial: {{1,2}, {2,3}}
    edges = [(1, 2), (2, 3)]
    universe = WolframHypergraph(edges=edges, max_node_id=3)
    print(f"    Initial Edges: {len(universe.edges)}")
    universe.evolve(steps=1)
    print(f"    Evolved Edges: {len(universe.edges)}")
    print(f"    Graph State: {universe.edges}")
    
    # 4. Symbiosis
    print("\n[4] Testing Human-AI Symbiosis...")
    sym = SymbiosisProtocol()
    thought = np.array([0.5, -0.5, 0.9])
    bits = sym.encode_thought(thought, urgency=0.2)
    print(f"    Encoded Thought: {bits}")
    sensation = sym.decode_sensation(bits)
    print(f"    {sensation}")

if __name__ == "__main__":
    run_unified_demo()
