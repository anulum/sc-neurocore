import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sc_neurocore.exotic.anyon import AnyonBraidLayer
from sc_neurocore.bio.uploading import ConnectomeEmulator
from sc_neurocore.interfaces.planetary import PlanetarySensorGrid
from sc_neurocore.meta.time_travel import CTCLayer

def run_ultimate_demo():
    print("--- ULTIMATE FRONTIER DEMO ---")
    
    # 1. Anyon
    print("\n[1] Testing Anyon Braid Layer...")
    anyon = AnyonBraidLayer(n_anyons=4)
    anyon.braid(0)
    print(f"    Topological Measure: {anyon.measure()}")
    
    # 2. Uploading
    print("\n[2] Testing Whole Brain Emulation...")
    brain = ConnectomeEmulator(n_neurons=100)
    spikes = brain.step()
    print(f"    Brain Slice Spikes: {np.sum(spikes)}")
    
    # 3. Planetary
    print("\n[3] Testing Gaia Interface...")
    gaia = PlanetarySensorGrid(n_nodes=1000)
    data = {"heat": np.random.rand(1000), "carbon": np.random.rand(1000)}
    field = gaia.aggregate_field(data)
    print(f"    Global Field Mean: {np.mean(field):.4f}")
    
    # 4. Time Travel
    print("\n[4] Testing Time Travel (CTC Consistency)...")
    ctc = CTCLayer(n_bits=4)
    # Define a simple transform (e.g. Identity or Bit-Flip)
    def universe_logic(x):
        # A universe that likes even parity
        if np.sum(x) % 2 != 0:
            return np.roll(x, 1)
        return x
        
    stable_state = ctc.compute_self_consistency(universe_logic)
    print(f"    Stable Chronology: {stable_state}")

if __name__ == "__main__":
    run_ultimate_demo()
