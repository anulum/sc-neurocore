import numpy as np
import sys
import os
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sc_neurocore.core.orchestrator import CognitiveOrchestrator
from sc_neurocore.core.immortality import DigitalSoul
from sc_neurocore.core.replication import VonNeumannProbe
from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer

def run_immortal_demo():
    print("--- IMMORTAL PROBE SYNERGY DEMO ---")
    
    # 1. Setup Original Agent
    orch = CognitiveOrchestrator()
    layer = VectorizedSCLayer(n_inputs=10, n_neurons=2)
    # Give it some unique weight state
    layer.weights.fill(0.777)
    orch.register_module("brain_layer", layer)
    orch.set_attention("brain_layer")
    
    # 2. Capture Soul
    soul = DigitalSoul(agent_id="Explorer-1")
    soul.capture_agent(orch)
    soul.save_soul("explorer_1.soul")
    
    # 3. Replicate (Von Neumann)
    probe = VonNeumannProbe(probe_id=1)
    target_sector = os.path.abspath("sector_theta_9")
    probe.replicate(target_sector)
    
    # 4. Simulate Reincarnation in New Sector
    # (We move the soul file to the new sector)
    shutil.move("explorer_1.soul", os.path.join(target_sector, "explorer_1.soul"))
    
    print("\n--- NEW SECTOR INITIALIZED ---")
    # In the new sector, a new agent is born
    new_orch = CognitiveOrchestrator()
    new_layer = VectorizedSCLayer(n_inputs=10, n_neurons=2) # Fresh weights
    new_orch.register_module("brain_layer", new_layer)
    
    print(f"    New Agent Weights (pre-soul): {new_layer.weights[0,0]:.4f}")
    
    # Load and Reincarnate
    new_soul = DigitalSoul.load_soul(os.path.join(target_sector, "explorer_1.soul"))
    new_soul.reincarnate(new_orch)
    
    print(f"    New Agent Weights (post-soul): {new_layer.weights[0,0]:.4f}")
    
    if new_layer.weights[0,0] == 0.777:
        print("\nSUCCESS: Digital Immortality achieved. Agent state preserved across replication.")
    else:
        print("\nFAILURE: Soul corruption detected.")

    # Cleanup
    # shutil.rmtree(target_sector) # Uncomment to delete the new sector files

if __name__ == "__main__":
    run_immortal_demo()
