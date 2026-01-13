
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sc_neurocore.core.orchestrator import CognitiveOrchestrator
from sc_neurocore.core.mdl_parser import MindDescriptionLanguage
from sc_neurocore.viz.neuro_art import NeuroArtGenerator
from sc_neurocore.security.ethics import AsimovGovernor, ActionRequest
from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer

def run_sentient_demo():
    print("--- SENTIENT AGENT DEMO ---")
    
    # 1. MDL
    print("\n[1] Testing Mind Description Language...")
    orch = CognitiveOrchestrator()
    layer = VectorizedSCLayer(n_inputs=2, n_neurons=2)
    orch.register_module("cortex", layer)
    
    yaml_dump = MindDescriptionLanguage.encode(orch, "Sentient-1")
    print(f"    MDL Dump (Partial):\n{yaml_dump[:100]}...")
    
    # 2. NeuroArt
    print("\n[2] Testing Generative Neuro-Art...")
    art = NeuroArtGenerator(resolution=32) # Small for terminal
    state = np.array([0.5, -0.2, 0.9])
    img = art.generate_visual(state)
    print(f"    Generated Art Shape: {img.shape} (Sum: {np.sum(img)})")
    
    # 3. Ethics
    print("\n[3] Testing Asimov Governor...")
    gov = AsimovGovernor()
    
    safe_act = ActionRequest(1, 'HEAL', 'HUMAN', 'SAFE')
    gov.check_laws(safe_act)
    
    unsafe_act = ActionRequest(2, 'FIRE', 'HUMAN', 'LETHAL')
    gov.check_laws(unsafe_act)

if __name__ == "__main__":
    run_sentient_demo()
