
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from sc_neurocore.core.orchestrator import CognitiveOrchestrator
from sc_neurocore.core.self_awareness import MetaCognitionLoop
from sc_neurocore.bio.neuromodulation import NeuromodulatorSystem
from sc_neurocore.analysis.qualia import QualiaTuringTest
from sc_neurocore.transcendent.noetic import SemioticTriad

def run_sapience_demo():
    print("--- SAPIENT AGENT DEMO ---")
    
    # 1. Orchestrator & Self-Awareness
    print("\n[1] Testing Self-Awareness (Meta-Cognition)...")
    orch = CognitiveOrchestrator()
    orch.register_module("vision", object())
    orch.register_module("motor", object())
    orch.active_goals = ["Explore Sector 7"]
    
    mirror = MetaCognitionLoop()
    mirror.observe(orch)
    print(f"    Self-Reflection: {mirror.reflect()}")
    
    # 2. Neuromodulation
    print("\n[2] Testing Neuromorphic Emotion (Hormonal Bias)...")
    emotions = NeuromodulatorSystem()
    print(f"    Initial State: DA={emotions.da_level}, 5HT={emotions.ht_level}")
    
    # Simulate a 'Win' (Reward) and 'Scary Event' (Stress)
    emotions.update_levels(reward=1.0, stress=0.8)
    print(f"    State after Event: DA={emotions.da_level:.2f}, 5HT={emotions.ht_level:.2f}, NE={emotions.ne_level:.2f}")
    
    params = {'v_threshold': 1.0, 'noise_std': 0.02}
    mod_params = emotions.modulate_neuron(params)
    print(f"    Modulated Neuron Params: {mod_params}")
    
    # 3. Qualia Test
    print("\n[3] Testing Qualia Turing Test...")
    semiotics = SemioticTriad()
    semiotics.learn_association("Fire", "Destruction")
    semiotics.learn_association("Destruction", "Chaos")
    semiotics.learn_association("Emotion", "Fire") # Context
    
    test = QualiaTuringTest(semiotics)
    # State with peak at 0 (Fire)
    state = np.array([1.0, 0.1, 0.1])
    test.administer_test(state)

if __name__ == "__main__":
    run_sapience_demo()
