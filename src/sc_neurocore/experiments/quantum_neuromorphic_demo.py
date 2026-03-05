# SPDX-License-Identifier: AGPL-3.0-or-later
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from sc_neurocore.solvers.ising import StochasticIsingGraph
from sc_neurocore.interfaces.dvs_input import DVSInputLayer
from sc_neurocore.learning.federated import FederatedAggregator
from sc_neurocore.utils.connectomes import ConnectomeGenerator
from sc_neurocore.neurons.homeostatic_lif import HomeostaticLIFNeuron
from sc_neurocore.models.zoo import SCDigitClassifier


def run_demo():  # type: ignore
    print("--- QUANTUM-NEUROMORPHIC DEMO ---")

    # 1. Ising Machine
    print("\n[1] Testing Ising Solver...")
    # Simple ferromagnetic chain: J=1 interactions
    J = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    h = np.zeros(3)
    ising = StochasticIsingGraph(num_spins=3, J=J, h=h, temperature=5.0)
    print(f"    Initial Energy: {ising.get_energy()}")
    for _ in range(100):
        ising.step()  # type: ignore
    print(f"    Final Energy: {ising.get_energy()}")
    print(f"    Config: {ising.get_config()} (Should be aligned)")

    # 2. DVS Interface
    print("\n[2] Testing DVS Interface...")
    dvs = DVSInputLayer(height=10, width=10)
    # Simulate a moving dot
    events = [(5, 5, 10.0, 1), (5, 6, 11.0, 1), (5, 7, 12.0, 1)]
    probs = dvs.process_events(events)
    print(f"    Max Activity: {np.max(probs):.4f}")
    bits = dvs.generate_bitstream_frame(length=10)
    print(f"    Bitstream Shape: {bits.shape}")

    # 3. Federated Learning
    print("\n[3] Testing Federated Aggregation...")
    g1 = np.array([1, 1, 0, 0, 1], dtype=np.uint8)
    g2 = np.array([1, 0, 0, 0, 1], dtype=np.uint8)
    g3 = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
    agg = FederatedAggregator.aggregate_gradients([g1, g2, g3])
    print(f"    Aggregated (Majority): {agg}")

    # 4. Connectomes
    print("\n[4] Testing Connectome Generation...")
    adj = ConnectomeGenerator.generate_watts_strogatz(n_neurons=10, k_neighbors=4, p_rewire=0.1)
    print(f"    Small-World Edges: {np.sum(adj)}")

    # 5. Homeostatic Plasticity
    print("\n[5] Testing Homeostatic Neuron...")
    neuron = HomeostaticLIFNeuron(target_rate=0.5, v_threshold=1.0)
    # Force high firing
    print(f"    Initial Threshold: {neuron.v_threshold}")
    for _ in range(50):
        neuron.step(input_current=10.0)
    print(f"    Adapted Threshold: {neuron.v_threshold:.4f} (Should Increase)")

    # 6. Model Zoo
    print("\n[6] Testing Model Zoo (Digit Classifier)...")
    model = SCDigitClassifier()  # type: ignore
    img = np.random.random((28, 28))
    pred = model.forward(img)
    print(f"    Prediction: {pred}")


if __name__ == "__main__":
    run_demo()  # type: ignore
