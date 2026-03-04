import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from sc_neurocore.layers.sc_learning_layer import SCLearningLayer


def run_learning_experiment():  # type: ignore
    print("Starting SC Learning Experiment (STDP)...")

    # 10 inputs, 2 neurons
    layer = SCLearningLayer(n_inputs=10, n_neurons=2, learning_rate=0.05, length=512)

    # Pattern A: High activity in first 5 inputs
    pattern_a = [0.8, 0.8, 0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1]
    # Pattern B: High activity in last 5 inputs
    pattern_b = [0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8, 0.8]

    epochs = 20
    weights_history = []

    for epoch in range(epochs):
        # Alternate patterns
        if epoch % 2 == 0:
            target = "A"
            layer.run_epoch(pattern_a)
        else:
            target = "B"
            layer.run_epoch(pattern_b)

        w = layer.get_weights()
        weights_history.append(w.copy())
        print(f"Epoch {epoch} ({target}) finished.")

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    wh = np.array(weights_history)  # (epochs, n_neurons, n_inputs)

    for i in range(layer.n_neurons):
        axes[i].plot(wh[:, i, :])
        axes[i].set_title(f"Neuron {i} Weights Evolution")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Weight Value")

    plt.tight_layout()
    plt.savefig("stdp_learning_result.png")
    print("Results saved to stdp_learning_result.png")


if __name__ == "__main__":
    run_learning_experiment()  # type: ignore
