#!/usr/bin/env python3
"""Example 02: Building and Running SC Neuron Layers.

Demonstrates constructing an SCDenseLayer, feeding it input probabilities,
and reading output firing rates.
"""

import numpy as np
from sc_neurocore import SCDenseLayer, StochasticLIFNeuron, BitstreamSynapse

def main():
    print("=== SC-NeuroCore: Dense Layer Demo ===\n")

    n_inputs = 4
    n_neurons = 3
    length = 512

    layer = SCDenseLayer(n_inputs=n_inputs, n_neurons=n_neurons, length=length)

    # Generate random input probabilities in [0, 1]
    input_probs = [0.2, 0.5, 0.8, 0.4]
    print(f"Input probabilities: {input_probs}")

    # Run several time steps and accumulate firing rates
    rates = np.zeros(n_neurons)
    n_steps = 20
    for step in range(n_steps):
        spikes = layer.run_epoch(np.array(input_probs))
        rates += spikes.mean(axis=1) if spikes.ndim > 1 else spikes

    rates /= n_steps
    print(f"Average firing rates over {n_steps} steps: {rates}")
    print("\nDone.")


if __name__ == "__main__":
    main()
