# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Example 02: Building and Running SC Neuron Layers

#!/usr/bin/env python3
"""Example 02: Building and Running SC Neuron Layers."""

import numpy as np
from sc_neurocore import SCDenseLayer


def main():
    print("=== SC-NeuroCore: Dense Layer Demo ===\n")

    n_neurons = 3
    length = 512

    # Generate random input probabilities in [0, 1]
    input_probs = [0.2, 0.5, 0.8, 0.4]
    weight_values = [1.0, 1.0, 1.0, 1.0]
    print(f"Input probabilities: {input_probs}")
    print(f"Weight values: {weight_values}")

    layer = SCDenseLayer(
        n_neurons=n_neurons,
        x_inputs=input_probs,
        weight_values=weight_values,
        x_min=0.0,
        x_max=1.0,
        w_min=0.0,
        w_max=1.0,
        length=length,
        neuron_params={"noise_std": 0.0, "v_threshold": 0.5, "resistance": 3.0},
        base_seed=42,
    )

    # Run several time steps and accumulate firing rates
    n_steps = 20
    layer.run(n_steps)
    rates = layer.get_spike_trains().mean(axis=1)
    print(f"Average firing rates over {n_steps} steps: {rates}")
    print(f"Summary: {layer.summary()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
