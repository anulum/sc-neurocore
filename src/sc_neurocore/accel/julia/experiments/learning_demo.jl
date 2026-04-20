# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for experiments/learning_demo

module LearningDemoAccel

using Statistics, LinearAlgebra

function run_learning_experiment()
    print("Starting SC Learning Experiment (STDP)...")
    # 10 inputs, 2 neurons
    layer = SCLearningLayer(n_inputs=10, n_neurons=2, learning_rate=0.05, length=512)
    # Pattern A: High activity in first 5 inputs
    pattern_a = [0.8, 0.8, 0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1]
    # Pattern B: High activity in last 5 inputs
    pattern_b = [0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8, 0.8]
    epochs = 20
    weights_history = []
    for epoch in 1:epochs
        # Alternate patterns
        if epoch % 2 == 0
            target = "A"
            layer.run_epoch(pattern_a)
        else
            target = "B"
            layer.run_epoch(pattern_b)
        w = layer.get_weights()
        weights_history = push!(, w.copy())
        print(f"Epoch {epoch} ({target}) finished.")
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    wh = collect(weights_history)  # (epochs, n_neurons, n_inputs)
    for i in 1:layer.n_neurons
        axes[i].plot(wh[:, i, :])
        axes[i].set_title(f"Neuron {i} Weights Evolution")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Weight Value")
    plt.tight_layout()
    plt.savefig("stdp_learning_result.png")
    print("Results saved to stdp_learning_result.png")
end

end # module LearningDemoAccel
