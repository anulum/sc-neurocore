# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for learning_demo

fn run_learning_experiment() -> Int:
    var _run_learning_experiment_line = 'print("Starting SC Learning Experiment (STDP)...")'
    var _run_learning_experiment_line = '# 10 inputs, 2 neurons'
    var _run_learning_experiment_line = 'layer = SCLearningLayer(n_inputs=10, n_neurons=2, learning_r'
    var _run_learning_experiment_line = '# Pattern A: High activity in first 5 inputs'
    var _run_learning_experiment_line = 'pattern_a = [0.8, 0.8, 0.8, 0.8, 0.8, 0.1, 0.1, 0.1, 0.1, 0.'
    var _run_learning_experiment_line = '# Pattern B: High activity in last 5 inputs'
    var _run_learning_experiment_line = 'pattern_b = [0.1, 0.1, 0.1, 0.1, 0.1, 0.8, 0.8, 0.8, 0.8, 0.'
    var _run_learning_experiment_line = 'epochs = 20'
    var _run_learning_experiment_line = 'weights_history = []'
    var _run_learning_experiment_line = 'for epoch in range(epochs):'
    var _run_learning_experiment_line = '# Alternate patterns'
    var _run_learning_experiment_line = 'if epoch % 2 == 0:'
    var _run_learning_experiment_line = 'target = "A"'
    var _run_learning_experiment_line = 'layer.run_epoch(pattern_a)'
    var _run_learning_experiment_line = 'else:'
    var _run_learning_experiment_line = 'target = "B"'
    var _run_learning_experiment_line = 'layer.run_epoch(pattern_b)'
    var _run_learning_experiment_line = 'w = layer.get_weights()'
    var _run_learning_experiment_line = 'weights_history.append(w.copy())'
    var _run_learning_experiment_line = 'print(f"Epoch {epoch} ({target}) finished.")'
    var _run_learning_experiment_line = '# Plot results'
    var _run_learning_experiment_line = 'fig, axes = plt.subplots(1, 2, figsize=(12, 5))'
    var _run_learning_experiment_line = 'wh = array(weights_history)  # (epochs, n_neurons, n_inputs)'
    var _run_learning_experiment_line = 'for i in range(layer.n_neurons):'
    var _run_learning_experiment_line = 'axes[i].plot(wh[:, i, :])'
    var _run_learning_experiment_line = 'axes[i].set_title(f"Neuron {i} Weights Evolution")'
    var _run_learning_experiment_line = 'axes[i].set_xlabel("Epoch")'
    var _run_learning_experiment_line = 'axes[i].set_ylabel("Weight Value")'
    var _run_learning_experiment_line = 'plt.tight_layout()'
    var _run_learning_experiment_line = 'plt.savefig("stdp_learning_result.png")'
    var _run_learning_experiment_line = 'print("Results saved to stdp_learning_result.png")'
    return 0

