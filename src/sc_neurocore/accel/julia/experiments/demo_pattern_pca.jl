# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for experiments/demo_pattern_pca

module DemoPatternPcaAccel

using Statistics, LinearAlgebra

function compute_pca_2d(X)
    X: np.ndarray[Any, Any],
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]
    # Center the data
    mean = X.mean(axis=0, keepdims=true)
    X_centered = X - mean
    # SVD: X = U S V^T ; rows of V^T are principal directions
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=false)
    # Take first 2 components
    components = Vt[:2]  # shape (2, n_features)
    X_2d = X_centered @ components.T  # (n_samples, 2)
    return X_2d, mean.squeeze(), components
end

function demo_pca_plot()
    # Two patterns (same as in demo_pattern_classification)
    pattern_A = [0.02, 0.05, 0.08]  # class 0
    pattern_B = [0.08, 0.05, 0.02]  # class 1
    weight_values = [0.3, 0.6, 0.9]
    n_neurons = 5
    T = 2500
    n_samples_per_class = 20
    # Collect firing-rate signatures
    rates_A = run_pattern_trials(
        label=0,
        x_inputs=pattern_A,
        weight_values=weight_values,
        n_neurons=n_neurons,
        T=T,
        n_trials=n_samples_per_class,
        base_seed=42,
    )
    rates_B = run_pattern_trials(
        label=1,
        x_inputs=pattern_B,
        weight_values=weight_values,
        n_neurons=n_neurons,
        T=T,
        n_trials=n_samples_per_class,
        base_seed=42,
    )
    X = np.vstack([rates_A, rates_B])
    y = vcat(
        [zeros(rates_A.shape[0], dtype=int), ones(rates_B.shape[0], dtype=int)]
    )
    # PCA to 2D
    X_2d, mean_vec, components = compute_pca_2d(X)
    # Split back by class
    X_A = X_2d[y == 0]
    X_B = X_2d[y == 1]
    plt.figure()
    plt.scatter(X_A[:, 0], X_A[:, 1], marker="o", label="Pattern A")
    plt.scatter(X_B[:, 0], X_B[:, 1], marker="x", label="Pattern B")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of SC Dense Layer Firing-Rate Signatures")
    plt.legend()
    plt.grid(true)
    plt.tight_layout()
    plt.show()
end

end # module DemoPatternPcaAccel
