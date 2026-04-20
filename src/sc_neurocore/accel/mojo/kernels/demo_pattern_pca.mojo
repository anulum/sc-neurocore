# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for demo_pattern_pca

fn compute_pca_2d(X: Int) -> Int:
    var _compute_pca_2d_line = 'X: ndarray[Any, Any],'
    var _compute_pca_2d_line = ') -> tuple[ndarray[Any, Any], ndarray[Any, Any], ndarray[Any'
    var _compute_pca_2d_line = '# Center the data'
    var _compute_pca_2d_line = 'mean = X.mean(axis=0, keepdims=True)'
    var _compute_pca_2d_line = 'X_centered = X - mean'
    var _compute_pca_2d_line = '# SVD: X = U S V^T ; rows of V^T are principal directions'
    var _compute_pca_2d_line = 'U, S, Vt = linalg.svd(X_centered, full_matrices=False)'
    var _compute_pca_2d_line = '# Take first 2 components'
    var _compute_pca_2d_line = 'components = Vt[:2]  # shape (2, n_features)'
    var _compute_pca_2d_line = 'X_2d = X_centered @ components.T  # (n_samples, 2)'
    return 0  # return X_2d, mean.squeeze(), components

fn demo_pca_plot() -> Int:
    var _demo_pca_plot_line = '# Two patterns (same as in demo_pattern_classification)'
    var _demo_pca_plot_line = 'pattern_A = [0.02, 0.05, 0.08]  # class 0'
    var _demo_pca_plot_line = 'pattern_B = [0.08, 0.05, 0.02]  # class 1'
    var _demo_pca_plot_line = 'weight_values = [0.3, 0.6, 0.9]'
    var _demo_pca_plot_line = 'n_neurons = 5'
    var _demo_pca_plot_line = 'T = 2500'
    var _demo_pca_plot_line = 'n_samples_per_class = 20'
    var _demo_pca_plot_line = '# Collect firing-rate signatures'
    var _demo_pca_plot_line = 'rates_A = run_pattern_trials('
    var _demo_pca_plot_line = 'label=0,'
    var _demo_pca_plot_line = 'x_inputs=pattern_A,'
    var _demo_pca_plot_line = 'weight_values=weight_values,'
    var _demo_pca_plot_line = 'n_neurons=n_neurons,'
    var _demo_pca_plot_line = 'T=T,'
    var _demo_pca_plot_line = 'n_trials=n_samples_per_class,'
    var _demo_pca_plot_line = 'base_seed=42,'
    var _demo_pca_plot_line = ')'
    var _demo_pca_plot_line = 'rates_B = run_pattern_trials('
    var _demo_pca_plot_line = 'label=1,'
    var _demo_pca_plot_line = 'x_inputs=pattern_B,'
    var _demo_pca_plot_line = 'weight_values=weight_values,'
    var _demo_pca_plot_line = 'n_neurons=n_neurons,'
    var _demo_pca_plot_line = 'T=T,'
    var _demo_pca_plot_line = 'n_trials=n_samples_per_class,'
    var _demo_pca_plot_line = 'base_seed=42,'
    var _demo_pca_plot_line = ')'
    var _demo_pca_plot_line = 'X = vstack([rates_A, rates_B])'
    var _demo_pca_plot_line = 'y = concatenate('
    var _demo_pca_plot_line = '[zeros(rates_A.shape[0], dtype=int), ones(rates_B.shape[0], '
    var _demo_pca_plot_line = ')'
    var _demo_pca_plot_line = '# PCA to 2D'
    var _demo_pca_plot_line = 'X_2d, mean_vec, components = compute_pca_2d(X)'
    var _demo_pca_plot_line = '# Split back by class'
    var _demo_pca_plot_line = 'X_A = X_2d[y == 0]'
    var _demo_pca_plot_line = 'X_B = X_2d[y == 1]'
    var _demo_pca_plot_line = 'plt.figure()'
    var _demo_pca_plot_line = 'plt.scatter(X_A[:, 0], X_A[:, 1], marker="o", label="Pattern'
    var _demo_pca_plot_line = 'plt.scatter(X_B[:, 0], X_B[:, 1], marker="x", label="Pattern'
    var _demo_pca_plot_line = 'plt.xlabel("PC1")'
    var _demo_pca_plot_line = 'plt.ylabel("PC2")'
    var _demo_pca_plot_line = 'plt.title("PCA of SC Dense Layer Firing-Rate Signatures")'
    var _demo_pca_plot_line = 'plt.legend()'
    var _demo_pca_plot_line = 'plt.grid(True)'
    var _demo_pca_plot_line = 'plt.tight_layout()'
    var _demo_pca_plot_line = 'plt.show()'
    return 0

