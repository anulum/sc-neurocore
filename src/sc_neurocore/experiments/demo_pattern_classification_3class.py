# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Run multiple trials of SCDenseLayer for a given pattern

from __future__ import annotations
from typing import Any
import numpy as np
from sc_neurocore.layers.sc_dense_layer import SCDenseLayer


def run_pattern_trials(
    label: int,
    x_inputs: list[float],
    weight_values: list[float],
    n_neurons: int,
    T: int,
    n_trials: int,
    base_seed: int = 42,
) -> np.ndarray[Any, Any]:
    """
    Run multiple trials of SCDenseLayer for a given pattern (x_inputs).
    Return matrix of shape (n_trials, n_neurons) with firing rates.
    """
    rates = []
    for trial in range(n_trials):
        layer = SCDenseLayer(
            n_neurons=n_neurons,
            x_inputs=x_inputs,
            weight_values=weight_values,
            x_min=0.0,
            x_max=0.1,
            w_min=0.0,
            w_max=1.0,
            length=4096,
            y_min=0.0,
            y_max=0.1,
            dt_ms=1.0,
            neuron_params=dict(
                v_rest=0.0,
                v_reset=0.0,
                v_threshold=1.0,
                tau_mem=25.0,
                noise_std=0.03,
                resistance=1.0,
            ),
            base_seed=base_seed + label * 1000 + trial,
        )
        layer.reset()
        layer.run(T)
        summary = layer.summary()
        rates.append([s["firing_rate_hz"] for s in summary["stats"]])
    return np.array(rates, dtype=float)


def nearest_centroid_multi(
    sample: np.ndarray[Any, Any],
    centroids: list[np.ndarray[Any, Any]],
) -> int:
    """
    Nearest-centroid classifier over K classes.
    centroids[k]: firing-rate centroid for class k.
    """
    dists = [np.linalg.norm(sample - c) for c in centroids]
    return int(np.argmin(dists))


def demo() -> None:
    # Three different 3-channel patterns
    pattern_A = [0.02, 0.05, 0.08]  # class 0
    pattern_B = [0.08, 0.05, 0.02]  # class 1
    pattern_C = [0.04, 0.09, 0.01]  # class 2 (distinct mix)
    weight_values = [0.3, 0.6, 0.9]
    patterns = [pattern_A, pattern_B, pattern_C]

    n_neurons = 5
    T_train = 2500
    T_test = 2500
    n_train = 12
    n_test_per_class = 18

    # --- TRAINING: centroids for 3 classes ---
    centroids = []
    for label, pattern in enumerate(patterns):
        rates = run_pattern_trials(
            label=label,
            x_inputs=pattern,
            weight_values=weight_values,
            n_neurons=n_neurons,
            T=T_train,
            n_trials=n_train,
            base_seed=42,
        )
        centroid = rates.mean(axis=0)
        centroids.append(centroid)
        print(f"Centroid class {label}: {centroid}")

    # --- TESTING: confusion matrix for 3 classes ---
    def gen_test_samples(
        label: int, pattern: list[float], n_samples: int
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        rates = run_pattern_trials(
            label=label,
            x_inputs=pattern,
            weight_values=weight_values,
            n_neurons=n_neurons,
            T=T_test,
            n_trials=n_samples,
            base_seed=999,
        )
        labels_true = np.full((n_samples,), label, dtype=int)
        return rates, labels_true

    all_rates = []
    all_labels_true = []
    for label, pattern in enumerate(patterns):
        rates, labels_true = gen_test_samples(label, pattern, n_test_per_class)
        all_rates.append(rates)
        all_labels_true.append(labels_true)

    X = np.vstack(all_rates)
    y_true = np.concatenate(all_labels_true)

    pred_list: list[int] = []
    for sample in X:
        pred_list.append(nearest_centroid_multi(sample, centroids))
    preds = np.array(pred_list, dtype=int)

    accuracy = float((preds == y_true).mean())

    # Confusion matrix
    K = 3
    conf_mat = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, preds):
        conf_mat[t, p] += 1

    print(f"\nOverall accuracy (3-class): {accuracy * 100:.1f}%")
    print("Confusion matrix (rows=true, cols=pred):")
    print(conf_mat)
    print("True labels: ", y_true)
    print("Predicted:   ", preds)


if __name__ == "__main__":
    demo()
