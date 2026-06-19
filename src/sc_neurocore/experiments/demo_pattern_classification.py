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
    n_neurons: int = 5,
    T: int = 2500,
    n_trials: int = 10,
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


def nearest_centroid_classify(
    sample: np.ndarray[Any, Any],
    centroid_A: np.ndarray[Any, Any],
    centroid_B: np.ndarray[Any, Any],
) -> int:
    """
    Simple nearest-centroid classifier in firing-rate space.
    Returns label 0 or 1.
    """
    dA = np.linalg.norm(sample - centroid_A)
    dB = np.linalg.norm(sample - centroid_B)
    return 0 if dA <= dB else 1


def demo() -> None:
    # Two different input patterns (3-channel) for the SC layer
    pattern_A = [0.02, 0.05, 0.08]  # e.g. "class 0"
    pattern_B = [0.08, 0.05, 0.02]  # e.g. "class 1" (reordered intensities)

    # Shared weights for simplicity
    weight_values = [0.3, 0.6, 0.9]

    n_neurons = 5
    T_train = 2500
    T_test = 2500
    n_train = 10
    n_test = 20  # 10 samples per class

    # --- TRAINING: collect firing-rate signatures for each pattern ---
    rates_A = run_pattern_trials(
        label=0,
        x_inputs=pattern_A,
        weight_values=weight_values,
        n_neurons=n_neurons,
        T=T_train,
        n_trials=n_train,
        base_seed=42,
    )
    rates_B = run_pattern_trials(
        label=1,
        x_inputs=pattern_B,
        weight_values=weight_values,
        n_neurons=n_neurons,
        T=T_train,
        n_trials=n_train,
        base_seed=42,
    )

    centroid_A = rates_A.mean(axis=0)
    centroid_B = rates_B.mean(axis=0)

    print("Centroid A (firing rates per neuron):", centroid_A)
    print("Centroid B (firing rates per neuron):", centroid_B)

    # --- TESTING: generate new samples and classify by nearest centroid ---
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

    test_A, labels_A = gen_test_samples(0, pattern_A, n_samples=n_test // 2)
    test_B, labels_B = gen_test_samples(1, pattern_B, n_samples=n_test // 2)

    test_all = np.vstack([test_A, test_B])
    labels_true = np.concatenate([labels_A, labels_B])

    pred_list: list[int] = []
    for sample in test_all:
        y = nearest_centroid_classify(sample, centroid_A, centroid_B)
        pred_list.append(y)
    preds = np.array(pred_list, dtype=int)

    accuracy = float((preds == labels_true).mean())
    print(f"\nTest accuracy (nearest-centroid in firing-rate space): {accuracy * 100:.1f}%")
    print("True labels: ", labels_true)
    print("Predicted:   ", preds)


if __name__ == "__main__":
    demo()
