from __future__ import annotations
import numpy as np
from sc_neurocore.layers.sc_dense_layer import SCDenseLayer

def run_pattern_trials(
    label: int,
    x_inputs,
    weight_values,
    n_neurons: int,
    T: int,
    noise_std: float,
    n_trials: int,
    base_seed: int = 42,
):
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
            length=T, # Use T for length
            y_min=0.0,
            y_max=0.1,
            dt_ms=1.0,
            neuron_params=dict(
                v_rest=0.0,
                v_reset=0.0,
                v_threshold=1.0,
                tau_mem=25.0,
                noise_std=noise_std,
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
    sample: np.ndarray,
    centroids: list[np.ndarray],
) -> int:
    """
    Nearest-centroid classifier over K classes.
    centroids[k]: firing-rate centroid for class k.
    """
    dists = [np.linalg.norm(sample - c) for c in centroids]
    return int(np.argmin(dists))

def demo():
    # Three different 3-channel patterns
    pattern_A = [0.02, 0.05, 0.08]  # class 0
    pattern_B = [0.08, 0.05, 0.02]  # class 1
    pattern_C = [0.04, 0.09, 0.01]  # class 2 (distinct mix)
    weight_values = [0.3, 0.6, 0.9]
    patterns = [pattern_A, pattern_B, pattern_C]
    
    # Grid search parameters
    n_neurons_grid = [5, 7, 10]
    T_grid = [2500, 3000, 4000]
    noise_std_grid = [0.03, 0.05, 0.07]

    best_accuracy = 0
    best_params = {}

    for n_neurons in n_neurons_grid:
        for T in T_grid:
            for noise_std in noise_std_grid:
                
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
                        T=T,
                        noise_std=noise_std,
                        n_trials=n_train,
                        base_seed=42,
                    )
                    centroid = rates.mean(axis=0)
                    centroids.append(centroid)

                # --- TESTING: confusion matrix for 3 classes ---
                all_rates = []
                all_labels_true = []
                for label, pattern in enumerate(patterns):
                    rates, labels_true = run_pattern_trials(
                        label=label,
                        x_inputs=pattern,
                        weight_values=weight_values,
                        n_neurons=n_neurons,
                        T=T,
                        noise_std=noise_std,
                        n_trials=n_test_per_class,
                        base_seed=999,
                    ), np.full((n_test_per_class,), label, dtype=int)
                    all_rates.append(rates)
                    all_labels_true.append(labels_true)

                X = np.vstack(all_rates)
                y_true = np.concatenate(all_labels_true)

                preds = []
                for sample in X:
                    preds.append(nearest_centroid_multi(sample, centroids))
                preds = np.array(preds, dtype=int)

                accuracy = float((preds == y_true).mean())
                
                print(f"Params: n_neurons={n_neurons}, T={T}, noise_std={noise_std} -> Accuracy: {accuracy * 100:.2f}%")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {'n_neurons': n_neurons, 'T': T, 'noise_std': noise_std}

    print("\n--- Best Parameters ---")
    print(f"Accuracy: {best_accuracy * 100:.2f}%")
    print(best_params)

if __name__ == "__main__":
    demo()
