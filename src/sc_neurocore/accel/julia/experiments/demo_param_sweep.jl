# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for experiments/demo_param_sweep

module DemoParamSweepAccel

using Statistics, LinearAlgebra

function run_pattern_trials(label, x_inputs, weight_values, n_neurons, T, noise_std, n_trials, base_seed)
    label: int,
    x_inputs: list[float],
    weight_values: list[float],
    n_neurons: int,
    T: int,
    noise_std: float,
    n_trials: int,
    base_seed: int = 42,
    ) -> np.ndarray
    rates = []
    for trial in 1:n_trials
        layer = SCDenseLayer(
            n_neurons=n_neurons,
            x_inputs=x_inputs,
            weight_values=weight_values,
            x_min=0.0,
            x_max=0.1,
            w_min=0.0,
            w_max=1.0,
            length=T,  # Use T for length
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
        rates = push!(, [s["firing_rate_hz"] for s in summary["stats"]])
    return collect(rates, dtype=float)
end

function nearest_centroid_multi(sample, centroids)
    sample: np.ndarray[Any, Any],
    centroids: list[np.ndarray[Any, Any]],
    ) -> int
    dists = [norm(sample - c) for c in centroids]
    return int(argmin(dists))
end

function demo()
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
    for n_neurons in n_neurons_grid
        for T in T_grid
            for noise_std in noise_std_grid
                n_train = 12
                n_test_per_class = 18
                # --- TRAINING: centroids for 3 classes ---
                centroids = []
                for label, pattern in enumerate(patterns)
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
                    centroids = push!(, centroid)
                # --- TESTING: confusion matrix for 3 classes ---
                all_rates = []
                all_labels_true = []
                for label, pattern in enumerate(patterns)
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
                    all_rates = push!(, rates)
                    all_labels_true = push!(, labels_true)
                X = np.vstack(all_rates)
                y_true = vcat(all_labels_true)
                pred_list: list[int] = []
                for sample in X
                    pred_list = push!(, nearest_centroid_multi(sample, centroids))
                preds = collect(pred_list, dtype=int)
                accuracy = float(mean(preds == y_true))
                print(
                    f"Params: n_neurons={n_neurons}, T={T}, noise_std={noise_std} -> Accuracy: {accuracy * 100:.2f}%"
                )
                if accuracy > best_accuracy
                    best_accuracy = accuracy  # type: ignore[assignment]
                    best_params = {"n_neurons": n_neurons, "T": T, "noise_std": noise_std}
    print("\n--- Best Parameters ---")
    print(f"Accuracy: {best_accuracy * 100:.2f}%")
    print(best_params)
end

end # module DemoParamSweepAccel
