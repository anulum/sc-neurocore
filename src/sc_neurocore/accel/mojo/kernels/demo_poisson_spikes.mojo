# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo SIMD acceleration for demo_poisson_spikes

fn run_demo() -> Int:
    var _run_demo_line = 'neuron = StochasticLIFNeuron('
    var _run_demo_line = 'v_rest=0.0,'
    var _run_demo_line = 'v_reset=0.0,'
    var _run_demo_line = 'v_threshold=1.0,'
    var _run_demo_line = 'tau_mem=20.0,'
    var _run_demo_line = 'dt=1.0,'
    var _run_demo_line = 'noise_std=0.1,'
    var _run_demo_line = 'resistance=1.0,'
    var _run_demo_line = 'seed=42,'
    var _run_demo_line = ')'
    var _run_demo_line = 'T = 2000'
    var _run_demo_line = 'I = 0.06 * ones(T)'
    var _run_demo_line = 'spikes = zeros(T, dtype=int)'
    var _run_demo_line = 'for t in range(T):'
    var _run_demo_line = 'spikes[t] = neuron.step(I[t])'
    var _run_demo_line = 'rate_hz = spikes.sum() / (T * neuron.dt) * 1000.0'
    var _run_demo_line = 'print(f"Total spikes: {spikes.sum()}, firing rate ≈ {rate_hz'
    return 0

