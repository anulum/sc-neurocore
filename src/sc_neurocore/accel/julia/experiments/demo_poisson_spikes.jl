# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for experiments/demo_poisson_spikes

module DemoPoissonSpikesAccel

using Statistics, LinearAlgebra

function run_demo()
    neuron = StochasticLIFNeuron(
        v_rest=0.0,
        v_reset=0.0,
        v_threshold=1.0,
        tau_mem=20.0,
        dt=1.0,
        noise_std=0.1,
        resistance=1.0,
        seed=42,
    )
    T = 2000
    I = 0.06 * ones(T)
    spikes = zeros(T, dtype=int)
    for t in 1:T
        spikes[t] = neuron.step(I[t])
    rate_hz = spikes.sum() / (T * neuron.dt) * 1000.0
    print(f"Total spikes: {spikes.sum()}, firing rate ≈ {rate_hz:.2f} Hz")
end

end # module DemoPoissonSpikesAccel
