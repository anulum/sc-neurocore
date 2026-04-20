# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for adapters_holonomic/l4_cell

module L4CellAccel

using Statistics, LinearAlgebra

mutable struct L4_CellularAdapterState
    n_cells::Float64
    bitstream_length::Float64
    omega_mean::Float64
    k_coupling::Float64
    sigma_noise::Float64
    critical_threshold::Float64
    rng_key::Float64
    phases::Float64
    avalanches::Float64
end

function L4_CellularAdapterState()
    L4_CellularAdapterState(400.0, 1024.0, 1.0, 0.3, 0.1, 0.6, 0.0, 0.0, 0.0)
end

function encode(s::L4_CellularAdapterState, domain_state)
    # Activity = (1 + cos(phase)) / 2
    activity = (1.0 + jcos(s.phases)) / 2.0
    s.rng_key, subkey = split_rng(s.rng_key)
    rands = uniform(subkey, (s.params.n_cells, s.params.bitstream_length))
    bitstreams = (rands < activity[:, nothing]).astype(jnp.uint8)
    return bitstreams
end

function _kuramoto_kernel(s::L4_CellularAdapterState)
    phases: jnp.ndarray, omega: float, k: float, dt: float, noise: jnp.ndarray
    ) -> jnp.ndarray
    n = phases.shape[0]
    # Calculate all-to-all coupling (can be optimized with neighbor masks later)
    diffs = phases[nothing, :] - phases[:, nothing]
    coupling = (k / n) * jsum(jsin(diffs), axis=1)
    d_phase = (2 * jpi * omega + coupling + noise) * dt
    return (phases + d_phase) % (2 * jpi)
end

function step_jax(s::L4_CellularAdapterState, dt, inputs)
    # 1. Generate Noise
    s.rng_key, subkey = split_rng(s.rng_key)
    noise = normal(subkey, (s.params.n_cells,)) * s.params.sigma_noise
    # 2. Update Phases via Kuramoto Kernel
    s.phases = s._kuramoto_kernel(
        s.phases, s.params.omega_mean, s.params.k_coupling, dt, noise
    )
    # 3. Model Avalanche Dynamics (Criticality readout)
    # If mean activity crosses threshold, ignition occurs
    mean_activity = jmean((1.0 + jcos(s.phases)) / 2.0)
    ignition = (mean_activity > s.params.critical_threshold).astype(jnp.float32)
    s.avalanches = 0.9 * s.avalanches + 0.1 * ignition
    # 4. Return encoded bitstreams
    return s.encode(nothing)
end

function decode(s::L4_CellularAdapterState, bitstreams)
    # Complex order parameter R = |1/N * sum(exp(i*theta))|
    # Approximated from bitstream means
    return {"synchronization_r4": float(jabs(jmean(jexp(1j * s.phases))))}
end

function get_metrics(s::L4_CellularAdapterState)
    return {
        "order_parameter": float(jabs(jmean(jexp(1j * s.phases)))),
        "avalanche_density": float(jmean(s.avalanches)),
    }
end

end # module L4CellAccel
