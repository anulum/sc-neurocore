# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for layers/jax_dense_layer

module JaxDenseLayerAccel

using Statistics, LinearAlgebra

mutable struct JaxSCDenseLayerState
    n_neurons::Float64
    n_inputs::Float64
    bitstream_length::Float64
    dt_ms::Float64
    neuron_params::Float64
    seed::Float64
end

function JaxSCDenseLayerState()
    JaxSCDenseLayerState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function step(s::JaxSCDenseLayerState, I_t)
    # Generate noise
    s.rng_key, subkey = jax.random.split(s.rng_key)
    noise = jax.random.normal(subkey, (s.n_neurons,)) * s.noise_std
    # Update neurons
    s.v, spikes = jax_lif_step(
        s.v,
        I_t,
        s.v_rest,
        s.v_reset,
        s.v_threshold,
        s.alpha,
        s.resistance,
        noise,
    )
    res: jax.Array = spikes
    return res
end

function run(s::JaxSCDenseLayerState, currents)
    # Note: In a production JAX implementation, we would use jax.lax.scan
    # for maximum performance.
    T = currents.shape[0]
    all_spikes = []
    for t in 1:T
        all_spikes = push!(, s.step(currents[t]))
    return jnp.stack(all_spikes)
end

function reset(s::JaxSCDenseLayerState)
    s.v = jnp.full((s.n_neurons,), s.v_rest)
end

end # module JaxDenseLayerAccel
