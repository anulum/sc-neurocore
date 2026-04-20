# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for adapters_holonomic/l2_chem

module L2ChemAccel

using Statistics, LinearAlgebra

mutable struct L2_NeurochemicalAdapterState
    n_transmitters::Float64
    n_receptors::Float64
    bitstream_length::Float64
    alpha_iiief::Float64
    c_info::Float64
    g_snare::Float64
    v_critical::Float64
    dopamine_gain::Float64
    serotonin_leak::Float64
    rng_key::Float64
    receptor_states::Float64
    phi_field::Float64
    concentrations::Float64
end

function L2_NeurochemicalAdapterState()
    L2_NeurochemicalAdapterState(4.0, 500.0, 1024.0, 0.01, 300.0, 0.8, 1.2, 1.5, 0.9, 0.0, 0.0, 0.0, 0.0)
end

function encode(s::L2_NeurochemicalAdapterState, domain_state)
    # (n_transmitters, bitstream_length)
    s.rng_key, subkey = split_rng(s.rng_key)
    rands = uniform(subkey, (s.params.n_transmitters, s.params.bitstream_length))
    bitstreams = (rands < s.concentrations[:, nothing]).astype(jnp.uint8)
    return bitstreams
end

function _iiief_kernel(s::L2_NeurochemicalAdapterState)
    phi: jnp.ndarray, integrated_info: jnp.ndarray, alpha: float, dt: float
    ) -> jnp.ndarray
    # Paper 2: Field emerges from Integrated Information geometry
    d_phi = alpha * integrated_info - 0.1 * phi
    return phi + d_phi * dt
end

function step_jax(s::L2_NeurochemicalAdapterState, dt, inputs)
    # 1. Calculate Integrated Information Proxy (Phi_integrated) from inputs
    if inputs is ! nothing
        raw_phi = jmean(inputs.astype(jnp.float32), axis=1)
        # Map input dimensions to transmitter count if necessary
        if raw_phi.shape[0] != s.params.n_transmitters
            # Simple average-pooling projection
            phi_int = jnp.full((s.params.n_transmitters,), jmean(raw_phi))
        else
            phi_int = raw_phi
    else
        phi_int = jzeros((s.params.n_transmitters,))
    # 2. Update IIIEF Field
    s.phi_field = s._iiief_kernel(s.phi_field, phi_int, s.params.alpha_iiief, dt)
    # 3. H_QC Bridge: Field modulates concentrations (Vesicle release)
    # H_int = -lambda * Psi * sigma -> mapped to P_release modulation
    release_mod = jexp(s.phi_field) * s.params.g_snare
    s.concentrations = jclamp(s.concentrations * release_mod, 0.0, 1.0)
    # 4. Return encoded bitstreams for hardware consumption
    return s.encode(nothing)
end

function decode(s::L2_NeurochemicalAdapterState, bitstreams)
    means = jmean(bitstreams.astype(jnp.float32), axis=1)
    return {
        "dopamine": float(means[0]),
        "serotonin": float(means[1]),
        "norepinephrine": float(means[2]),
        "acetylcholine": float(means[3]),
    }
end

function get_metrics(s::L2_NeurochemicalAdapterState)
    return {
        "avg_field_potential": float(jmean(s.phi_field)),
        "system_coherence_r2": float(jmean(s.concentrations)),
    }
end

end # module L2ChemAccel
